from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from minisgl.afd_fmha_runtime import (
    AfdFmhaRuntime,
    _FabricRecord,
    _stream_identity,
)
from minisgl.kernel.fabric_memory import FabricTensor


class _Context:
    def __init__(self) -> None:
        self.moe_num_token_non_padded = None
        self.moe_deepep_dispatch_max_tokens_per_rank = None
        self.mlp_deepep_buffer = None
        self.attn_backend = None
        self._forward_batch_active = False

    @contextlib.contextmanager
    def forward_batch(self, batch):
        if self._forward_batch_active:
            raise AssertionError("Nested forward_batch is not allowed")
        self._forward_batch_active = True
        try:
            yield
        finally:
            self._forward_batch_active = False


class _Layer:
    def __init__(
        self,
        layer_id: int,
        calls: list[tuple],
        *,
        lane_observations: list[tuple] | None = None,
        active_stream: list[str] | None = None,
        ctx: _Context | None = None,
        tp_plugin=None,
    ) -> None:
        self.layer_id = layer_id
        self.calls = calls
        self.lane_observations = lane_observations
        self.active_stream = active_stream
        self.ctx = ctx
        self.tp_plugin = tp_plugin
        self.self_attn = SimpleNamespace(
            finish_attention=self.finish_attention,
            finish_attention_fp8=self.finish_attention_fp8,
        )
        self.post_attention_layernorm = SimpleNamespace(forward=self.post_norm)
        self.mlp = SimpleNamespace(
            forward=self.run_mlp,
            prepare_deepep=self.prepare_deepep,
            finish_deepep=self.finish_deepep,
            dispatch_deepep=self.dispatch_deepep,
            run_deepep_experts=self.run_deepep_experts,
            combine_deepep=self.combine_deepep,
        )

    def prepare_attention(self, hidden, residual):
        self.calls.append(("prepare", self.layer_id, int(hidden.shape[0])))
        if self.lane_observations is not None:
            self.lane_observations.append(
                ("prepare", self.layer_id, self.active_stream[0])
            )
        if residual is None:
            residual = torch.zeros_like(hidden)
        return hidden, residual

    def finish_attention(self, output):
        self.calls.append(("finish", self.layer_id, int(output.shape[0])))
        if self.lane_observations is not None:
            self.lane_observations.append(
                ("finish", self.layer_id, self.active_stream[0])
            )
        return output

    def finish_attention_fp8(self, output, scale):
        return self.finish_attention(output)

    def post_norm(self, hidden, residual):
        self.calls.append(("post_norm", self.layer_id, int(hidden.shape[0])))
        if self.lane_observations is not None:
            self.lane_observations.append(
                ("post_norm", self.layer_id, self.active_stream[0])
            )
        return hidden, residual

    def run_mlp(self, hidden):
        self.calls.append(("mlp", self.layer_id, int(hidden.shape[0])))
        if self.lane_observations is not None:
            assert self.active_stream is not None and self.ctx is not None
            self.lane_observations.append(
                (
                    "mlp",
                    self.layer_id,
                    self.active_stream[0],
                    self.ctx.mlp_deepep_buffer,
                    None if self.tp_plugin is None else self.tp_plugin.comm,
                )
            )
        return hidden

    def prepare_deepep(self, hidden):
        assert self.active_stream is not None and self.ctx is not None
        self.lane_observations.append(
            (
                "pre_moe",
                self.layer_id,
                self.active_stream[0],
                self.ctx.mlp_deepep_buffer,
                None if self.tp_plugin is None else self.tp_plugin.comm,
            )
        )
        return hidden

    def finish_deepep(self, prepared):
        return self.combine_deepep(
            self.run_deepep_experts(self.dispatch_deepep(prepared))
        )

    def dispatch_deepep(self, prepared):
        assert self.active_stream is not None and self.ctx is not None
        self.lane_observations.append(
            (
                "dispatch",
                self.layer_id,
                self.active_stream[0],
                self.ctx.mlp_deepep_buffer,
                None if self.tp_plugin is None else self.tp_plugin.comm,
            )
        )
        return prepared

    def run_deepep_experts(self, dispatched):
        assert self.active_stream is not None and self.ctx is not None
        self.lane_observations.append(
            (
                "experts",
                self.layer_id,
                self.active_stream[0],
                self.ctx.mlp_deepep_buffer,
                None if self.tp_plugin is None else self.tp_plugin.comm,
            )
        )
        return dispatched

    def combine_deepep(
        self, expert_output, *, release_turn=None, release_value=0
    ):
        assert self.active_stream is not None and self.ctx is not None
        if release_turn is not None:
            self.lane_observations.append(
                (
                    "combine_release",
                    self.layer_id,
                    self.active_stream[0],
                    release_turn,
                    release_value,
                )
            )
        self.lane_observations.append(
            (
                "combine",
                self.layer_id,
                self.active_stream[0],
                self.ctx.mlp_deepep_buffer,
                None if self.tp_plugin is None else self.tp_plugin.comm,
            )
        )
        return expert_output


class _Model:
    def __init__(self, layers, *, observations=None, active_stream=None) -> None:
        self.model = SimpleNamespace(layers=SimpleNamespace(op_list=layers))
        self.observations = observations
        self.active_stream = active_stream

    def embed_input_ids(self, input_ids):
        if self.observations is not None:
            self.observations.append(("embed", self.active_stream[0]))
        return input_ids.to(torch.float32).reshape(-1, 1)

    def finalize_hidden(self, hidden, residual):
        if self.observations is not None:
            self.observations.append(("finalize", self.active_stream[0]))
        return hidden + residual


class _FakeStream:
    def __init__(
        self,
        name: str,
        waits: list[tuple[str, str]],
        timeline: list[tuple] | None = None,
    ) -> None:
        self.name = name
        self.waits = waits
        self.timeline = timeline

    def wait_stream(self, other: "_FakeStream") -> None:
        self.waits.append((self.name, other.name))
        if self.timeline is not None:
            self.timeline.append(("wait_stream", self.name, other.name))

    def wait_event(self, event: "_FakeEvent") -> None:
        self.waits.append((self.name, event.name))
        if self.timeline is not None:
            self.timeline.append(("wait_event", self.name, event.name))


class _FakeEvent:
    def __init__(
        self,
        name: str,
        records: list[tuple[str, str]],
        timeline: list[tuple] | None = None,
    ) -> None:
        self.name = name
        self.records = records
        self.timeline = timeline

    def record(self, stream: _FakeStream) -> None:
        self.records.append((self.name, stream.name))
        if self.timeline is not None:
            self.timeline.append(("record_event", self.name, stream.name))


class AfdFmhaRuntimeScheduleTest(unittest.TestCase):
    def test_shared_publication_arena_is_imported_once_for_all_views(self) -> None:
        runtime = object.__new__(AfdFmhaRuntime)
        runtime._imports = {}
        runtime._imported_allocations = {}
        shared_allocation = object()
        calls: list[tuple[str, int, object | None]] = []
        record = _FabricRecord(
            role="attention",
            dp_rank=7,
            tp_rank=0,
            q_handle=b"shared-arena",
            q_shape=(2, 4),
            q_offset=16,
            q_ready_handle=b"shared-arena",
            q_ready_shape=(2, 1),
            q_ready_offset=32,
        )

        def fake_import(handle, shape, *, dtype, byte_offset, allocation):
            calls.append((handle.decode(), byte_offset, allocation))
            selected = shared_allocation if allocation is None else allocation
            return FabricTensor(
                torch.empty(shape, dtype=dtype),
                selected,
                byte_offset,
            )

        with patch(
            "minisgl.afd_fmha_runtime.import_fabric_tensor",
            side_effect=fake_import,
        ):
            q = runtime._import("q", record)
            q_ready = runtime._import("q_ready", record)

        self.assertIs(q.allocation, shared_allocation)
        self.assertIs(q_ready.allocation, shared_allocation)
        self.assertEqual(
            calls,
            [
                ("shared-arena", 16, None),
                ("shared-arena", 32, shared_allocation),
            ],
        )

    def test_model_plan_tables_are_namespaced_per_attention_dp(self) -> None:
        runtime = object.__new__(AfdFmhaRuntime)
        runtime._table_namespace_stride = 7
        token_block = SimpleNamespace(
            table_idx=3,
            start_pos=0,
            tokens=torch.tensor([1, 2], dtype=torch.int32),
        )
        plan = SimpleNamespace(
            table_indices=(0, 5),
            free_table_indices=(1,),
            exec_table_indices=(0, 6),
            token_blocks=(token_block,),
        )

        namespaced = runtime._namespace_model_plan(plan, 2)

        self.assertEqual(namespaced.table_indices, (14, 19))
        self.assertEqual(namespaced.free_table_indices, (15,))
        self.assertEqual(namespaced.exec_table_indices, (14, 20))
        self.assertEqual(namespaced.token_blocks[0].table_idx, 17)
        self.assertEqual(plan.table_indices, (0, 5))
        self.assertEqual(token_block.table_idx, 3)
        with self.assertRaisesRegex(RuntimeError, "outside"):
            runtime._namespace_model_plan(
                SimpleNamespace(
                    table_indices=(7,),
                    free_table_indices=(),
                    exec_table_indices=(),
                    token_blocks=(),
                ),
                0,
            )

    def test_ready_descriptor_selects_writer_word_in_each_slot(self) -> None:
        tensor = torch.empty((2, 3), dtype=torch.int64)
        mapping = SimpleNamespace(tensor=tensor)

        self.assertEqual(
            AfdFmhaRuntime._ready_descriptor(mapping, (2, 5, 7), 5),
            [tensor.data_ptr(), 1, 3],
        )

        fanin_tensor = torch.empty((2, 4, 3), dtype=torch.int64)
        fanin_mapping = SimpleNamespace(tensor=fanin_tensor)
        self.assertEqual(
            AfdFmhaRuntime._ready_descriptor(
                fanin_mapping,
                (2, 5, 7),
                5,
                source_index=2,
                source_count=4,
            ),
            [fanin_tensor.data_ptr(), 7, 12],
        )

    def test_model_o_receive_waits_once_for_all_sources(self) -> None:
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.fanin = 2
        runtime.source_max_rows = 3
        runtime.max_rows = 6
        runtime.transport_slots = 2
        runtime.o_slots = torch.arange(12, dtype=torch.float32).view(2, 6, 1, 1)
        runtime.o_scale_packed_groups = 1
        runtime.o_scale_row_stride = 8
        runtime.o_scale_slots = torch.arange(16, dtype=torch.int32).view(2, 8)
        runtime.table = SimpleNamespace(model_local_q_heads=1)
        runtime.device = torch.device("cpu")
        runtime.o_ready = torch.ones((2, 2, 1), dtype=torch.int64)
        runtime.timeout_record = object()
        runtime.timeout_ms = 1
        runtime.head_dim = 1
        waits = []

        with patch(
            "minisgl.afd_fmha_runtime.wait_ready",
            side_effect=lambda ready, *args, **kwargs: waits.append(
                (ready, kwargs["expected_ready"])
            ),
        ):
            decode, decode_scale = runtime.receive_attention_o(
                SimpleNamespace(
                    phase="decode",
                    afd_fmha_source_token_offsets=(0, 2, 4),
                ),
                0,
            )
            prefill, prefill_scale = runtime.receive_attention_o(
                SimpleNamespace(
                    phase="prefill",
                    afd_fmha_source_token_offsets=(0, 2, 3),
                ),
                1,
            )

        self.assertEqual(len(waits), 2)
        self.assertEqual(tuple(waits[0][0].shape), (2,))
        self.assertEqual([ticket for _, ticket in waits], [1, 2])
        self.assertEqual(decode[:, 0].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(prefill[:, 0].tolist(), [6.0, 7.0, 9.0])
        self.assertEqual(decode_scale[:, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(prefill_scale[:, 0].tolist(), [8, 9, 11])

    def test_model_fanin_merge_preserves_microbatch_and_source_order(self) -> None:
        class _Materializer:
            @staticmethod
            def _collect_ag_sample_meta(plan):
                return plan.sample_meta

            @staticmethod
            def _stage_ag_sample_meta(last, tables, positions, batches):
                return torch.tensor(last), {
                    "table_indices": torch.tensor(tables),
                    "write_positions": torch.tensor(positions),
                    "batch_indices": torch.tensor(batches),
                }

        runtime = object.__new__(AfdFmhaRuntime)
        runtime.device = torch.device("cpu")
        runtime.num_mb = 2
        runtime._mapped_attn_dp_ranks = (0, 1)
        runtime.worker = SimpleNamespace(runtime=_Materializer())

        def source_batch(base: int):
            subs = []
            for mb in range(2):
                values = torch.tensor(
                    [base + mb * 2, base + mb * 2 + 1],
                    dtype=torch.int32,
                )
                reqs = [object(), object()]
                subs.append(
                    SimpleNamespace(
                        reqs=reqs,
                        padded_reqs=reqs,
                        input_ids=values,
                        positions=values.clone(),
                        out_loc=values.clone(),
                    )
                )
            return SimpleNamespace(
                reqs=[req for sub in subs for req in sub.reqs],
                afd_mb_subbatches=subs,
            )

        plans = tuple(
            SimpleNamespace(
                microbatch_token_offsets=(0, 2, 4),
                sample_meta=(
                    [1, 3],
                    [source * 10, source * 10 + 1],
                    [8, 9],
                    [0, 1],
                ),
            )
            for source in range(2)
        )
        sampling = object()
        merged = runtime._merge_model_fanin_batches(
            SimpleNamespace(phase="decode", sampling=sampling),
            plans,
            (source_batch(10), source_batch(20)),
        )

        self.assertEqual(merged.input_ids.tolist(), [10, 11, 20, 21, 12, 13, 22, 23])
        self.assertEqual(merged.afd_microbatch_token_offsets, (0, 4, 8))
        self.assertEqual(
            [sub.afd_fmha_source_token_offsets for sub in merged.afd_mb_subbatches],
            [(0, 2, 4), (0, 2, 4)],
        )
        self.assertTrue(
            all(
                sub.afd_fmha_source_token_offsets_device is None
                for sub in merged.afd_mb_subbatches
            )
        )
        self.assertEqual(merged.afd_last_indices.tolist(), [1, 5, 3, 7])
        self.assertEqual(merged.afd_writeback["batch_indices"].tolist(), [0, 1, 2, 3])
        self.assertIs(merged.afd_sampling_plan, sampling)

        prefill = runtime._merge_model_fanin_batches(
            SimpleNamespace(phase="prefill", sampling=sampling),
            plans,
            (source_batch(10), source_batch(20)),
        )
        self.assertEqual(
            [
                sub.afd_fmha_source_token_offsets_device.tolist()
                for sub in prefill.afd_mb_subbatches
            ],
            [[0, 2, 4], [0, 2, 4]],
        )

    def test_model_fanin_decode_stages_one_merged_input_mapping(self) -> None:
        class _Materializer:
            def __init__(self):
                self.token_pool = torch.arange(100, dtype=torch.int32).view(100, 1)
                self.mapping_calls = []
                self.recorded_stages = []

            def _attach_input_mapping(self, batch, plan):
                table_indices = torch.tensor(
                    [req.table_idx for req in batch.padded_reqs],
                    dtype=torch.int64,
                )
                positions = torch.zeros(len(batch.padded_reqs), dtype=torch.int32)
                batch.positions_host = positions
                batch.positions = positions
                batch.out_loc = table_indices.to(torch.int32) + 100
                batch.afd_req_table_indices_gpu = table_indices
                self.mapping_calls.append(
                    ([req.table_idx for req in batch.padded_reqs], plan.phase)
                )
                return 2, table_indices, positions.to(torch.int64)

            def _record_input_mapping_stage(self, stage_index):
                self.recorded_stages.append(stage_index)

            @staticmethod
            def _collect_ag_sample_meta(plan):
                return plan.sample_meta

            @staticmethod
            def _stage_ag_sample_meta(last, tables, positions, batches):
                return torch.tensor(last), {
                    "table_indices": torch.tensor(tables),
                    "write_positions": torch.tensor(positions),
                    "batch_indices": torch.tensor(batches),
                }

        materializer = _Materializer()
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 2
        runtime._mapped_attn_dp_ranks = (0, 1)
        runtime.worker = SimpleNamespace(runtime=materializer)

        source_tables = ((10, 11, 12, 13), (20, 21, 22, 23))
        source_batches = tuple(
            SimpleNamespace(
                reqs=[SimpleNamespace(table_idx=value) for value in tables],
                padded_reqs=[SimpleNamespace(table_idx=value) for value in tables],
            )
            for tables in source_tables
        )
        plans = tuple(
            SimpleNamespace(
                phase="decode",
                microbatch_offsets=(0, 2, 4),
                microbatch_token_offsets=(0, 2, 4),
                sample_meta=(
                    [1, 3],
                    [source * 10, source * 10 + 1],
                    [8, 9],
                    [0, 1],
                ),
            )
            for source in range(2)
        )
        sampling = object()

        merged = runtime._materialize_model_fanin_decode_batch(
            SimpleNamespace(phase="decode", sampling=sampling),
            plans,
            source_batches,
        )

        expected_order = [10, 11, 20, 21, 12, 13, 22, 23]
        self.assertEqual(materializer.mapping_calls, [(expected_order, "decode")])
        self.assertEqual(materializer.recorded_stages, [2])
        self.assertEqual(merged.input_ids.tolist(), expected_order)
        self.assertEqual(merged.out_loc.tolist(), [value + 100 for value in expected_order])
        self.assertEqual(merged.afd_microbatch_offsets, (0, 4, 8))
        self.assertEqual(merged.afd_microbatch_token_offsets, (0, 4, 8))
        self.assertEqual(
            [sub.afd_fmha_source_token_offsets for sub in merged.afd_mb_subbatches],
            [(0, 2, 4), (0, 2, 4)],
        )
        self.assertEqual(merged.afd_last_indices.tolist(), [1, 5, 3, 7])
        self.assertEqual(merged.afd_writeback["batch_indices"].tolist(), [0, 1, 2, 3])
        self.assertIs(merged.afd_sampling_plan, sampling)

    def test_ready_descriptor_rejects_inconsistent_writer_metadata(self) -> None:
        mapping = SimpleNamespace(tensor=torch.empty((2, 2), dtype=torch.int64))
        with self.assertRaisesRegex(RuntimeError, "does not name"):
            AfdFmhaRuntime._ready_descriptor(mapping, (2, 5), 7)
        with self.assertRaisesRegex(RuntimeError, "shape does not match"):
            AfdFmhaRuntime._ready_descriptor(mapping, (2, 5, 7), 5)

    def test_ready_descriptor_accepts_singleton_source_axis(self) -> None:
        mapping = SimpleNamespace(tensor=torch.empty((2, 1, 1), dtype=torch.int64))
        base = int(mapping.tensor.data_ptr())
        self.assertEqual(
            AfdFmhaRuntime._ready_descriptor(
                mapping,
                (7,),
                7,
                source_index=0,
                source_count=1,
            ),
            [base, 0, 1],
        )

    def test_stream_identity_accepts_generic_and_cuda_stream_wrappers(self) -> None:
        generic = SimpleNamespace(device_type=1, device_index=2, stream_id=41)
        cuda = SimpleNamespace(
            device_type=1,
            device_index=2,
            stream_id=41,
            cuda_stream=999,
        )
        other = SimpleNamespace(device_type=1, device_index=2, stream_id=42)

        self.assertEqual(_stream_identity(generic), _stream_identity(cuda))
        self.assertNotEqual(_stream_identity(generic), _stream_identity(other))

    def test_mb2_model_fanin_runs_two_rounds_per_layer(self) -> None:
        calls: list[tuple] = []
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 2
        runtime.state = SimpleNamespace(ctx=_Context())
        runtime.model = _Model([_Layer(0, calls), _Layer(1, calls)])
        runtime.publish_model_qkv = lambda qkv, sub, *, layer, epoch: calls.append(
            ("publish", layer, epoch, int(qkv.shape[0]))
        )

        def receive(sub, epoch):
            rows = int(sub.positions.numel())
            calls.append(("receive", epoch, rows))
            output = torch.full((rows, 1), float(epoch))
            return output, torch.ones((rows, 1), dtype=torch.int32)

        runtime.receive_attention_o = receive

        sub0 = SimpleNamespace(positions=torch.arange(12))
        sub1 = SimpleNamespace(positions=torch.arange(12))
        batch = SimpleNamespace(
            input_ids=torch.arange(24, dtype=torch.int32),
            positions=torch.arange(24),
            afd_mb_subbatches=[sub0, sub1],
            afd_microbatch_token_offsets=(0, 12, 24),
        )
        valid0 = torch.tensor([10], dtype=torch.int64)
        valid1 = torch.tensor([9], dtype=torch.int64)

        output = runtime._run_model_forward(
            batch,
            slot_base=0,
            real_token_counts=(valid0, valid1),
            dispatch_bucket=12,
        )

        self.assertEqual(tuple(output.shape), (24, 1))
        self.assertEqual(
            [call for call in calls if call[0] == "receive"],
            [
                ("receive", 0, 12),
                ("receive", 1, 12),
                ("receive", 2, 12),
                ("receive", 3, 12),
            ],
        )
        self.assertLess(
            calls.index(("publish", 1, 2, 12)),
            calls.index(("receive", 1, 12)),
        )
        self.assertLess(
            calls.index(("publish", 1, 3, 12)),
            calls.index(("receive", 2, 12)),
        )
        self.assertEqual(
            [call for call in calls if call[0] == "mlp"],
            [
                ("mlp", 0, 12),
                ("mlp", 0, 12),
                ("mlp", 1, 12),
                ("mlp", 1, 12),
            ],
        )
        self.assertIs(runtime.state.ctx.moe_num_token_non_padded, valid1)
        self.assertEqual(runtime.state.ctx.moe_deepep_dispatch_max_tokens_per_rank, 12)

    def test_six_prefill_microbatches_roll_over_two_live_transport_slots(self) -> None:
        calls: list[tuple] = []
        live_epochs: dict[int, int] = {}
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 6
        runtime.state = SimpleNamespace(ctx=_Context())
        runtime.model = _Model([_Layer(0, calls), _Layer(1, calls)])

        def publish(qkv, sub, *, layer, epoch):
            slot = epoch % 2
            self.assertNotIn(slot, live_epochs)
            live_epochs[slot] = epoch
            calls.append(("publish", layer, epoch, int(qkv.shape[0])))

        def receive(sub, epoch):
            rows = int(sub.positions.numel())
            slot = epoch % 2
            self.assertEqual(live_epochs.pop(slot), epoch)
            calls.append(("receive", epoch, rows))
            output = torch.full((rows, 1), float(epoch))
            return output, torch.ones((rows, 1), dtype=torch.int32)

        runtime.publish_model_qkv = publish
        runtime.receive_attention_o = receive
        subs = [SimpleNamespace(positions=torch.arange(1)) for _ in range(6)]
        batch = SimpleNamespace(
            input_ids=torch.arange(6, dtype=torch.int32),
            positions=torch.arange(6),
            afd_mb_subbatches=subs,
            afd_microbatch_token_offsets=tuple(range(7)),
        )

        output = runtime._run_model_forward(
            batch,
            slot_base=0,
            real_token_counts=(1,) * 6,
            dispatch_bucket=1,
        )

        self.assertEqual(tuple(output.shape), (6, 1))
        self.assertFalse(live_epochs)
        self.assertEqual(
            [call[2] for call in calls if call[0] == "publish"],
            list(range(12)),
        )
        self.assertEqual(
            [call[1] for call in calls if call[0] == "receive"],
            list(range(12)),
        )

    def test_mb2_decode_serializes_ffn_qkv_compute_units(self) -> None:
        calls: list[tuple] = []
        observations: list[tuple] = []
        waits: list[tuple[str, str]] = []
        event_records: list[tuple[str, str]] = []
        active_stream = ['root']
        root = _FakeStream('root', waits, observations)
        lanes = (
            _FakeStream('mb0', waits, observations),
            _FakeStream('mb1', waits, observations),
        )
        bootstrap_events = tuple(
            _FakeEvent(f'bootstrap_q{mb}', event_records, observations)
            for mb in range(2)
        )
        compute_events = tuple(
            tuple(
                _FakeEvent(
                    f'compute_l{layer}_mb{mb}',
                    event_records,
                    observations,
                )
                for mb in range(2)
            )
            for layer in range(2)
        )
        ctx = _Context()
        buffers = (object(), object())
        communicators = (object(), object())
        tp_plugin = SimpleNamespace(comm=object())
        original_communicator = tp_plugin.comm

        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 2
        runtime.state = SimpleNamespace(ctx=ctx, stream=root)
        runtime._moe_buffers = buffers
        runtime._tp_lane_communicators = communicators
        runtime._tp_distributed_plugin = tp_plugin
        runtime.model = _Model(
            [
                _Layer(
                    layer,
                    calls,
                    lane_observations=observations,
                    active_stream=active_stream,
                    ctx=ctx,
                    tp_plugin=tp_plugin,
                )
                for layer in range(2)
            ],
            observations=observations,
            active_stream=active_stream,
        )

        def prepare_model_qkv(qkv, sub, *, layer):
            observations.append(('qk_norm_rope', layer, active_stream[0]))
            return qkv, qkv, qkv

        def publish_model_qkv_payload(q, k, v, sub, *, layer, epoch):
            observations.append(('publish', layer, epoch, active_stream[0]))

        def receive(sub, epoch):
            rows = int(sub.positions.numel())
            observations.append(('receive', epoch, rows, active_stream[0]))
            output = torch.full((rows, 1), float(epoch))
            return output, torch.ones((rows, 1), dtype=torch.int32)

        runtime.prepare_model_qkv = prepare_model_qkv
        runtime.publish_model_qkv_payload = publish_model_qkv_payload
        runtime.receive_attention_o = receive
        sub0 = SimpleNamespace(positions=torch.arange(4), phase='decode')
        sub1 = SimpleNamespace(positions=torch.arange(4), phase='decode')
        batch = SimpleNamespace(
            input_ids=torch.arange(8, dtype=torch.int32),
            positions=torch.arange(8),
            afd_mb_subbatches=[sub0, sub1],
            afd_microbatch_token_offsets=(0, 4, 8),
        )

        @contextlib.contextmanager
        def use_stream(stream):
            previous = active_stream[0]
            active_stream[0] = stream.name
            try:
                yield
            finally:
                active_stream[0] = previous

        with patch('torch.cuda.stream', side_effect=use_stream):
            output = runtime._run_model_forward(
                batch,
                slot_base=0,
                real_token_counts=(4, 3),
                dispatch_bucket=4,
                lane_streams=lanes,
                bootstrap_compute_events=bootstrap_events,
                compute_done_events=compute_events,
            )

        self.assertEqual(tuple(output.shape), (8, 1))
        self.assertEqual(
            waits,
            [
                ('mb1', 'bootstrap_q0'),
                ('mb0', 'bootstrap_q1'),
                ('mb1', 'compute_l0_mb0'),
                ('mb0', 'compute_l0_mb1'),
                ('mb1', 'compute_l1_mb0'),
                ('mb0', 'mb1'),
            ],
        )
        self.assertEqual(
            event_records,
            [
                ('bootstrap_q0', 'mb0'),
                ('bootstrap_q1', 'mb1'),
                ('compute_l0_mb0', 'mb0'),
                ('compute_l0_mb1', 'mb1'),
                ('compute_l1_mb0', 'mb0'),
                ('compute_l1_mb1', 'mb1'),
            ],
        )

        # Bootstrap QKV compute is exclusive; publication begins only after the
        # handoff event, allowing it to overlap peer compute.
        self.assertLess(
            observations.index(('qk_norm_rope', 0, 'mb0')),
            observations.index(('record_event', 'bootstrap_q0', 'mb0')),
        )
        self.assertLess(
            observations.index(('record_event', 'bootstrap_q0', 'mb0')),
            observations.index(('publish', 0, 0, 'mb0')),
        )
        self.assertLess(
            observations.index(('wait_event', 'mb1', 'bootstrap_q0')),
            observations.index(('qk_norm_rope', 0, 'mb1')),
        )

        # Unit A is FFN(A,L-1)+QKV(A,L). B receives eagerly but cannot run its
        # first heavy kernel until the A handoff recorded after next-layer QKV.
        a_combine = ('combine', 0, 'mb0', buffers[0], communicators[0])
        a_next_qkv = ('qk_norm_rope', 1, 'mb0')
        a_done = ('record_event', 'compute_l0_mb0', 'mb0')
        a_publish = ('publish', 1, 2, 'mb0')
        b_receive = ('receive', 1, 4, 'mb1')
        b_wait = ('wait_event', 'mb1', 'compute_l0_mb0')
        b_finish = ('finish', 0, 'mb1')
        for before, after in (
            (a_combine, a_next_qkv),
            (a_next_qkv, a_done),
            (a_done, a_publish),
            (b_receive, b_wait),
            (b_wait, b_finish),
        ):
            self.assertLess(observations.index(before), observations.index(after))

        # Every combine is immediate and belongs to the exclusive unit; the old
        # deferred odd-lane release mechanism is absent.
        self.assertEqual(
            [item[1:3] for item in observations if item[0] == 'combine'],
            [(0, 'mb0'), (0, 'mb1'), (1, 'mb0'), (1, 'mb1')],
        )
        self.assertFalse(
            any(item[0] in ('combine_release', 'wait_turn') for item in observations)
        )
        self.assertLess(
            observations.index(('combine', 1, 'mb0', buffers[0], communicators[0])),
            observations.index(('record_event', 'compute_l1_mb0', 'mb0')),
        )
        self.assertLess(
            observations.index(('wait_event', 'mb1', 'compute_l1_mb0')),
            observations.index(('finish', 1, 'mb1')),
        )
        self.assertEqual(
            [item for item in observations if item[0] == 'finalize'],
            [('finalize', 'mb1'), ('finalize', 'mb1')],
        )
        self.assertLess(
            observations.index(('finalize', 'mb1')),
            observations.index(('wait_stream', 'mb0', 'mb1')),
        )
        self.assertIsNone(ctx.mlp_deepep_buffer)
        self.assertIs(tp_plugin.comm, original_communicator)

    def test_three_model_microbatches_reuse_two_ffn_lanes(self) -> None:
        calls: list[tuple] = []
        observations: list[tuple] = []
        waits: list[tuple[str, str]] = []
        records: list[tuple[str, str]] = []
        active_stream = ['root']
        lanes = tuple(
            _FakeStream(f'lane{lane}', waits, observations) for lane in range(2)
        )
        bootstrap_events = tuple(
            _FakeEvent(f'bootstrap_q{mb}', records, observations)
            for mb in range(2)
        )
        events = (
            tuple(
                _FakeEvent(f'compute_mb{mb}', records, observations)
                for mb in range(3)
            ),
        )
        ctx = _Context()
        buffers = (object(), object())
        communicators = (object(), object())
        tp_plugin = SimpleNamespace(comm=object())

        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 3
        runtime.state = SimpleNamespace(ctx=ctx)
        runtime._moe_buffers = buffers
        runtime._tp_lane_communicators = communicators
        runtime._tp_distributed_plugin = tp_plugin
        runtime.model = _Model(
            [
                _Layer(
                    0,
                    calls,
                    lane_observations=observations,
                    active_stream=active_stream,
                    ctx=ctx,
                    tp_plugin=tp_plugin,
                )
            ],
            observations=observations,
            active_stream=active_stream,
        )
        runtime.prepare_model_qkv = lambda qkv, sub, *, layer: (qkv, qkv, qkv)
        runtime.publish_model_qkv_payload = lambda *args, **kwargs: None
        runtime.receive_attention_o = lambda sub, epoch: (
            torch.full((int(sub.positions.numel()), 1), float(epoch)),
            torch.ones((int(sub.positions.numel()), 1), dtype=torch.int32),
        )
        subs = [SimpleNamespace(positions=torch.arange(2)) for _ in range(3)]
        batch = SimpleNamespace(
            input_ids=torch.arange(6, dtype=torch.int32),
            positions=torch.arange(6),
            afd_mb_subbatches=subs,
            afd_microbatch_token_offsets=(0, 2, 4, 6),
        )

        @contextlib.contextmanager
        def use_stream(stream):
            previous = active_stream[0]
            active_stream[0] = stream.name
            try:
                yield
            finally:
                active_stream[0] = previous

        with patch('torch.cuda.stream', side_effect=use_stream):
            output = runtime._run_model_forward(
                batch,
                slot_base=0,
                real_token_counts=(2, 2, 2),
                dispatch_bucket=2,
                lane_streams=lanes,
                bootstrap_compute_events=bootstrap_events,
                compute_done_events=events,
            )

        self.assertEqual(tuple(output.shape), (6, 1))
        self.assertEqual(
            [item[2:] for item in observations if item[0] == 'pre_moe'],
            [
                ('lane0', buffers[0], communicators[0]),
                ('lane1', buffers[1], communicators[1]),
                ('lane0', buffers[0], communicators[0]),
            ],
        )
        self.assertEqual(
            records,
            [
                ('bootstrap_q0', 'lane0'),
                ('bootstrap_q1', 'lane1'),
                ('compute_mb0', 'lane0'),
                ('compute_mb1', 'lane1'),
                ('compute_mb2', 'lane0'),
            ],
        )
        self.assertEqual(
            waits,
            [
                ('lane1', 'bootstrap_q0'),
                ('lane0', 'bootstrap_q1'),
                ('lane1', 'compute_mb0'),
                ('lane0', 'compute_mb1'),
            ],
        )
        self.assertFalse(
            any(item[0] == 'combine_release' for item in observations)
        )

    def test_mb2_attention_runs_wait_fmha_and_publish_on_matching_main_stream(self) -> None:
        observations: list[tuple] = []
        waits: list[tuple[str, str]] = []
        event_records: list[tuple[str, str]] = []
        active_stream = ["root"]
        root = _FakeStream("root", waits, observations)
        lanes = (
            _FakeStream("mb0", waits, observations),
            _FakeStream("mb1", waits, observations),
        )
        ctx = _Context()

        def make_backend(mb: int):
            def forward_prepared(q, layer, sub):
                observations.append(
                    ("fmha", layer, mb, active_stream[0], ctx.attn_backend)
                )
                return torch.full((int(sub.positions.numel()), 1), float(mb))

            return SimpleNamespace(forward_prepared=forward_prepared)

        backends = (make_backend(0), make_backend(1))
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 2
        runtime.transport_slots = 2
        runtime.model_config = SimpleNamespace(num_layers=2)
        runtime.state = SimpleNamespace(ctx=ctx, stream=root, attn_backends=backends)
        runtime.max_rows = 4
        runtime.source_max_rows = 4
        runtime.q_slots = torch.zeros((2, 4, 1))
        runtime.q_ready = torch.zeros((2, 1), dtype=torch.int64)
        runtime.timeout_record = object()
        runtime.timeout_ms = 1
        runtime.head_dim = 1
        runtime.o_descriptors = object()
        runtime.o_ready_descriptors = object()
        runtime.o_publish_counters = torch.zeros((2,), dtype=torch.int32)
        runtime.o_quantization_counters = torch.zeros((2,), dtype=torch.int32)
        runtime.o_fp8_staging = torch.empty((2, 4, 1))
        runtime.o_scale_staging = torch.empty((2, 4, 1), dtype=torch.uint8)
        runtime._attention_turn = object()
        sub0 = SimpleNamespace(positions=torch.arange(4), phase="decode")
        sub1 = SimpleNamespace(positions=torch.arange(4), phase="decode")
        batch = SimpleNamespace(
            positions=torch.arange(8),
            afd_mb_subbatches=[sub0, sub1],
            afd_microbatch_token_offsets=(0, 4, 8),
        )

        @contextlib.contextmanager
        def use_stream(stream):
            previous = active_stream[0]
            active_stream[0] = stream.name
            try:
                yield
            finally:
                active_stream[0] = previous

        def fake_wait_ready(*args, **kwargs):
            observations.append(
                ("wait", active_stream[0], kwargs.get("expected_turn"))
            )

        def fake_publish(*args, **kwargs):
            observations.append(("publish", active_stream[0], kwargs["slot"]))

        def fake_publish_release(*args, **kwargs):
            observations.append(
                ("release", active_stream[0], kwargs.get("next_turn"))
            )
            observations.append(("publish", active_stream[0], kwargs["slot"]))

        with (
            patch("torch.cuda.stream", side_effect=use_stream),
            patch("minisgl.afd_fmha_runtime.wait_ready", side_effect=fake_wait_ready),
            patch(
                "minisgl.afd_fmha_runtime.quantize_publish_o_fp8_release_turn",
                side_effect=fake_publish_release,
            ),
            patch(
                "minisgl.afd_fmha_runtime.quantize_publish_o_fp8",
                side_effect=fake_publish,
            ),
        ):
            runtime._run_attention_body(
                batch,
                0,
                lane_streams=lanes,
            )

        self.assertEqual(
            [item for item in observations if item[0] == "fmha"],
            [
                ("fmha", 0, 0, "mb0", backends[0]),
                ("fmha", 0, 1, "mb1", backends[1]),
                ("fmha", 1, 0, "mb0", backends[0]),
                ("fmha", 1, 1, "mb1", backends[1]),
            ],
        )
        self.assertEqual(
            [item[1:] for item in observations if item[0] == "wait"],
            [("mb0", 0), ("mb1", 1), ("mb0", 2), ("mb1", 3)],
        )
        self.assertEqual(
            [item[1:] for item in observations if item[0] == "publish"],
            [
                ("mb0", 0),
                ("mb1", 1),
                ("mb0", 0),
                ("mb1", 1),
            ],
        )
        self.assertEqual(
            [item[1:] for item in observations if item[0] == "release"],
            [("mb0", 1), ("mb1", 2), ("mb0", 3), ("mb1", 0)],
        )
        self.assertEqual(waits, [])
        self.assertEqual(event_records, [])
        self.assertEqual(
            [
                item
                for item in observations
                if item[0]
                in {"wait", "wait_event", "fmha", "release", "record_event", "publish"}
            ],
            [
                ("wait", "mb0", 0),
                ("wait", "mb1", 1),
                ("fmha", 0, 0, "mb0", backends[0]),
                ("release", "mb0", 1),
                ("publish", "mb0", 0),
                ("wait", "mb0", 2),
                ("fmha", 0, 1, "mb1", backends[1]),
                ("release", "mb1", 2),
                ("publish", "mb1", 1),
                ("wait", "mb1", 3),
                ("fmha", 1, 0, "mb0", backends[0]),
                ("release", "mb0", 3),
                ("publish", "mb0", 0),
                ("fmha", 1, 1, "mb1", backends[1]),
                ("release", "mb1", 0),
                ("publish", "mb1", 1),
            ],
        )
        self.assertIsNone(ctx.attn_backend)

    def test_eager_attention_body_does_not_use_decode_device_turn(self) -> None:
        calls: list[dict] = []
        backend = object()
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 2
        runtime.model_config = SimpleNamespace(num_layers=1)
        runtime.state = SimpleNamespace(
            ctx=_Context(),
            attn_backends=(backend, backend),
        )
        runtime._attention_turn = object()
        runtime.run_attention_fmha = lambda *args, **kwargs: calls.append(kwargs)
        batch = SimpleNamespace(
            positions=torch.arange(8),
            afd_mb_subbatches=[
                SimpleNamespace(positions=torch.arange(4), phase="decode"),
                SimpleNamespace(positions=torch.arange(4), phase="decode"),
            ],
            afd_microbatch_token_offsets=(0, 4, 8),
        )

        runtime._run_attention_body(batch, 0)

        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIsNone(call["lane_stream"])
            self.assertIsNone(call["attention_turn"])
            self.assertIsNone(call["expected_turn"])
            self.assertIsNone(call["next_turn"])

    def test_three_microbatches_ping_pong_on_two_streams_and_slots(self) -> None:
        observations: list[tuple] = []
        active_stream = ["root"]
        lanes = tuple(_FakeStream(f"lane{lane}", []) for lane in range(2))
        ctx = _Context()

        def make_backend(mb: int):
            def forward_prepared(q, layer, sub):
                observations.append(("fmha", layer, mb, active_stream[0]))
                return torch.full((int(sub.positions.numel()), 1), float(mb))

            return SimpleNamespace(forward_prepared=forward_prepared)

        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 3
        runtime.transport_slots = 2
        runtime.model_config = SimpleNamespace(num_layers=2)
        runtime.state = SimpleNamespace(
            ctx=ctx,
            attn_backends=tuple(make_backend(mb) for mb in range(3)),
        )
        runtime.max_rows = 3
        runtime.q_slots = torch.zeros((2, 3, 1))
        runtime.q_ready = torch.zeros((2, 1), dtype=torch.int64)
        runtime.timeout_record = object()
        runtime.timeout_ms = 1
        runtime.head_dim = 1
        runtime.o_descriptors = object()
        runtime.o_ready_descriptors = object()
        runtime.o_publish_counters = torch.zeros((2,), dtype=torch.int32)
        runtime.o_quantization_counters = torch.zeros((2,), dtype=torch.int32)
        runtime.o_fp8_staging = torch.empty((2, 3, 1))
        runtime.o_scale_staging = torch.empty((2, 3, 1), dtype=torch.uint8)
        runtime._attention_turn = object()
        subs = [
            SimpleNamespace(positions=torch.arange(3), phase="decode")
            for _ in range(3)
        ]
        batch = SimpleNamespace(
            positions=torch.arange(9),
            afd_mb_subbatches=subs,
            afd_microbatch_token_offsets=(0, 3, 6, 9),
        )

        @contextlib.contextmanager
        def use_stream(stream):
            previous = active_stream[0]
            active_stream[0] = stream.name
            try:
                yield
            finally:
                active_stream[0] = previous

        def fake_wait_ready(*args, **kwargs):
            observations.append(
                ("wait", active_stream[0], kwargs.get("expected_turn"))
            )

        def fake_publish_release(*args, **kwargs):
            observations.append(
                (
                    "release",
                    active_stream[0],
                    kwargs["slot"],
                    kwargs["next_turn"],
                )
            )

        with (
            patch("torch.cuda.stream", side_effect=use_stream),
            patch("minisgl.afd_fmha_runtime.wait_ready", side_effect=fake_wait_ready),
            patch(
                "minisgl.afd_fmha_runtime.quantize_publish_o_fp8_release_turn",
                side_effect=fake_publish_release,
            ),
        ):
            runtime._run_attention_body(batch, 0, lane_streams=lanes)

        self.assertEqual(
            [item for item in observations if item[0] == "fmha"],
            [
                ("fmha", layer, mb, f"lane{mb % 2}")
                for layer in range(2)
                for mb in range(3)
            ],
        )
        self.assertEqual(
            [item[1:] for item in observations if item[0] == "wait"],
            [
                ("lane0", 0),
                ("lane1", 1),
                ("lane0", 2),
                ("lane0", 3),
                ("lane1", 4),
                ("lane0", 5),
            ],
        )
        self.assertEqual(
            [item[1:] for item in observations if item[0] == "release"],
            [
                ("lane0", 0, 1),
                ("lane1", 1, 2),
                ("lane0", 0, 3),
                ("lane0", 1, 4),
                ("lane1", 0, 5),
                ("lane0", 1, 0),
            ],
        )
        self.assertIsNone(ctx.attn_backend)

    def test_mb2_attention_requires_device_turn(self) -> None:
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 2
        runtime.model_config = SimpleNamespace(num_layers=1)
        runtime.state = SimpleNamespace(ctx=_Context())
        runtime._attention_turn = None
        batch = SimpleNamespace(
            positions=torch.arange(8),
            afd_mb_subbatches=[
                SimpleNamespace(positions=torch.arange(4), phase="decode"),
                SimpleNamespace(positions=torch.arange(4), phase="decode"),
            ],
            afd_microbatch_token_offsets=(0, 4, 8),
        )
        with self.assertRaisesRegex(RuntimeError, "device turn"):
            runtime._run_attention_body(
                batch,
                0,
                lane_streams=(_FakeStream("mb0", []), _FakeStream("mb1", [])),
            )

    def test_ping_pong_event_grid_must_match_layers_and_microbatches(self) -> None:
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 2
        runtime.state = SimpleNamespace(ctx=_Context())
        runtime.model = _Model([_Layer(0, [])])
        batch = SimpleNamespace(
            input_ids=torch.arange(8, dtype=torch.int32),
            positions=torch.arange(8),
            afd_mb_subbatches=[
                SimpleNamespace(positions=torch.arange(4), phase='decode'),
                SimpleNamespace(positions=torch.arange(4), phase='decode'),
            ],
            afd_microbatch_token_offsets=(0, 4, 8),
        )
        lanes = (_FakeStream('mb0', []), _FakeStream('mb1', []))
        with self.assertRaisesRegex(RuntimeError, 'bootstrap compute-event count'):
            runtime._run_model_forward(batch, slot_base=0, lane_streams=lanes)

        records: list[tuple[str, str]] = []
        bootstrap_events = tuple(
            _FakeEvent(f'bootstrap_q{mb}', records) for mb in range(2)
        )
        with self.assertRaisesRegex(RuntimeError, 'compute-event grid'):
            runtime._run_model_forward(
                batch,
                slot_base=0,
                lane_streams=lanes,
                bootstrap_compute_events=bootstrap_events,
            )

    def test_decode_uses_exactly_two_lanes_for_multiple_microbatches(self) -> None:
        runtime = object.__new__(AfdFmhaRuntime)
        runtime.num_mb = 3
        with self.assertRaisesRegex(RuntimeError, "exactly two streams"):
            runtime._validate_lane_streams((_FakeStream("mb0", []),))
        with self.assertRaisesRegex(RuntimeError, "exactly two streams"):
            runtime._validate_lane_streams(
                tuple(_FakeStream(f"mb{mb}", []) for mb in range(3))
            )


if __name__ == "__main__":
    unittest.main()
