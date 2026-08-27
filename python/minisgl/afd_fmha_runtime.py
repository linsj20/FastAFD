from __future__ import annotations

import os
from copy import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from minisgl.afd_fmha_protocol import (
    AfdHeadSlice,
    AfdTpSliceTable,
    build_moe_tp_column_groups,
    validate_decode_microbatch_token_counts,
)
from minisgl.kernel.afd_fmha_transport import (
    ensure_afd_fmha_transport_built,
    publish_o_fp8,
    publish_o_fp8_release_turn,
    publish_qkv,
    wait_ready,
)
from minisgl.kernel.fabric_memory import (
    FabricTensor,
    allocate_fabric_tensors,
    import_fabric_tensor,
)
from minisgl.utils import div_ceil, torch_dtype

from .afd_support import log_line, resolve_afd_moe_backend
from .afd_protocol import AfdModelStepPlan


def _stream_identity(stream: Any) -> tuple[int, int, int]:
    """Return the exact c10 stream identity for either PyTorch stream wrapper."""
    return (
        int(stream.device_type),
        int(stream.device_index),
        int(stream.stream_id),
    )


@dataclass(frozen=True)
class _FabricRecord:
    role: str
    dp_rank: int
    tp_rank: int
    q_handle: bytes | None = None
    q_shape: tuple[int, ...] = ()
    q_offset: int = 0
    q_ready_handle: bytes | None = None
    q_ready_shape: tuple[int, ...] = ()
    q_ready_offset: int = 0
    q_ready_writers: tuple[int, ...] = ()
    kv_handle: bytes | None = None
    kv_shape: tuple[int, ...] = ()
    kv_offset: int = 0
    o_handle: bytes | None = None
    o_shape: tuple[int, ...] = ()
    o_offset: int = 0
    o_scale_handle: bytes | None = None
    o_scale_shape: tuple[int, ...] = ()
    o_scale_offset: int = 0
    o_ready_handle: bytes | None = None
    o_ready_shape: tuple[int, ...] = ()
    o_ready_offset: int = 0
    o_ready_writers: tuple[int, ...] = ()


@dataclass(frozen=True)
class AfdFmhaInFlight:
    step_id: int
    sent_ns: int
    done: torch.cuda.Event
    retained_batch: Any


@dataclass(frozen=True)
class AfdFmhaFanInFlight:
    step_id: int
    sent_ns: int
    done: torch.cuda.Event
    retained_batch: Any
    next_tokens_cpu: torch.Tensor
    attn_dp_ranks: tuple[int, ...]
    next_token_offsets: tuple[int, ...]


@dataclass
class _FmhaDecodeGraph:
    graph: torch.cuda.CUDAGraph
    batch: Any
    next_tokens: torch.Tensor | None
    valid_token_counts: tuple[torch.Tensor, ...]
    valid_token_counts_host: tuple[torch.Tensor, ...]
    bootstrap_compute_events: tuple[torch.cuda.Event, ...] = ()
    compute_done_events: tuple[tuple[torch.cuda.Event, ...], ...] = ()


class AfdFmhaRuntime:
    """Role-split model/FMHA runtime for one AFD worker."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.state = worker.runtime_state
        self.config = self.state.config
        self.model_config = self.config.model_config
        self.device = self.state.device
        self.role = worker.role
        self.head_dim = int(self.model_config.head_dim)
        if self.head_dim != 128 or self.config.dtype != torch.bfloat16:
            raise RuntimeError(
                "FMHA-only placement requires BF16 activations with head_dim=128, "
                f"got dtype={self.config.dtype} head_dim={self.head_dim}"
            )
        if worker.attn_dp_size < worker.mlp_dp_size:
            raise RuntimeError(
                "FMHA-only placement requires at least as many attention DP lanes "
                "as model DP lanes: "
                f"attn_dp={worker.attn_dp_size} model_dp={worker.mlp_dp_size}"
            )
        topology = worker._topology
        self._mapped_attn_dp_ranks = (
            topology.attn_dps_for_mlp_dp(int(worker.mlp_dp_rank))
            if self.role == "mlp"
            else (int(worker.attn_dp_rank),)
        )
        if not self._mapped_attn_dp_ranks:
            raise RuntimeError(
                "FMHA-only model DP has no assigned attention DP lanes: "
                f"mlp_dp_rank={int(worker.mlp_dp_rank)}"
            )
        self.num_mb = int(worker.afd_num_mb)
        if self.num_mb < 1:
            raise RuntimeError(
                "FMHA-only placement requires at least one microbatch, "
                f"got --afd-num-mb {self.num_mb}"
            )
        self.num_lanes = min(2, self.num_mb)
        self.transport_slots = 2
        if (
            worker.enable_decode_graph
            and int(self.model_config.num_layers) * self.num_mb
            <= self.transport_slots
        ):
            raise RuntimeError(
                "FMHA-only exact readiness tickets require more graph rounds "
                "than transport slots so replay cannot observe its prior final "
                "ticket: "
                f"layers={int(self.model_config.num_layers)} num_mb={self.num_mb} "
                f"slots={self.transport_slots}"
            )
        if self.role == "attention" and len(self.state.attn_backends) < self.num_mb:
            raise RuntimeError(
                "FMHA-only placement requires one attention backend per microbatch: "
                f"backends={len(self.state.attn_backends)} num_mb={self.num_mb}"
            )
        worker_lane_streams = tuple(getattr(worker, "_afd_comm_streams", ()))
        self._decode_lane_streams = (
            worker_lane_streams[: self.num_lanes] if self.num_mb > 1 else ()
        )
        if self.num_mb > 1:
            if len(self._decode_lane_streams) != self.num_lanes:
                raise RuntimeError(
                    "FMHA-only pipelined decode requires exactly two CUDA streams: "
                    f"streams={len(self._decode_lane_streams)} num_mb={self.num_mb}"
                )
            stream_handles = tuple(
                int(stream.cuda_stream) for stream in self._decode_lane_streams
            )
            if len(set(stream_handles)) != self.num_lanes:
                raise RuntimeError(
                    "FMHA-only pipelined decode requires distinct CUDA streams: "
                    f"handles={stream_handles}"
                )
        self.table = AfdTpSliceTable.build(
            attn_tp_size=worker.attn_tp_size,
            model_tp_size=worker.mlp_tp_size,
            num_qo_heads=int(self.model_config.num_qo_heads),
            num_kv_heads=int(self.model_config.num_kv_heads),
        )
        self.source_max_rows = int(worker.max_comm_tokens)
        self.fanin = len(self._mapped_attn_dp_ranks) if self.role == "mlp" else 1
        self.max_rows = self.source_max_rows * self.fanin
        invalid_o_slices = tuple(
            head_slice
            for head_slice in self.table.o_slices
            if head_slice.source_head_start % 4
            or head_slice.destination_head_start % 4
            or head_slice.head_count % 4
        )
        if invalid_o_slices:
            raise RuntimeError(
                "FP8 O transport requires four-head-aligned TP slices: "
                f"slices={invalid_o_slices}"
            )
        self.o_scale_packed_groups = div_ceil(
            self.table.model_local_q_heads, 4
        )
        self.o_scale_row_stride = div_ceil(self.max_rows, 4) * 4
        self.o_scale_slot_elements = (
            self.o_scale_row_stride * self.o_scale_packed_groups
        )
        self.timeout_ms = int(float(self.config.distributed_timeout) * 1000)
        self._imports: dict[tuple[str, str, int, int], FabricTensor] = {}
        self._imported_allocations: dict[bytes, Any] = {}
        self._moe_backend = resolve_afd_moe_backend(worker.model_placement)
        self._mega_moe: Any | None = None
        self._moe_buffers: tuple[Any, ...] = ()
        self._tp_lane_communicators: tuple[Any, ...] = ()
        self._tp_distributed_plugin: Any | None = None
        self._create_moe_group()
        self._allocate_and_map_transport()
        if self.role == "mlp":
            self._init_model_materializer_namespaces()
            self._create_tp_lane_communicators()
            self._load_model()
            self._create_moe_buffer()
        else:
            self.model = None
            self.sampler = None
        self._graph_pool = None
        self._decode_graphs: dict[tuple[int, int], _FmhaDecodeGraph] = {}
        self._graph_output_free: torch.cuda.Event | None = None
        self._capture_backends_initialized: set[int] = set()
        self._init_graph_buffers()
        ensure_afd_fmha_transport_built(self.head_dim)
        if self.role == "attention":
            self._prewarm_attention_decode_kernels()
        else:
            self._prewarm_model_decode_kernels()
        dist.barrier()
        log_line(
            worker.log_path,
            f"[{self.role} rank={worker.tp_rank}] fmha_only_ready "
            f"q_edges={int(self.q_descriptors.shape[0])} "
            f"kv_edges={int(self.kv_descriptors.shape[0])} "
            f"o_edges={int(self.o_descriptors.shape[0])} "
            f"q_ready_edges={int(self.q_ready_descriptors.shape[0])} "
            f"o_ready_edges={int(self.o_ready_descriptors.shape[0])} "
            f"max_rows={self.max_rows} "
            f"mapped_attn_dps={self._mapped_attn_dp_ranks} "
            f"decode_lane_streams={len(self._decode_lane_streams)} "
            f"moe_backend={self._moe_backend} "
            f"moe_buffers={len(self._moe_buffers)} "
            f"tp_lane_communicators={len(self._tp_lane_communicators)}",
            flush=True,
        )

    def _init_graph_buffers(self) -> None:
        graph_sizes = tuple(int(x) for x in self.worker.decode_graph_bs if int(x) > 0)
        if not graph_sizes:
            self._graph_input_ids = None
            return
        graph_fanin = self.fanin if self.role == "mlp" else 1
        max_bs = max(graph_sizes) * self.num_mb * graph_fanin
        self._graph_input_ids = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._graph_positions = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._graph_out_loc = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._graph_last_indices = torch.zeros(
            (max_bs,), dtype=torch.int64, device=self.device
        )
        dummy_table = int(self.config.max_running_req)
        self._graph_wb_table = torch.full(
            (max_bs,), dummy_table, dtype=torch.int64, device=self.device
        )
        self._graph_wb_position = torch.zeros(
            (max_bs,), dtype=torch.int64, device=self.device
        )
        self._graph_wb_batch = torch.zeros(
            (max_bs,), dtype=torch.int64, device=self.device
        )

    def _create_moe_group(self) -> None:
        self.moe_group = None
        ep_size = int(self.worker.mlp_ep_size)
        if ep_size <= 1:
            return
        topology = self.worker._topology
        coordinate_groups = build_moe_tp_column_groups(
            mlp_dp_size=int(topology.mlp_dp_size),
            mlp_tp_size=int(topology.mlp_tp_size),
            ep_size=ep_size,
        )
        groups = tuple(
            tuple(topology.mlp_worker_rank(dp_rank, tp_rank) for dp_rank, tp_rank in group)
            for group in coordinate_groups
        )
        world_rank = int(self.worker.comm_rank)
        for ranks in groups:
            group = dist.new_group(ranks=list(ranks), backend="gloo")
            if world_rank in ranks:
                self.moe_group = group
        if self.role == "mlp" and self.moe_group is None:
            raise RuntimeError("model worker was not assigned to an EG-only MoE group")
        if self.role == "mlp":
            expected_group_size = ep_size // int(topology.mlp_tp_size)
            actual_group_size = int(dist.get_world_size(self.moe_group))
            if actual_group_size != expected_group_size:
                raise RuntimeError(
                    "EG-only MoE TP-column group has the wrong size: "
                    f"actual={actual_group_size} expected={expected_group_size}"
                )

    def _allocate_and_map_transport(self) -> None:
        worker = self.worker
        if self.role == "attention":
            q_ready_writers = tuple(
                sorted(
                    {
                        head_slice.source_tp_rank
                        for head_slice in (*self.table.q_slices, *self.table.kv_slices)
                        if head_slice.destination_tp_rank == int(worker.tp_rank)
                    }
                )
            )
            if not q_ready_writers:
                raise RuntimeError("attention rank has no Q/K/V transport writers")
            self.q_owner = self.state.fabric_q
            self.q_slots = self.q_owner.tensor
            self.q_ready_owner = self.state.fabric_q_ready
            self.q_ready = self.q_ready_owner.tensor
            expected_q_shape = (
                self.transport_slots,
                self.max_rows,
                self.table.attn_local_q_heads,
                self.head_dim,
            )
            expected_q_ready_shape = (self.transport_slots, len(q_ready_writers))
            if tuple(self.q_slots.shape) != expected_q_shape:
                raise RuntimeError(
                    f"FMHA Q arena shape mismatch: {tuple(self.q_slots.shape)} != {expected_q_shape}"
                )
            if tuple(self.q_ready.shape) != expected_q_ready_shape:
                raise RuntimeError(
                    "FMHA Q-ready arena shape mismatch: "
                    f"{tuple(self.q_ready.shape)} != {expected_q_ready_shape}"
                )
            self.q_ready.zero_()
            self.o_publish_counters = torch.zeros(
                (self.transport_slots,), dtype=torch.int32, device=self.device
            )
            kv_owner = getattr(self.state, "fabric_kv", None)
            if kv_owner is None:
                raise RuntimeError("FMHA-only attention state is missing fabric-owned KV")
            self.o_owner = None
            self.o_scale_owner = None
            attention_handle = self.q_owner.handle
            if not (
                self.q_owner.allocation is self.q_ready_owner.allocation
                and self.q_owner.allocation is kv_owner.allocation
            ):
                raise RuntimeError("FMHA Q/K/V and ready views must share one fabric arena")
            local_record = _FabricRecord(
                role=self.role,
                dp_rank=int(worker.attn_dp_rank),
                tp_rank=int(worker.tp_rank),
                q_handle=attention_handle,
                q_shape=tuple(int(x) for x in self.q_slots.shape),
                q_offset=int(self.q_owner.byte_offset),
                q_ready_handle=attention_handle,
                q_ready_shape=tuple(int(x) for x in self.q_ready.shape),
                q_ready_offset=int(self.q_ready_owner.byte_offset),
                q_ready_writers=q_ready_writers,
                kv_handle=attention_handle,
                kv_shape=tuple(int(x) for x in kv_owner.tensor.shape),
                kv_offset=int(kv_owner.byte_offset),
            )
        else:
            o_ready_writers = tuple(
                sorted(
                    {
                        head_slice.source_tp_rank
                        for head_slice in self.table.o_slices
                        if head_slice.destination_tp_rank == int(worker.tp_rank)
                    }
                )
            )
            if not o_ready_writers:
                raise RuntimeError("model rank has no O transport writers")
            (
                self.o_owner,
                self.o_scale_owner,
                self.o_ready_owner,
            ) = allocate_fabric_tensors(
                (
                    (
                        (
                            self.transport_slots,
                            self.max_rows,
                            self.table.model_local_q_heads,
                            self.head_dim,
                        ),
                        torch.float8_e4m3fn,
                    ),
                    (
                        (self.transport_slots, self.o_scale_slot_elements),
                        torch.int32,
                    ),
                    (
                        (self.transport_slots, self.fanin, len(o_ready_writers)),
                        torch.int64,
                    ),
                )
            )
            self.o_slots = self.o_owner.tensor
            self.o_scale_slots = self.o_scale_owner.tensor
            self.o_ready = self.o_ready_owner.tensor
            self.o_ready.zero_()
            self.q_publish_counters = torch.zeros(
                (self.transport_slots,), dtype=torch.int32, device=self.device
            )
            self.q_owner = None
            model_handle = self.o_owner.handle
            if not (
                self.o_owner.allocation is self.o_scale_owner.allocation
                and self.o_owner.allocation is self.o_ready_owner.allocation
            ):
                raise RuntimeError(
                    "FMHA FP8 O, scales, and ready views must share one fabric arena"
                )
            local_record = _FabricRecord(
                role=self.role,
                dp_rank=int(worker.mlp_dp_rank),
                tp_rank=int(worker.tp_rank),
                o_handle=model_handle,
                o_shape=tuple(int(x) for x in self.o_slots.shape),
                o_offset=int(self.o_owner.byte_offset),
                o_scale_handle=model_handle,
                o_scale_shape=tuple(int(x) for x in self.o_scale_slots.shape),
                o_scale_offset=int(self.o_scale_owner.byte_offset),
                o_ready_handle=model_handle,
                o_ready_shape=tuple(int(x) for x in self.o_ready.shape),
                o_ready_offset=int(self.o_ready_owner.byte_offset),
                o_ready_writers=o_ready_writers,
            )
        torch.cuda.synchronize(self.device)
        records: list[_FabricRecord | None] = [None] * int(self.worker._topology.total_workers)
        dist.all_gather_object(records, local_record)
        if any(record is None for record in records):
            raise RuntimeError("incomplete AFD fabric-handle exchange")
        typed_records = tuple(record for record in records if record is not None)
        self._build_descriptors(typed_records)
        if self.role == "mlp":
            expected_import_arenas = len(self._mapped_attn_dp_ranks)
        else:
            mlp_dp_rank = self.worker._topology.mlp_dp_for_attn_dp(
                int(self.worker.attn_dp_rank)
            )
            shared_q_ready_dp = self.worker._topology.attn_dps_for_mlp_dp(
                mlp_dp_rank
            )[0]
            expected_import_arenas = 1 + int(
                int(self.worker.attn_dp_rank) != int(shared_q_ready_dp)
            )
        if len(self._imported_allocations) != expected_import_arenas:
            raise RuntimeError(
                "FMHA transport did not coalesce publications to one imported "
                f"arena per peer: actual={len(self._imported_allocations)} "
                f"expected={expected_import_arenas} role={self.role}"
            )
        log_line(
            worker.log_path,
            f"[{self.role} rank={worker.tp_rank}] fmha_transport_arenas "
            f"imports={len(self._imported_allocations)} "
            f"mapped_attn_dps={self._mapped_attn_dp_ranks}",
            flush=True,
        )
        self.timeout_record = torch.zeros((2,), dtype=torch.int64, device=self.device)
        self._attention_turn = (
            torch.zeros((1,), dtype=torch.int64, device=self.device)
            if self.role == "attention" and self._decode_lane_streams
            else None
        )
        if self._attention_turn is not None:
            torch.cuda.synchronize(self.device)

    def _record(
        self,
        records: tuple[_FabricRecord, ...],
        role: str,
        dp_rank: int,
        tp_rank: int,
    ) -> _FabricRecord:
        matches = [
            record
            for record in records
            if record.role == role
            and record.dp_rank == dp_rank
            and record.tp_rank == tp_rank
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "AFD fabric record lookup is not unique: "
                f"role={role} dp={dp_rank} tp={tp_rank} matches={len(matches)}"
            )
        return matches[0]

    def _import(
        self,
        kind: str,
        record: _FabricRecord,
    ) -> FabricTensor:
        key = (kind, record.role, record.dp_rank, record.tp_rank)
        existing = self._imports.get(key)
        if existing is not None:
            return existing
        handle = getattr(record, f"{kind}_handle")
        shape = getattr(record, f"{kind}_shape")
        if handle is None or not shape:
            raise RuntimeError(f"AFD fabric record is missing {kind} allocation")
        if kind.endswith("_ready"):
            dtype = torch.int64
        elif kind == "o_scale":
            dtype = torch.int32
        elif kind == "o":
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.bfloat16
        allocation = self._imported_allocations.get(handle)
        mapping = import_fabric_tensor(
            handle,
            shape,
            dtype=dtype,
            byte_offset=int(getattr(record, f"{kind}_offset")),
            allocation=allocation,
        )
        self._imported_allocations.setdefault(handle, mapping.allocation)
        self._imports[key] = mapping
        return mapping

    def _build_descriptors(self, records: tuple[_FabricRecord, ...]) -> None:
        worker = self.worker
        if self.role == "mlp":
            q_rows: list[list[int]] = []
            kv_rows: list[list[int]] = []
            q_ready_rows: list[list[int]] = []
            for source_index, attn_dp_rank in enumerate(
                self._mapped_attn_dp_ranks
            ):
                self._append_model_input_descriptors(
                    records,
                    attn_dp_rank=attn_dp_rank,
                    source_index=source_index,
                    include_q_ready=source_index == 0,
                    q_rows=q_rows,
                    kv_rows=kv_rows,
                    q_ready_rows=q_ready_rows,
                )
            self.q_descriptors = self._descriptor_tensor(q_rows, 7)
            self.kv_descriptors = self._descriptor_tensor(kv_rows, 8)
            self.q_ready_descriptors = self._descriptor_tensor(q_ready_rows, 3)
            self.o_descriptors = self._descriptor_tensor([], 10)
            self.o_ready_descriptors = self._descriptor_tensor([], 3)
            return

        o_rows: list[list[int]] = []
        o_ready_rows: list[list[int]] = []
        mlp_dp_rank = self.worker._topology.mlp_dp_for_attn_dp(
            int(worker.attn_dp_rank)
        )
        mapped_sources = self.worker._topology.attn_dps_for_mlp_dp(mlp_dp_rank)
        source_index = mapped_sources.index(int(worker.attn_dp_rank))
        shared_q_ready_dp = int(mapped_sources[0])
        if int(worker.attn_dp_rank) != shared_q_ready_dp:
            shared_q_ready_record = self._record(
                records,
                "attention",
                shared_q_ready_dp,
                int(worker.tp_rank),
            )
            self.q_ready = self._import(
                "q_ready", shared_q_ready_record
            ).tensor
        for head_slice in self.table.o_slices:
            if head_slice.source_tp_rank != int(worker.tp_rank):
                continue
            record = self._record(
                records, "mlp", mlp_dp_rank, head_slice.destination_tp_rank
            )
            mapping = self._import("o", record)
            scale_mapping = self._import("o_scale", record)
            o_rows.append(
                self._fp8_o_slot_descriptor(
                    mapping,
                    scale_mapping,
                    record,
                    head_slice,
                    self.table.model_local_q_heads,
                    source_index=source_index,
                    source_count=len(mapped_sources),
                )
            )
        o_ready_destinations = sorted(
            {
                head_slice.destination_tp_rank
                for head_slice in self.table.o_slices
                if head_slice.source_tp_rank == int(worker.tp_rank)
            }
        )
        for destination_tp_rank in o_ready_destinations:
            record = self._record(records, "mlp", mlp_dp_rank, destination_tp_rank)
            o_ready_rows.append(
                self._ready_descriptor(
                    self._import("o_ready", record),
                    record.o_ready_writers,
                    int(worker.tp_rank),
                    source_index=source_index,
                    source_count=len(mapped_sources),
                )
            )
        self.q_descriptors = self._descriptor_tensor([], 7)
        self.kv_descriptors = self._descriptor_tensor([], 8)
        self.q_ready_descriptors = self._descriptor_tensor([], 3)
        self.o_descriptors = self._descriptor_tensor(o_rows, 10)
        self.o_ready_descriptors = self._descriptor_tensor(o_ready_rows, 3)

    def _append_model_input_descriptors(
        self,
        records: tuple[_FabricRecord, ...],
        *,
        attn_dp_rank: int,
        source_index: int,
        include_q_ready: bool,
        q_rows: list[list[int]],
        kv_rows: list[list[int]],
        q_ready_rows: list[list[int]],
    ) -> None:
        worker = self.worker
        for head_slice in self.table.q_slices:
            if head_slice.source_tp_rank != int(worker.tp_rank):
                continue
            record = self._record(
                records,
                "attention",
                int(attn_dp_rank),
                head_slice.destination_tp_rank,
            )
            mapping = self._import("q", record)
            q_rows.append(
                self._slot_descriptor(
                    mapping,
                    head_slice,
                    self.table.attn_local_q_heads,
                    slot_rows=int(record.q_shape[1]),
                    source_index=source_index,
                )
            )
        for head_slice in self.table.kv_slices:
            if head_slice.source_tp_rank != int(worker.tp_rank):
                continue
            record = self._record(
                records,
                "attention",
                int(attn_dp_rank),
                head_slice.destination_tp_rank,
            )
            mapping = self._import("kv", record)
            cache_rows = int(record.kv_shape[2]) * int(record.kv_shape[3])
            kv_rows.append(
                [
                    int(mapping.tensor.data_ptr()),
                    head_slice.source_head_start,
                    head_slice.destination_head_start,
                    head_slice.head_count,
                    self.table.attn_local_kv_heads,
                    cache_rows,
                    int(self.model_config.num_layers),
                    int(source_index),
                ]
            )
        if not include_q_ready:
            return
        q_ready_destinations = sorted(
            {
                head_slice.destination_tp_rank
                for head_slice in (*self.table.q_slices, *self.table.kv_slices)
                if head_slice.source_tp_rank == int(worker.tp_rank)
            }
        )
        for destination_tp_rank in q_ready_destinations:
            record = self._record(
                records, "attention", int(attn_dp_rank), destination_tp_rank
            )
            q_ready_rows.append(
                self._ready_descriptor(
                    self._import("q_ready", record),
                    record.q_ready_writers,
                    int(worker.tp_rank),
                )
            )

    def _init_model_materializer_namespaces(self) -> None:
        runtime = self.worker.runtime
        total_table_slots = int(self.config.max_running_req) + 1
        attn_dp_size = int(self.worker.attn_dp_size)
        if total_table_slots % attn_dp_size:
            raise RuntimeError(
                "FMHA-only model table capacity cannot be partitioned by attention DP: "
                f"slots={total_table_slots} attn_dp={attn_dp_size}"
            )
        self._table_namespace_stride = total_table_slots // attn_dp_size
        if self._table_namespace_stride < 2:
            raise RuntimeError(
                "FMHA-only attention table namespace has no real request slot: "
                f"stride={self._table_namespace_stride}"
            )
        attrs = (
            "_free_pages",
            "_table_cached_len",
            "_table_pages",
            "_table_reqs",
        )
        self._materializer_state_attrs = attrs
        first_dp = self._mapped_attn_dp_ranks[0]
        first_state = {name: getattr(runtime, name) for name in attrs}
        self._model_materializer_states: dict[int, dict[str, Any]] = {
            first_dp: first_state
        }
        initial_free_pages = tuple(first_state["_free_pages"])
        for attn_dp_rank in self._mapped_attn_dp_ranks[1:]:
            self._model_materializer_states[attn_dp_rank] = {
                "_free_pages": list(initial_free_pages),
                "_table_cached_len": {},
                "_table_pages": {},
                "_table_reqs": {},
            }
        self._active_materializer_attn_dp = first_dp

    def _activate_model_materializer(self, attn_dp_rank: int) -> None:
        attn_dp_rank = int(attn_dp_rank)
        if attn_dp_rank == self._active_materializer_attn_dp:
            return
        runtime = self.worker.runtime
        current = self._model_materializer_states[
            self._active_materializer_attn_dp
        ]
        for name in self._materializer_state_attrs:
            current[name] = getattr(runtime, name)
        try:
            selected = self._model_materializer_states[attn_dp_rank]
        except KeyError as exc:
            raise RuntimeError(
                "FMHA-only model materializer received an unassigned attention lane: "
                f"mlp_dp={int(self.worker.mlp_dp_rank)} attn_dp={attn_dp_rank} "
                f"assigned={self._mapped_attn_dp_ranks}"
            ) from exc
        for name in self._materializer_state_attrs:
            setattr(runtime, name, selected[name])
        self._active_materializer_attn_dp = attn_dp_rank

    def _namespace_model_plan(self, plan: Any, attn_dp_rank: int) -> Any:
        stride = int(self._table_namespace_stride)
        base = int(attn_dp_rank) * stride

        def table_index(local_index: int) -> int:
            local_index = int(local_index)
            if local_index < 0 or local_index >= stride:
                raise RuntimeError(
                    "FMHA-only plan table index is outside its attention-DP namespace: "
                    f"attn_dp={attn_dp_rank} index={local_index} stride={stride}"
                )
            return base + local_index

        namespaced = copy(plan)
        namespaced.table_indices = tuple(table_index(x) for x in plan.table_indices)
        namespaced.free_table_indices = tuple(
            table_index(x) for x in plan.free_table_indices
        )
        namespaced.exec_table_indices = tuple(
            table_index(x) for x in plan.exec_table_indices
        )
        token_blocks = []
        for block in plan.token_blocks:
            namespaced_block = copy(block)
            namespaced_block.table_idx = table_index(block.table_idx)
            token_blocks.append(namespaced_block)
        namespaced.token_blocks = tuple(token_blocks)
        return namespaced

    def _plan_attn_dp_rank(self, plan: Any) -> int:
        attn_dp_rank = int(plan.attn_dp_rank)
        if attn_dp_rank < 0 or attn_dp_rank >= int(self.worker.attn_dp_size):
            raise RuntimeError(
                "FMHA-only plan has an invalid attention DP rank: "
                f"attn_dp={attn_dp_rank} size={int(self.worker.attn_dp_size)}"
            )
        if self.role == "attention" and attn_dp_rank != int(
            self.worker.attn_dp_rank
        ):
            raise RuntimeError(
                "FMHA-only attention worker received another DP lane's plan: "
                f"worker={int(self.worker.attn_dp_rank)} plan={attn_dp_rank}"
            )
        return attn_dp_rank

    def _slot_descriptor(
        self,
        mapping: FabricTensor,
        head_slice: AfdHeadSlice,
        destination_local_heads: int,
        *,
        slot_rows: int,
        source_index: int,
    ) -> list[int]:
        return [
            int(mapping.tensor.data_ptr()),
            head_slice.source_head_start,
            head_slice.destination_head_start,
            head_slice.head_count,
            destination_local_heads,
            int(slot_rows),
            int(source_index),
        ]

    def _fp8_o_slot_descriptor(
        self,
        mapping: FabricTensor,
        scale_mapping: FabricTensor,
        record: _FabricRecord,
        head_slice: AfdHeadSlice,
        destination_local_heads: int,
        *,
        source_index: int,
        source_count: int,
    ) -> list[int]:
        if (
            head_slice.source_head_start % 4
            or head_slice.destination_head_start % 4
            or head_slice.head_count % 4
        ):
            raise RuntimeError(
                "FP8 O descriptor requires four-head-aligned slices: "
                f"slice={head_slice}"
            )
        slot_rows = int(record.o_shape[1])
        expected_scale_elements = (
            div_ceil(slot_rows, 4)
            * 4
            * div_ceil(int(destination_local_heads), 4)
        )
        scale_shape = tuple(int(x) for x in record.o_scale_shape)
        if scale_shape != (self.transport_slots, expected_scale_elements):
            raise RuntimeError(
                "FP8 O scale arena shape mismatch: "
                f"shape={scale_shape} "
                f"expected={(self.transport_slots, expected_scale_elements)}"
            )
        if mapping.allocation is not scale_mapping.allocation:
            raise RuntimeError("FP8 O payload and scales require one fabric arena")
        return [
            int(mapping.tensor.data_ptr()),
            int(scale_mapping.tensor.data_ptr()),
            head_slice.source_head_start,
            head_slice.destination_head_start,
            head_slice.head_count,
            int(destination_local_heads),
            slot_rows,
            expected_scale_elements,
            int(source_index),
            int(source_count),
        ]

    @staticmethod
    def _ready_descriptor(
        mapping: FabricTensor,
        writers: tuple[int, ...],
        source_tp_rank: int,
        *,
        source_index: int = 0,
        source_count: int = 1,
    ) -> list[int]:
        try:
            writer_index = writers.index(int(source_tp_rank))
        except ValueError as error:
            raise RuntimeError(
                "AFD readiness allocation does not name its payload writer: "
                f"source_tp_rank={source_tp_rank} writers={writers}"
            ) from error
        shape = tuple(int(x) for x in mapping.tensor.shape)
        if len(shape) == 2:
            expected_shape = (2, len(writers))
            if int(source_count) != 1 or int(source_index) != 0:
                raise RuntimeError(
                    "AFD two-dimensional readiness mapping cannot address a "
                    "source axis: "
                    f"shape={shape} source_index={source_index} "
                    f"source_count={source_count}"
                )
        else:
            expected_shape = (2, int(source_count), len(writers))
        if shape != expected_shape:
            raise RuntimeError(
                "AFD readiness mapping shape does not match its writer table: "
                f"shape={shape} expected={expected_shape}"
            )
        writers_per_source = len(writers)
        return [
            int(mapping.tensor.data_ptr()),
            int(source_index) * writers_per_source + writer_index,
            int(source_count) * writers_per_source,
        ]

    def _descriptor_tensor(self, rows: list[list[int]], width: int) -> torch.Tensor:
        if not rows:
            return torch.empty((0, width), dtype=torch.int64, device=self.device)
        return torch.tensor(rows, dtype=torch.int64, device=self.device)

    def _load_model(self) -> None:
        from minisgl.engine.sample import Sampler
        from minisgl.models import load_weight
        from minisgl.models.qwen3_moe import Qwen3MoeForCausalLM
        from minisgl.layers import set_rope_device

        architectures = tuple(getattr(self.model_config, "architectures", ()))
        if "Qwen3MoeForCausalLM" not in architectures:
            raise RuntimeError(
                "FMHA-only placement currently supports Qwen3MoeForCausalLM, "
                f"got architectures={architectures}"
            )
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(self.config.dtype):
            model = Qwen3MoeForCausalLM(self.model_config)
        weights = {
            key: (
                value
                if value.dtype == torch.float8_e4m3fn
                or (value.dtype == torch.float32 and key.endswith("_scale"))
                else value.to(self.config.dtype)
            )
            for key, value in load_weight(self.config.model_path, self.device)
        }
        model.load_state_dict(weights)
        self.model = model
        self.sampler = Sampler(self.device, int(self.model_config.vocab_size))


    @staticmethod
    def _quantize_attention_o(
        o: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from minisgl.kernel import deepgemm as deep_gemm

        return deep_gemm.per_token_cast_to_fp8(
            o.contiguous(),
            use_ue8m0=True,
            gran_k=128,
            use_packed_ue8m0=True,
        )

    def _prewarm_attention_decode_kernels(self) -> None:
        """Initialize both original FMHA backends before fabric waits are armed."""
        graph_sizes = tuple(
            int(size) for size in self.worker.decode_graph_bs if int(size) > 0
        )
        if not graph_sizes:
            raise RuntimeError("FMHA-only attention prewarm requires a decode graph bucket")
        per_mb_bs = max(graph_sizes)
        batch = self._build_capture_batch(per_mb_bs)
        ctx = self.state.ctx
        for mb, sub in enumerate(batch.afd_mb_subbatches):
            stream = (
                self._decode_lane_streams[mb % self.num_lanes]
                if self._decode_lane_streams
                else self.state.stream
            )
            backend = self.state.attn_backends[mb]
            q = torch.zeros(
                (
                    per_mb_bs,
                    self.table.attn_local_q_heads,
                    self.head_dim,
                ),
                dtype=self.config.dtype,
                device=self.device,
            )
            previous_backend = ctx.attn_backend
            ctx.attn_backend = backend
            try:
                with (
                    torch.no_grad(),
                    torch.cuda.stream(stream),
                    ctx.forward_batch(sub),
                ):
                    o = backend.forward_prepared(q, 0, sub)
                    self._quantize_attention_o(o.view(per_mb_bs, -1))
                stream.synchronize()
            finally:
                ctx.attn_backend = previous_backend

    def _prewarm_model_decode_kernels(self) -> None:
        """Finish model-side lazy setup before attention can arm fabric waits."""
        graph_sizes = tuple(
            int(size) for size in self.worker.decode_graph_bs if int(size) > 0
        )
        if not graph_sizes:
            raise RuntimeError("FMHA-only model prewarm requires a decode graph bucket")
        per_mb_bs = max(graph_sizes)
        batch = self._build_capture_batch(per_mb_bs)
        sub = batch.afd_mb_subbatches[0]
        merged_per_mb_bs = int(sub.positions.numel())
        layers = list(self.model.model.layers.op_list)
        if not layers:
            raise RuntimeError("FMHA-only model prewarm requires transformer layers")

        ctx = self.state.ctx
        stream = (
            self._decode_lane_streams[0]
            if self._decode_lane_streams
            else self.state.stream
        )
        previous_buffer = getattr(ctx, "mlp_deepep_buffer", None)
        previous_valid_count = getattr(ctx, "moe_num_token_non_padded", None)
        previous_dispatch_bucket = getattr(
            ctx, "moe_deepep_dispatch_max_tokens_per_rank", None
        )
        if self._moe_buffers:
            ctx.mlp_deepep_buffer = self._moe_buffers[0]
        ctx.moe_num_token_non_padded = merged_per_mb_bs
        ctx.moe_deepep_dispatch_max_tokens_per_rank = merged_per_mb_bs
        try:
            with torch.no_grad(), torch.cuda.stream(stream):
                if getattr(self, "_mega_moe", None) is None:
                    for layer in layers:
                        runner = layer.mlp.experts.runner
                        prepare_layer = getattr(runner, "prepare_layer", None)
                        if not callable(prepare_layer):
                            raise RuntimeError(
                                "FMHA-only DeepGEMM runner cannot pre-materialize "
                                "model weights"
                            )
                        prepare_layer(layer.mlp.experts)
                        layer.mlp.experts.dispatcher.prepare_for_capture(self.device)

                with self._tp_lane_context(0), ctx.forward_batch(sub):
                    hidden = self.model.embed_input_ids(sub.input_ids)
                    _qkv, residual = layers[0].prepare_attention(hidden, None)
                    o = torch.zeros(
                        (
                            merged_per_mb_bs,
                            self.table.model_local_q_heads * self.head_dim,
                        ),
                        dtype=self.config.dtype,
                        device=self.device,
                    )
                    o_fp8, o_scale = self._quantize_attention_o(o)
                    hidden = layers[0].self_attn.finish_attention_fp8(
                        o_fp8, o_scale
                    )
                    hidden, _residual = layers[0].post_attention_layernorm.forward(
                        hidden, residual
                    )
                    self._run_model_moe(0, layers[0], hidden, lane=0)
                    # The fused router removes the model path's former cuBLAS
                    # gate GEMM. Materialize the remaining cuBLAS handle and
                    # exact-batch LM-head plan before CUDA graph capture.
                    lm_head_input = torch.zeros(
                        (
                            merged_per_mb_bs * self.num_mb,
                            int(self.model_config.hidden_size),
                        ),
                        dtype=self.config.dtype,
                        device=self.device,
                    )
                    self.model.forward_lm_head(lm_head_input)
            stream.synchronize()
        finally:
            ctx.mlp_deepep_buffer = previous_buffer
            ctx.moe_num_token_non_padded = previous_valid_count
            ctx.moe_deepep_dispatch_max_tokens_per_rank = previous_dispatch_bucket

    def _create_tp_lane_communicators(self) -> None:
        tp_size = int(self.worker.mlp_tp_size)
        if self.num_mb <= 1 or tp_size <= 1:
            return
        from minisgl.distributed.impl import (
            DistributedCommunicator,
            PyNCCLDistributedImpl,
        )
        from minisgl.kernel import init_pynccl

        plugin = DistributedCommunicator.plugins[-1]
        if not isinstance(plugin, PyNCCLDistributedImpl):
            raise RuntimeError(
                "FMHA-only multi-lane TP requires the PyNCCL communicator backend"
            )
        max_bytes = (
            int(self.config.max_forward_len)
            * int(self.model_config.hidden_size)
            * int(self.config.dtype.itemsize)
        )
        communicators = [plugin.comm]
        for _lane in range(1, self.num_lanes):
            communicators.append(
                init_pynccl(
                    tp_rank=int(self.worker.tp_rank),
                    tp_size=tp_size,
                    tp_cpu_group=self.state.tp_cpu_group,
                    max_size_bytes=max_bytes,
                )
            )
        self._tp_lane_communicators = tuple(communicators)
        self._tp_distributed_plugin = plugin
        log_line(
            self.worker.log_path,
            f"[{self.role} rank={self.worker.tp_rank}] fmha_tp_communicators "
            f"lanes={len(self._tp_lane_communicators)} tp_size={tp_size}",
            flush=True,
        )

    @contextmanager
    def _tp_lane_context(self, microbatch: int):
        communicators = getattr(self, "_tp_lane_communicators", ())
        plugin = getattr(self, "_tp_distributed_plugin", None)
        if not communicators:
            yield
            return
        if plugin is None or len(communicators) != min(2, self.num_mb):
            raise RuntimeError(
                "FMHA-only TP lane communicators are not fully initialized"
            )
        previous = plugin.comm
        plugin.comm = communicators[microbatch % len(communicators)]
        try:
            yield
        finally:
            plugin.comm = previous

    def _create_moe_buffer(self) -> None:
        if int(self.worker.mlp_ep_size) <= 1:
            if self.config.afd_moe_a2a_backend != "none":
                raise RuntimeError("EP1 FMHA-only placement requires MoE A2A backend 'none'")
            if self._moe_backend != "deepep":
                raise RuntimeError("FMHA-only MegaMoE currently requires EP > 1")
            return
        if self.config.afd_moe_a2a_backend != "deepep":
            raise RuntimeError("FMHA-only EP requires --afd-moe-a2a-backend deepep")
        if self.config.afd_moe_runner_backend != "deep_gemm":
            raise RuntimeError("FMHA-only EP requires --afd-moe-runner-backend deep_gemm")
        if self._moe_backend == "megamoe":
            self._create_megamoe()
            return
        if self._moe_backend != "deepep":
            raise RuntimeError(
                "FMHA-only MoE backend must be 'deepep' or same-rank 'megamoe', "
                f"got {self._moe_backend!r}"
            )
        from minisgl.kernel.deepep_moe import DeepEPMoeElasticBuffer

        # Low fan-in retains the aligned 24-SM DeepEP grid. Once at least
        # eight attention sources feed one model lane, use the measured one-cluster
        # policy and give DeepGEMM the complementary device-width launch budget.
        communication_sms = (
            DeepEPMoeElasticBuffer.HIGH_FANIN_OVERLAP_NUM_SMS
            if self.fanin >= DeepEPMoeElasticBuffer.HIGH_FANIN_MIN_SOURCES
            else DeepEPMoeElasticBuffer.OVERLAP_NUM_SMS
        )
        total_sms = int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )
        compute_sms = total_sms - communication_sms
        if compute_sms <= 0:
            raise RuntimeError(
                "FMHA-only DeepEP overlap requires more device SMs than its "
                f"{communication_sms}-SM communication grid; device has {total_sms}"
            )
        # The launch budget already excludes the communication SMs. Clear the
        # MegaMoE-only heuristic reservation to avoid subtracting them twice.
        os.environ["MINISGL_MEGAMOE_AG_SMS"] = "0"
        from minisgl.kernel import deepgemm as deep_gemm

        deep_gemm.set_num_sms(compute_sms)
        if int(deep_gemm.get_num_sms()) != compute_sms:
            raise RuntimeError(
                "DeepGEMM rejected the FMHA-only compute-SM budget: "
                f"requested={compute_sms} actual={int(deep_gemm.get_num_sms())}"
            )

        assert self.moe_group is not None
        if self.num_mb > 1:
            # Assign each complete microbatch, including DeepEP dispatch/combine,
            # to one of two alternating main streams. Bind one buffer to each
            # lane rather than creating hidden internal communication streams.
            os.environ["MINISGL_DEEPEP_USE_CURRENT_STREAM"] = "1"
            os.environ.pop("MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM", None)
        local_experts = div_ceil(
            int(self.model_config.num_experts), int(self.worker.mlp_ep_size)
        )
        group_experts = local_experts * int(dist.get_world_size(self.moe_group))
        buffer_count = self.num_lanes
        buffers = []
        for lane in range(buffer_count):
            stream_context = (
                torch.cuda.stream(self._decode_lane_streams[lane])
                if self.num_mb > 1
                else nullcontext()
            )
            with stream_context:
                buffer = DeepEPMoeElasticBuffer(
                    group=self.moe_group,
                    num_max_dispatch_tokens_per_rank=self.max_rows,
                    hidden_size=int(self.model_config.hidden_size),
                    num_experts=group_experts,
                    top_k=int(self.model_config.num_experts_per_tok),
                    overlap_num_sms=communication_sms,
                )
            if self.num_mb > 1:
                buffer_stream = buffer.runtime.get_comm_stream()
                buffer_identity = _stream_identity(buffer_stream)
                main_identity = _stream_identity(self._decode_lane_streams[lane])
                if buffer_identity != main_identity:
                    raise RuntimeError(
                        "FMHA-only DeepEP stream is not its microbatch main stream: "
                        f"lane={lane} buffer={buffer_identity} main={main_identity}"
                    )
            buffers.append(buffer)
        self._moe_buffers = tuple(buffers)
        self.state.ctx.mlp_deepep_buffer = self._moe_buffers[0]
        log_line(
            self.worker.log_path,
            f"[{self.role} rank={self.worker.tp_rank}] fmha_deepep_buffers "
            f"lanes={len(self._moe_buffers)} "
            f"communication_sms={communication_sms} compute_sms={compute_sms} "
            "comm_stream_binding="
            f"{'microbatch_main' if self.num_mb > 1 else 'default'}",
            flush=True,
        )

    def _create_megamoe(self) -> None:
        """Transform FP8 experts and allocate one symmetric buffer per lane."""
        from minisgl.kernel import deepgemm as deep_gemm
        from minisgl.moe.megamoe_afd import MegaMoEAfdAdapter

        assert self.moe_group is not None
        layers = list(self.model.model.layers.op_list)
        if not layers:
            raise RuntimeError("FMHA-only MegaMoE requires transformer layers")
        experts0 = layers[0].mlp.experts
        adapter = MegaMoEAfdAdapter(
            group=self.moe_group,
            num_experts=int(self.model_config.num_experts),
            num_local_experts=int(experts0.local_num_experts),
            hidden_size=int(self.model_config.hidden_size),
            intermediate_size=int(self.model_config.moe_intermediate_size),
            top_k=int(self.model_config.num_experts_per_tok),
            num_max_tokens_per_rank=self.max_rows,
            num_lanes=self.num_lanes,
            tp_size=int(experts0.tp_size),
            tp_rank=int(self.worker.tp_rank),
            gate_renormalize=bool(self.model_config.norm_topk_prob),
        )
        for layer_id, layer in enumerate(layers):
            experts = layer.mlp.experts
            adapter.register_layer_weights(layer_id, experts)
            experts.gate_up_proj = None
            experts.gate_up_proj_scale = None
            experts.down_proj = None
            experts.down_proj_scale = None
            if layer_id % 4 == 3:
                torch.cuda.empty_cache()
        adapter.allocate_buffers()
        self._mega_moe = adapter

        total_sms = int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )
        # MegaMoE launches 2-CTA clusters.  The peer lane's one-CTA O-ready
        # waiter is intentionally resident during compute, so leave one whole
        # cluster available rather than stranding the final MegaMoE cluster.
        reserved_sms = 2
        compute_sms = total_sms - reserved_sms
        if compute_sms <= 0 or compute_sms % 2:
            raise RuntimeError(
                "MegaMoE requires a positive even SM grid after reserving one "
                f"2-CTA cluster: device_sms={total_sms} reserved_sms={reserved_sms}"
            )
        deep_gemm.set_num_sms(compute_sms)
        if int(deep_gemm.get_num_sms()) != compute_sms:
            raise RuntimeError(
                "DeepGEMM rejected the MegaMoE compute-SM budget: "
                f"requested={compute_sms} actual={int(deep_gemm.get_num_sms())}"
            )
        torch.cuda.empty_cache()
        log_line(
            self.worker.log_path,
            f"[{self.role} rank={self.worker.tp_rank}] fmha_megamoe_buffers "
            f"precision={adapter.precision} lanes={len(adapter.buffers)} "
            f"group_ranks={adapter.group_size} group_experts={adapter.num_experts} "
            f"compute_sms={compute_sms} reserved_sms={reserved_sms}",
            flush=True,
        )

    def _run_model_moe(
        self,
        layer_id: int,
        layer: Any,
        hidden: torch.Tensor,
        *,
        lane: int,
    ) -> torch.Tensor:
        mega_moe = getattr(self, "_mega_moe", None)
        if mega_moe is None:
            prepared = layer.mlp.prepare_deepep(hidden)
            return layer.mlp.finish_deepep(prepared)
        output = mega_moe.forward(
            layer_id,
            hidden,
            layer.mlp.gate.weight,
            lane=lane,
            num_token_non_padded=self.state.ctx.moe_num_token_non_padded,
        )
        if int(layer.mlp.experts.tp_size) > 1:
            output = layer.mlp.experts._comm.all_reduce(output)
        return output

    def warmup_decode_graphs(self, bs_list: tuple[int, ...]) -> list[int]:
        if not self.worker.enable_decode_graph:
            raise RuntimeError(
                "FMHA-only placement requires both attention and model decode graphs"
            )
        captured: list[int] = []
        rounds_per_step = int(self.model_config.num_layers) * self.num_mb
        slot_bases = tuple(
            sorted(
                {
                    (step * rounds_per_step) % self.transport_slots
                    for step in range(self.transport_slots)
                }
            )
        )
        for bs in sorted(int(x) for x in bs_list if int(x) > 0):
            for slot_base in slot_bases:
                self._capture_decode_graph(bs, slot_base)
            captured.append(bs)
        if not captured:
            raise RuntimeError("FMHA-only placement requires at least one decode graph bucket")
        return captured

    def _build_capture_batch(self, per_mb_bs: int) -> Any:
        from minisgl.core import Batch

        dummy_req = self.worker.runtime._dummy_req
        if dummy_req is None or self._graph_input_ids is None:
            raise RuntimeError("FMHA-only graph buffers are not initialized")
        source_count = self.fanin if self.role == "mlp" else 1
        per_mb_rows = int(per_mb_bs) * source_count
        bs = per_mb_rows * self.num_mb
        reqs = [dummy_req] * int(bs)
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = reqs
        batch.input_ids = self._graph_input_ids[:bs]
        batch.positions = self._graph_positions[:bs]
        batch.out_loc = self._graph_out_loc[:bs]
        batch.afd_num_sampling = int(bs)
        batch.afd_last_indices = self._graph_last_indices[:bs]
        batch.afd_writeback = {
            "table_indices": self._graph_wb_table[:bs],
            "write_positions": self._graph_wb_position[:bs],
            "batch_indices": self._graph_wb_batch[:bs],
        }
        offsets = tuple(mb * per_mb_rows for mb in range(self.num_mb + 1))
        batch.afd_microbatch_offsets = offsets
        batch.afd_microbatch_token_offsets = offsets
        subs = []
        metadata = []
        for mb in range(self.num_mb):
            start, end = offsets[mb], offsets[mb + 1]
            sub_reqs = reqs[start:end]
            sub = Batch(reqs=sub_reqs, phase="decode")
            sub.padded_reqs = sub_reqs
            sub.input_ids = self._graph_input_ids[start:end]
            sub.positions = self._graph_positions[start:end]
            sub.out_loc = self._graph_out_loc[start:end]
            if self.role == "attention":
                backend = self.state.attn_backends[mb]
                if mb not in self._capture_backends_initialized:
                    backend.init_capture_graph(
                        max_seq_len=self.state.max_seq_len,
                        bs_list=self.worker.decode_graph_bs,
                    )
                    self._capture_backends_initialized.add(mb)
                backend.prepare_for_capture(sub)
                metadata.append(sub.attn_metadata)
            else:
                source_offsets = tuple(
                    source * int(per_mb_bs) for source in range(source_count + 1)
                )
                sub.afd_fmha_source_token_offsets = source_offsets
                sub.afd_fmha_source_token_offsets_device = torch.tensor(
                    source_offsets,
                    dtype=torch.int64,
                    device=self.device,
                )
            subs.append(sub)
        batch.afd_mb_subbatches = subs
        if metadata:
            batch.attn_metadata = metadata[0]
            batch.attn_metadata_mbs = metadata
        return batch

    def _capture_decode_graph(self, per_mb_bs: int, slot_base: int) -> None:
        from .cuda_graph_utils import capture_cuda_graph

        graph_fanin = self.fanin if self.role == "mlp" else 1
        bs = int(per_mb_bs) * self.num_mb * graph_fanin
        key = (bs, int(slot_base))
        if key in self._decode_graphs:
            return
        batch = self._build_capture_batch(per_mb_bs)
        sampling_batch = (
            self.worker.runtime.build_sampling_batch(bs) if self.role == "mlp" else None
        )
        valid_token_counts_host = (
            tuple(
                torch.tensor(
                    [int(per_mb_bs) * graph_fanin],
                    dtype=torch.int64,
                    pin_memory=True,
                )
                for _ in range(self.num_mb)
            )
            if self.role == "mlp"
            else ()
        )
        valid_token_counts = tuple(
            torch.empty((1,), dtype=torch.int64, device=self.device)
            for _ in valid_token_counts_host
        )
        # Serialize the initial QKV/RoPE work, then hand exclusive local
        # compute ownership across FFN(L-1)+QKV/RoPE(L) units. Publication
        # stays after the event so the communication tail overlaps peer compute.
        bootstrap_compute_events = (
            tuple(torch.cuda.Event() for _ in range(min(2, self.num_mb)))
            if self.role == 'mlp' and self._decode_lane_streams
            else ()
        )
        compute_done_events = (
            tuple(
                tuple(torch.cuda.Event() for _ in range(self.num_mb))
                for _ in range(int(self.model_config.num_layers))
            )
            if self.role == 'mlp' and self._decode_lane_streams
            else ()
        )
        output: list[torch.Tensor | None] = [None]

        def body() -> None:
            if self.role == "attention":
                self._run_attention_body(
                    batch,
                    slot_base,
                    lane_streams=self._decode_lane_streams,
                )
            else:
                assert sampling_batch is not None
                output[0] = self._run_model_graph_body(
                    batch,
                    slot_base=slot_base,
                    sampling_batch=sampling_batch,
                    valid_token_counts=valid_token_counts,
                    lane_streams=self._decode_lane_streams,
                    bootstrap_compute_events=bootstrap_compute_events,
                    compute_done_events=compute_done_events,
                )

        with torch.cuda.stream(self.state.stream):
            for device_count, host_count in zip(
                valid_token_counts,
                valid_token_counts_host,
                strict=True,
            ):
                device_count.copy_(host_count, non_blocking=True)
        # Both roles were fully prewarmed before the global ready barrier.
        # Do not execute a redundant fabric-driven body here: a resident Q/O
        # wait would create a cross-role cycle before the peer enters capture.
        graph_root_stream = (
            self._decode_lane_streams[0]
            if self._decode_lane_streams
            else self.state.stream
        )
        graph_peer_stream = (
            self._decode_lane_streams[1]
            if len(self._decode_lane_streams) > 1
            else graph_root_stream
        )
        graph, self._graph_pool = capture_cuda_graph(
            device=self.device,
            engine_stream=graph_root_stream,
            comm_stream=graph_peer_stream,
            overlap_comm=bool(self._decode_lane_streams),
            pool=self._graph_pool,
            fn=body,
        )
        self._decode_graphs[key] = _FmhaDecodeGraph(
            graph=graph,
            batch=batch,
            next_tokens=output[0],
            valid_token_counts=valid_token_counts,
            valid_token_counts_host=valid_token_counts_host,
            bootstrap_compute_events=bootstrap_compute_events,
            compute_done_events=compute_done_events,
        )
        schedule = (
            "mb_exclusive_compute_ping_pong"
            if self._decode_lane_streams
            else "engine_serial"
        )
        log_line(
            self.worker.log_path,
            f"[{self.role} rank={self.worker.tp_rank}] fmha_decode_graph "
            f"captured total_bs={bs} per_mb_bs={int(per_mb_bs)} "
            f"num_mb={self.num_mb} slot_base={slot_base} "
            f"main_streams={len(self._decode_lane_streams)} "
            f"graph_root={'mb0' if self._decode_lane_streams else 'engine'} "
            f"schedule={schedule}",
            flush=True,
        )

    def _run_attention_body(
        self,
        batch: Any,
        slot_base: int,
        *,
        lane_streams: tuple[torch.cuda.Stream, ...] = (),
    ) -> None:
        ctx = self.state.ctx
        subs = self._microbatches(batch)
        self._validate_lane_streams(lane_streams)
        num_layers = int(self.model_config.num_layers)
        attention_turn = self._attention_turn if lane_streams else None
        if lane_streams and attention_turn is None:
            raise RuntimeError(
                "FMHA-only pipelined attention is missing its device turn"
            )

        # Keep the accepted MB2 schedule: both one-CTA Q waits are resident
        # before either FMHA starts, and each lane arms its next-layer wait as
        # soon as it publishes O.  This lets the peer Q publication wake a
        # resident waiter while the other lane is still pushing O.  More than
        # two logical microbatches share two physical streams, so they use the
        # inline wait below rather than queueing a future turn ahead of work on
        # the same stream.
        prearm_q_waits = bool(lane_streams and self.num_mb == 2)
        if prearm_q_waits:
            for mb, sub in enumerate(subs):
                with torch.cuda.stream(lane_streams[mb]):
                    self._wait_attention_q(
                        sub,
                        epoch=int(slot_base) + mb,
                        turn=attention_turn,
                        expected_turn=mb,
                    )

        total_rounds = num_layers * self.num_mb
        for layer in range(num_layers):
            for mb, sub in enumerate(subs):
                lane_stream = (
                    lane_streams[mb % len(lane_streams)] if lane_streams else None
                )
                stream_context = (
                    torch.cuda.stream(lane_stream)
                    if lane_streams
                    else nullcontext()
                )
                with stream_context:
                    previous = ctx.attn_backend
                    ctx.attn_backend = self.state.attn_backends[mb]
                    try:
                        with ctx.forward_batch(sub):
                            epoch = int(slot_base) + layer * self.num_mb + mb
                            turn_index = layer * self.num_mb + mb
                            self.run_attention_fmha(
                                sub,
                                layer=layer,
                                epoch=epoch,
                                lane_stream=lane_stream,
                                q_wait_already_enqueued=prearm_q_waits,
                                enqueue_next_q_wait=bool(
                                    prearm_q_waits and layer + 1 < num_layers
                                ),
                                attention_turn=attention_turn,
                                expected_turn=(turn_index if lane_streams else None),
                                next_turn=(
                                    (
                                        0
                                        if turn_index + 1 == total_rounds
                                        else turn_index + 1
                                    )
                                    if lane_streams
                                    else None
                                ),
                            )
                    finally:
                        ctx.attn_backend = previous

    def _validate_lane_streams(
        self, lane_streams: tuple[torch.cuda.Stream, ...]
    ) -> None:
        expected = min(2, self.num_mb)
        if lane_streams and len(lane_streams) != expected:
            raise RuntimeError(
                "FMHA-only decode must use one stream for one microbatch or "
                "exactly two streams for multiple microbatches: "
                f"streams={len(lane_streams)} expected={expected} "
                f"num_mb={self.num_mb}"
            )

    def _microbatches(self, batch: Any) -> list[Any]:
        subs = getattr(batch, "afd_mb_subbatches", None)
        if subs is None or len(subs) != self.num_mb:
            raise RuntimeError(
                "FMHA-only batch is missing its declared microbatch views: "
                f"actual={0 if subs is None else len(subs)} expected={self.num_mb}"
            )
        token_offsets = tuple(int(x) for x in batch.afd_microbatch_token_offsets)
        if len(token_offsets) != self.num_mb + 1:
            raise RuntimeError(
                "FMHA-only token offsets do not match the microbatch count: "
                f"offsets={token_offsets} num_mb={self.num_mb}"
            )
        if token_offsets[0] != 0 or token_offsets[-1] != int(batch.positions.numel()):
            raise RuntimeError(
                "FMHA-only token offsets do not cover the materialized batch: "
                f"offsets={token_offsets} rows={int(batch.positions.numel())}"
            )
        return list(subs)

    def _run_model_forward(
        self,
        batch: Any,
        *,
        slot_base: int,
        real_token_counts: tuple[Any, ...] | None = None,
        dispatch_bucket: int | None = None,
        lane_streams: tuple[torch.cuda.Stream, ...] = (),
        bootstrap_compute_events: tuple[torch.cuda.Event, ...] = (),
        compute_done_events: tuple[tuple[torch.cuda.Event, ...], ...] = (),
    ) -> torch.Tensor:
        ctx = self.state.ctx
        subs = self._microbatches(batch)
        self._validate_lane_streams(lane_streams)
        token_offsets = tuple(int(x) for x in batch.afd_microbatch_token_offsets)
        if real_token_counts is None:
            real_token_counts = tuple(int(item.positions.numel()) for item in subs)
        if len(real_token_counts) != self.num_mb:
            raise RuntimeError(
                "FMHA-only real-token counts do not match the microbatch count: "
                f"counts={real_token_counts} num_mb={self.num_mb}"
            )
        if dispatch_bucket is None:
            dispatch_bucket = max(int(item.positions.numel()) for item in subs)

        layers = list(self.model.model.layers.op_list)
        if not layers:
            raise RuntimeError("FMHA-only model has no transformer layers")
        if lane_streams:
            expected_bootstrap_events = min(2, self.num_mb)
            if len(bootstrap_compute_events) != expected_bootstrap_events:
                raise RuntimeError(
                    'FMHA-only bootstrap compute-event count does not match lanes: '
                    f'events={len(bootstrap_compute_events)} '
                    f'expected={expected_bootstrap_events}'
                )
            if len(compute_done_events) != len(layers) or any(
                len(layer_events) != self.num_mb
                for layer_events in compute_done_events
            ):
                raise RuntimeError(
                    'FMHA-only compute-event grid does not match layers/microbatches: '
                    f'layers={len(compute_done_events)}/{len(layers)} '
                    f'widths={tuple(len(events) for events in compute_done_events)} '
                    f'num_mb={self.num_mb}'
                )
            return self._run_model_forward_ping_pong(
                batch=batch,
                subs=subs,
                token_offsets=token_offsets,
                layers=layers,
                slot_base=slot_base,
                real_token_counts=real_token_counts,
                dispatch_bucket=int(dispatch_bucket),
                lane_streams=lane_streams,
                bootstrap_compute_events=bootstrap_compute_events,
                compute_done_events=compute_done_events,
            )
        if bootstrap_compute_events or compute_done_events:
            raise RuntimeError(
                "FMHA-only ping-pong synchronization requires microbatch streams"
            )

        with ctx.forward_batch(batch):
            hidden_full = self.model.embed_input_ids(batch.input_ids)
        hidden = [
            hidden_full[token_offsets[mb] : token_offsets[mb + 1]]
            for mb in range(self.num_mb)
        ]
        residual: list[Any | None] = [None] * self.num_mb

        def publish_layer_qkv(layer_id: int, mb: int) -> None:
            layer = layers[layer_id]
            sub = subs[mb]
            with self._tp_lane_context(mb), ctx.forward_batch(sub):
                qkv, residual[mb] = layer.prepare_attention(
                    hidden[mb], residual[mb]
                )
                epoch = int(slot_base) + layer_id * self.num_mb + mb
                self.publish_model_qkv(qkv, sub, layer=layer_id, epoch=epoch)

        # Eager prefill shares the same two transport slots as decode. Keep at
        # most one live round per physical lane: once a round consumes its O
        # payload and completes FFN, publish the round that reuses its slot.
        # For MB2 this is exactly the original two-stage wavefront; larger
        # logical microbatch counts roll over the same two slots without
        # overwriting an in-flight payload.
        window = min(2, self.num_mb)
        total_rounds = len(layers) * self.num_mb
        for round_index in range(min(window, total_rounds)):
            layer_id, mb = divmod(round_index, self.num_mb)
            publish_layer_qkv(layer_id, mb)

        for round_index in range(total_rounds):
            layer_id, mb = divmod(round_index, self.num_mb)
            layer = layers[layer_id]
            sub = subs[mb]
            epoch = int(slot_base) + round_index
            ctx.moe_num_token_non_padded = real_token_counts[mb]
            ctx.moe_deepep_dispatch_max_tokens_per_rank = int(dispatch_bucket)
            previous_buffer = getattr(ctx, "mlp_deepep_buffer", None)
            moe_buffers = getattr(self, "_moe_buffers", ())
            if moe_buffers:
                ctx.mlp_deepep_buffer = moe_buffers[mb % len(moe_buffers)]
            try:
                with self._tp_lane_context(mb), ctx.forward_batch(sub):
                    o_fp8, o_scale = self.receive_attention_o(sub, epoch)
                    hidden[mb] = layer.self_attn.finish_attention_fp8(
                        o_fp8, o_scale
                    )
                    hidden[mb], residual[mb] = (
                        layer.post_attention_layernorm.forward(
                            hidden[mb], residual[mb]
                        )
                    )
                    hidden[mb] = (
                        self._run_model_moe(
                            layer_id, layer, hidden[mb], lane=mb % self.num_lanes
                        )
                        if getattr(self, "_mega_moe", None) is not None
                        else layer.mlp.forward(hidden[mb])
                    )
                next_round = round_index + window
                if next_round < total_rounds:
                    next_layer_id, next_mb = divmod(next_round, self.num_mb)
                    publish_layer_qkv(next_layer_id, next_mb)
            finally:
                ctx.mlp_deepep_buffer = previous_buffer

        finals = []
        for mb, sub in enumerate(subs):
            with self._tp_lane_context(mb), ctx.forward_batch(sub):
                finals.append(self.model.finalize_hidden(hidden[mb], residual[mb]))
        return finals[0] if self.num_mb == 1 else torch.cat(finals, dim=0)

    def _run_model_forward_ping_pong(
        self,
        *,
        batch: Any,
        subs: list[Any],
        token_offsets: tuple[int, ...],
        layers: list[Any],
        slot_base: int,
        real_token_counts: tuple[Any, ...],
        dispatch_bucket: int,
        lane_streams: tuple[torch.cuda.Stream, ...],
        bootstrap_compute_events: tuple[torch.cuda.Event, ...],
        compute_done_events: tuple[tuple[torch.cuda.Event, ...], ...],
    ) -> torch.Tensor:
        '''Alternate exclusive FFN(L-1)+QKV/RoPE(L) compute units.'''
        ctx = self.state.ctx
        num_lanes = len(lane_streams)
        moe_buffers = getattr(self, '_moe_buffers', ())
        if moe_buffers and len(moe_buffers) != num_lanes:
            raise RuntimeError(
                'FMHA-only pipelining requires one DeepEP buffer per stream: '
                f'buffers={len(moe_buffers)} streams={num_lanes}'
            )
        hidden: list[torch.Tensor | None] = [None] * self.num_mb
        residual: list[Any | None] = [None] * self.num_mb

        @contextmanager
        def lane_context(mb: int, lane: int):
            previous_buffer = getattr(ctx, 'mlp_deepep_buffer', None)
            ctx.moe_num_token_non_padded = real_token_counts[mb]
            ctx.moe_deepep_dispatch_max_tokens_per_rank = int(dispatch_bucket)
            if moe_buffers:
                ctx.mlp_deepep_buffer = moe_buffers[lane]
            try:
                with (
                    torch.cuda.stream(lane_streams[lane]),
                    self._tp_lane_context(lane),
                    ctx.forward_batch(subs[mb]),
                ):
                    yield
            finally:
                ctx.mlp_deepep_buffer = previous_buffer

        def prepare_and_publish_qkv(
            round_index: int, handoff_event: torch.cuda.Event
        ) -> None:
            layer_id, mb = divmod(round_index, self.num_mb)
            lane = round_index % num_lanes
            layer = layers[layer_id]
            sub = subs[mb]
            lane_stream = lane_streams[lane]
            epoch = int(slot_base) + round_index
            with lane_context(mb, lane):
                if layer_id == 0 and hidden[mb] is None:
                    hidden[mb] = self.model.embed_input_ids(
                        batch.input_ids[
                            token_offsets[mb] : token_offsets[mb + 1]
                        ]
                    )
                if hidden[mb] is None:
                    raise RuntimeError(
                        f'FMHA-only round {round_index} has no hidden state'
                    )
                qkv, residual[mb] = layer.prepare_attention(
                    hidden[mb], residual[mb]
                )
                q, k, v = self.prepare_model_qkv(qkv, sub, layer=layer_id)
                handoff_event.record(lane_stream)
                self.publish_model_qkv_payload(
                    q,
                    k,
                    v,
                    sub,
                    layer=layer_id,
                    epoch=epoch,
                )

        total_rounds = len(layers) * self.num_mb
        window = min(num_lanes, total_rounds)

        # Bootstrap the two transport slots. Only QKV publication may overlap
        # the peer lane initial QKV/RoPE compute.
        for round_index in range(window):
            lane = round_index % num_lanes
            if round_index > 0:
                lane_streams[lane].wait_event(
                    bootstrap_compute_events[round_index - 1]
                )
            prepare_and_publish_qkv(
                round_index, bootstrap_compute_events[round_index]
            )

        previous_compute_event = bootstrap_compute_events[window - 1]
        for round_index in range(total_rounds):
            layer_id, mb = divmod(round_index, self.num_mb)
            lane = round_index % num_lanes
            layer = layers[layer_id]
            sub = subs[mb]
            lane_stream = lane_streams[lane]
            epoch = int(slot_base) + round_index
            compute_done_event = compute_done_events[layer_id][mb]

            # The O wait can be resident while the peer computes. Heavy local
            # work starts only after both O-ready and prior-compute-done.
            with lane_context(mb, lane):
                o_fp8, o_scale = self.receive_attention_o(sub, epoch)
                lane_stream.wait_event(previous_compute_event)
                hidden[mb] = layer.self_attn.finish_attention_fp8(
                    o_fp8, o_scale
                )
                hidden[mb], residual[mb] = (
                    layer.post_attention_layernorm.forward(
                        hidden[mb], residual[mb]
                    )
                )
                hidden[mb] = self._run_model_moe(
                    layer_id, layer, hidden[mb], lane=lane
                )

            next_round = round_index + window
            if next_round < total_rounds:
                if next_round % num_lanes != lane:
                    raise RuntimeError(
                        'FMHA-only transport slot changed physical lane'
                    )
                prepare_and_publish_qkv(next_round, compute_done_event)
            else:
                compute_done_event.record(lane_stream)
            previous_compute_event = compute_done_event

        # Final norm stays on the last compute owner so it is physically
        # stream-ordered after the final combine epilogue. Lane zero consumes
        # only the fully materialized result for sampling.
        final_stream = lane_streams[(total_rounds - 1) % num_lanes]
        finals = []
        with torch.cuda.stream(final_stream):
            for mb in range(self.num_mb):
                with self._tp_lane_context(mb), ctx.forward_batch(subs[mb]):
                    if hidden[mb] is None:
                        raise RuntimeError(
                            f'FMHA-only microbatch {mb} has no final hidden state'
                        )
                    finals.append(
                        self.model.finalize_hidden(hidden[mb], residual[mb])
                    )
            result = finals[0] if self.num_mb == 1 else torch.cat(finals, dim=0)
        root_stream = lane_streams[0]
        if final_stream is not root_stream:
            root_stream.wait_stream(final_stream)
        return result

    def _run_model_graph_body(
        self,
        batch: Any,
        *,
        slot_base: int,
        sampling_batch: Any,
        valid_token_counts: tuple[torch.Tensor, ...],
        lane_streams: tuple[torch.cuda.Stream, ...] = (),
        bootstrap_compute_events: tuple[torch.cuda.Event, ...] = (),
        compute_done_events: tuple[tuple[torch.cuda.Event, ...], ...] = (),
    ) -> torch.Tensor:
        ctx = self.state.ctx
        final = self._run_model_forward(
            batch,
            slot_base=slot_base,
            real_token_counts=valid_token_counts,
            lane_streams=lane_streams,
            bootstrap_compute_events=bootstrap_compute_events,
            compute_done_events=compute_done_events,
        )
        sampling_stream_context = (
            torch.cuda.stream(lane_streams[0]) if lane_streams else nullcontext()
        )
        with sampling_stream_context:
            selected = final.index_select(0, batch.afd_last_indices)
            with ctx.forward_batch(sampling_batch):
                logits = self.model.forward_lm_head(selected)
                next_tokens = torch.argmax(logits, dim=-1).to(torch.int32)
            wb = batch.afd_writeback
            self.worker.runtime.token_pool[
                wb["table_indices"], wb["write_positions"]
            ] = next_tokens[wb["batch_indices"]]
        return next_tokens

    @staticmethod
    def _copy_and_pad(
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        size: int,
        pad: int,
    ) -> None:
        count = int(source.numel())
        if count > size:
            raise RuntimeError(f"graph metadata has {count} entries for bucket {size}")
        if count:
            destination[:count].copy_(source)
        if count < size:
            destination[count:size].fill_(int(pad))

    def _prepare_graph_replay(
        self,
        entry: _FmhaDecodeGraph,
        batch: Any,
        plan: Any,
        bs: int,
    ) -> None:
        assert self._graph_input_ids is not None
        self._graph_positions[:bs].copy_(batch.positions)
        self._graph_out_loc[:bs].copy_(batch.out_loc)
        if self.role == "attention":
            capture_batch = entry.batch
            capture_batch.reqs = batch.reqs
            capture_batch.padded_reqs = batch.padded_reqs
            capture_subs = self._microbatches(capture_batch)
            real_subs = self._microbatches(batch)
            for mb, (capture_sub, real_sub) in enumerate(
                zip(capture_subs, real_subs, strict=True)
            ):
                capture_sub.reqs = real_sub.reqs
                capture_sub.padded_reqs = real_sub.padded_reqs
                capture_sub.attn_metadata = real_sub.attn_metadata
                self.state.attn_backends[mb].prepare_for_replay(capture_sub)
            return
        valid_counts = validate_decode_microbatch_token_counts(
            microbatch_real_token_counts=tuple(
                int(value) for value in plan.microbatch_real_token_counts
            ),
            microbatch_token_offsets=tuple(
                int(value) for value in plan.microbatch_token_offsets
            ),
        )
        if not (
            len(entry.valid_token_counts)
            == len(entry.valid_token_counts_host)
            == len(valid_counts)
        ):
            raise RuntimeError("decode graph is missing per-microbatch token counts")
        capture_subs = self._microbatches(entry.batch)
        real_subs = self._microbatches(batch)
        for mb, (capture_sub, real_sub) in enumerate(
            zip(capture_subs, real_subs, strict=True)
        ):
            capture_offsets = tuple(
                int(value) for value in capture_sub.afd_fmha_source_token_offsets
            )
            real_offsets = tuple(
                int(value) for value in real_sub.afd_fmha_source_token_offsets
            )
            capture_spans = tuple(
                right - left
                for left, right in zip(capture_offsets, capture_offsets[1:])
            )
            real_spans = tuple(
                right - left for left, right in zip(real_offsets, real_offsets[1:])
            )
            if real_spans != capture_spans:
                raise RuntimeError(
                    "FMHA-only model decode source layout does not match its graph: "
                    f"mb={mb} actual={real_spans} captured={capture_spans}"
                )
        for device_count, host_count, valid_count in zip(
            entry.valid_token_counts,
            entry.valid_token_counts_host,
            valid_counts,
            strict=True,
        ):
            host_count[0] = valid_count
            device_count.copy_(host_count, non_blocking=True)
        self._graph_input_ids[:bs].copy_(batch.input_ids)
        self._copy_and_pad(
            self._graph_last_indices, batch.afd_last_indices, size=bs, pad=0
        )
        wb = batch.afd_writeback
        self._copy_and_pad(
            self._graph_wb_table,
            wb["table_indices"],
            size=bs,
            pad=int(self.config.max_running_req),
        )
        self._copy_and_pad(
            self._graph_wb_position, wb["write_positions"], size=bs, pad=0
        )
        self._copy_and_pad(
            self._graph_wb_batch, wb["batch_indices"], size=bs, pad=0
        )

    def _replay_decode_graph(self, batch: Any, plan: Any) -> torch.Tensor | None:
        if not bool(plan.use_decode_graph):
            raise RuntimeError(
                f"FMHA-only decode step {plan.step_id} was not assigned a graph bucket"
            )
        if plan.sampling.temperatures:
            raise RuntimeError("FMHA-only captured sampling currently requires greedy decoding")
        bs = int(batch.positions.numel())
        slot_base = (
            int(plan.step_id) * int(self.model_config.num_layers) * self.num_mb
        ) % self.transport_slots
        entry = self._decode_graphs.get((bs, slot_base))
        if entry is None:
            raise RuntimeError(
                f"FMHA-only decode graph is missing bucket={bs} slot_base={slot_base}"
            )
        with torch.cuda.stream(self.state.stream):
            if self._graph_output_free is not None:
                self.state.stream.wait_event(self._graph_output_free)
            self._prepare_graph_replay(entry, batch, plan, bs)
            entry.graph.replay()
        if self.role == "attention":
            return None
        if entry.next_tokens is None:
            raise RuntimeError("FMHA-only model graph has no sampled-token output")
        self.worker.runtime.stream.wait_stream(self.state.stream)
        return entry.next_tokens[: int(batch.afd_num_sampling)]

    def publish_model_qkv(
        self,
        qkv: torch.Tensor,
        batch: Any,
        *,
        layer: int,
        epoch: int,
    ) -> None:
        q, k, v = self.prepare_model_qkv(qkv, batch, layer=layer)
        self.publish_model_qkv_payload(
            q,
            k,
            v,
            batch,
            layer=layer,
            epoch=epoch,
        )

    def prepare_model_qkv(
        self,
        qkv: torch.Tensor,
        batch: Any,
        *,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_dim = self.table.model_local_q_heads * self.head_dim
        kv_dim = self.table.model_local_kv_heads * self.head_dim
        q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
        attention = self.model.model.layers.op_list[layer].self_attn.attn
        q, k = attention.apply_qk_norm_rope(batch.positions, q, k)
        return q, k, v

    def publish_model_qkv_payload(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: Any,
        *,
        layer: int,
        epoch: int,
    ) -> None:
        slot = epoch % self.transport_slots
        source_offsets_device = batch.afd_fmha_source_token_offsets_device
        if source_offsets_device is None:
            raise RuntimeError(
                "FMHA QKV publication requires capture-owned or eager source offsets"
            )
        publish_qkv(
            q,
            k,
            v,
            batch.out_loc,
            source_offsets_device,
            self.q_descriptors,
            self.kv_descriptors,
            self.q_ready_descriptors,
            self.q_publish_counters[slot : slot + 1],
            layer=layer,
            slot=slot,
            head_dim=self.head_dim,
            ready_value=epoch + 1,
        )

    def _wait_attention_q(
        self,
        batch: Any,
        *,
        epoch: int,
        turn: torch.Tensor | None = None,
        expected_turn: int | None = None,
    ) -> torch.Tensor:
        rows = int(batch.positions.numel())
        if rows > self.max_rows:
            raise RuntimeError(f"AFD Q rows {rows} exceed slot capacity {self.max_rows}")
        slot = epoch % self.transport_slots
        q_slot = self.q_slots[slot, :rows]
        wait_ready(
            self.q_ready[slot],
            self.timeout_record,
            timeout_ms=self.timeout_ms,
            head_dim=self.head_dim,
            expected_ready=epoch + 1,
            turn=turn,
            expected_turn=expected_turn,
        )
        return q_slot

    def run_attention_fmha(
        self,
        batch: Any,
        *,
        layer: int,
        epoch: int,
        lane_stream: torch.cuda.Stream | None = None,
        q_wait_already_enqueued: bool = False,
        enqueue_next_q_wait: bool = False,
        attention_turn: torch.Tensor | None = None,
        expected_turn: int | None = None,
        next_turn: int | None = None,
    ) -> None:
        ticketed = any(
            value is not None
            for value in (attention_turn, expected_turn, next_turn)
        )
        if (q_wait_already_enqueued or enqueue_next_q_wait or ticketed) and (
            lane_stream is None
            or attention_turn is None
            or expected_turn is None
            or next_turn is None
        ):
            raise RuntimeError(
                "attention Q prewaits require one owning lane stream and a "
                "complete device-turn handoff"
            )
        rows = int(batch.positions.numel())
        if rows > self.max_rows:
            raise RuntimeError(f"AFD Q rows {rows} exceed slot capacity {self.max_rows}")
        destination_source_stride = (
            rows if batch.phase == "decode" else self.source_max_rows
        )
        slot = epoch % self.transport_slots
        q_slot = self.q_slots[slot, :rows]
        if not q_wait_already_enqueued:
            q_slot = self._wait_attention_q(
                batch,
                epoch=epoch,
                turn=attention_turn,
                expected_turn=expected_turn,
            )
        o = self.state.ctx.attn_backend.forward_prepared(q_slot, layer, batch)
        o_fp8, o_scale = self._quantize_attention_o(o.view(rows, -1))
        if attention_turn is None:
            publish_o_fp8(
                o_fp8,
                o_scale,
                self.o_descriptors,
                self.o_ready_descriptors,
                self.o_publish_counters[slot : slot + 1],
                slot=slot,
                destination_source_stride=destination_source_stride,
                head_dim=self.head_dim,
                ready_value=epoch + 1,
            )
        else:
            assert next_turn is not None
            # Quantization remains part of the attention compute unit. Once it
            # completes, the fused FP8 O publication releases the next lane while
            # payload and packed-scale stores proceed.
            publish_o_fp8_release_turn(
                o_fp8,
                o_scale,
                self.o_descriptors,
                self.o_ready_descriptors,
                self.o_publish_counters[slot : slot + 1],
                attention_turn,
                slot=slot,
                destination_source_stride=destination_source_stride,
                next_turn=next_turn,
                head_dim=self.head_dim,
                ready_value=epoch + 1,
            )
        if enqueue_next_q_wait:
            assert expected_turn is not None
            self._wait_attention_q(
                batch,
                epoch=epoch + self.num_mb,
                turn=attention_turn,
                expected_turn=expected_turn + self.num_mb,
            )

    def receive_attention_o(
        self, batch: Any, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_offsets = tuple(
            int(value) for value in batch.afd_fmha_source_token_offsets
        )
        if (
            len(source_offsets) != self.fanin + 1
            or source_offsets[0] != 0
            or any(
                right <= left
                for left, right in zip(source_offsets, source_offsets[1:])
            )
        ):
            raise RuntimeError(
                "FMHA-only model O receive has invalid source offsets: "
                f"offsets={source_offsets} fanin={self.fanin}"
            )
        rows = source_offsets[-1]
        if rows > self.max_rows:
            raise RuntimeError(f"AFD O rows {rows} exceed slot capacity {self.max_rows}")
        slot = epoch % self.transport_slots
        wait_ready(
            self.o_ready[slot].reshape(-1),
            self.timeout_record,
            timeout_ms=self.timeout_ms,
            head_dim=self.head_dim,
            expected_ready=epoch + 1,
        )
        source_rows = tuple(
            right - left for left, right in zip(source_offsets, source_offsets[1:])
        )
        scale_stride = div_ceil(rows, 4) * 4
        if batch.phase == "decode":
            if len(set(source_rows)) != 1:
                raise RuntimeError(
                    "FMHA-only decode requires equal graph rows from every source: "
                    f"rows={source_rows}"
                )
            o_fp8 = self.o_slots[slot, :rows].view(rows, -1)
            o_scale = torch.as_strided(
                self.o_scale_slots[slot],
                size=(rows, self.o_scale_packed_groups),
                stride=(1, scale_stride),
            )
            return o_fp8, o_scale
        if any(count > self.source_max_rows for count in source_rows):
            raise RuntimeError(
                "FMHA-only prefill source exceeds its O partition: "
                f"rows={source_rows} capacity={self.source_max_rows}"
            )
        o_fp8 = torch.empty(
            (
                rows,
                self.table.model_local_q_heads,
                self.head_dim,
            ),
            dtype=self.o_slots.dtype,
            device=self.device,
        )
        o_scale = torch.empty_strided(
            (rows, self.o_scale_packed_groups),
            (1, scale_stride),
            dtype=torch.int32,
            device=self.device,
        )
        scale_slot = self.o_scale_slots[slot]
        scale_slot_offset = int(scale_slot.storage_offset())
        for source, (left, right) in enumerate(
            zip(source_offsets, source_offsets[1:])
        ):
            count = right - left
            source_start = source * self.source_max_rows
            o_fp8[left:right].copy_(
                self.o_slots[slot, source_start : source_start + count]
            )
            source_scale = torch.as_strided(
                scale_slot,
                size=(count, self.o_scale_packed_groups),
                stride=(1, self.o_scale_row_stride),
                storage_offset=scale_slot_offset + source_start,
            )
            o_scale[left:right].copy_(source_scale)
        return o_fp8.view(rows, -1), o_scale

    def run_step(self, plan: Any, *, sent_ns: int) -> AfdFmhaInFlight:
        if int(plan.num_mb) != self.num_mb:
            raise RuntimeError(
                "FMHA-only plan/runtime microbatch mismatch: "
                f"plan={int(plan.num_mb)} runtime={self.num_mb}"
            )
        self._plan_attn_dp_rank(plan)
        if self.role == "mlp":
            raise RuntimeError(
                "FMHA-only model workers require one grouped fan-in plan per step"
            )
        batch = self.worker.runtime.materialize_ag_plan(plan)
        self.state.stream.wait_stream(self.worker.runtime.stream)
        if plan.phase == "decode":
            self._replay_decode_graph(batch, plan)
        else:
            self._run_attention_step(batch, plan)
        done = torch.cuda.Event()
        done.record(self.state.stream)
        return AfdFmhaInFlight(
            step_id=int(plan.step_id),
            sent_ns=int(sent_ns),
            done=done,
            # Materialization tensors are produced on runtime.stream and read
            # asynchronously on state.stream. Keep their owning batch alive
            # until the in-flight event is retired so the caching allocator
            # cannot reuse metadata or out_loc storage while kernels consume it.
            retained_batch=batch,
        )

    def run_fanin_step(
        self,
        plan: AfdModelStepPlan,
        *,
        sent_ns: int,
    ) -> AfdFmhaFanInFlight:
        if self.role != "mlp":
            raise RuntimeError("only FMHA-only model workers can run a fan-in step")
        if int(plan.num_mb) != self.num_mb:
            raise RuntimeError(
                "FMHA-only model fan-in microbatch mismatch: "
                f"plan={int(plan.num_mb)} runtime={self.num_mb}"
            )
        attn_dp_ranks = tuple(
            self._plan_attn_dp_rank(source_plan)
            for source_plan in plan.source_plans
        )
        if attn_dp_ranks != self._mapped_attn_dp_ranks:
            raise RuntimeError(
                "FMHA-only model fan-in does not cover its assigned sources: "
                f"actual={attn_dp_ranks} expected={self._mapped_attn_dp_ranks}"
            )
        if self.fanin == 1:
            self._activate_model_materializer(attn_dp_ranks[0])
            namespaced = self._namespace_model_plan(
                plan.source_plans[0], attn_dp_ranks[0]
            )
            batch = self.worker.runtime.materialize_ag_plan(namespaced)
            with torch.cuda.stream(self.worker.runtime.stream):
                for sub in batch.afd_mb_subbatches:
                    rows = int(sub.positions.numel())
                    sub.afd_fmha_source_token_offsets = (0, rows)
                    sub.afd_fmha_source_token_offsets_device = (
                        None
                        if plan.phase == "decode"
                        else torch.tensor((0, rows), dtype=torch.int64, device=self.device)
                    )
        else:
            namespaced_plans = []
            source_batches = []
            for source_plan, attn_dp_rank in zip(
                plan.source_plans,
                attn_dp_ranks,
                strict=True,
            ):
                self._activate_model_materializer(attn_dp_rank)
                namespaced = self._namespace_model_plan(source_plan, attn_dp_rank)
                namespaced_plans.append(namespaced)
                if plan.phase == "decode":
                    source_batches.append(
                        self.worker.runtime.materialize_ag_plan_state_only(namespaced)
                    )
                else:
                    source_batches.append(
                        self.worker.runtime.materialize_ag_plan(
                            namespaced,
                            attach_sampling=False,
                        )
                    )
            with torch.cuda.stream(self.worker.runtime.stream):
                if plan.phase == "decode":
                    batch = self._materialize_model_fanin_decode_batch(
                        plan,
                        tuple(namespaced_plans),
                        tuple(source_batches),
                    )
                else:
                    batch = self._merge_model_fanin_batches(
                        plan,
                        tuple(namespaced_plans),
                        tuple(source_batches),
                    )
        self.state.stream.wait_stream(self.worker.runtime.stream)
        if plan.phase == "decode":
            next_tokens = self._replay_decode_graph(batch, plan)
            assert next_tokens is not None
        else:
            next_tokens = self._run_model_step(batch, plan)
        d2h = self.worker._d2h_stream
        d2h.wait_stream(self.state.stream)
        next_tokens.record_stream(d2h)
        with torch.cuda.stream(d2h):
            next_tokens_cpu = next_tokens.to("cpu", non_blocking=True)
            done = torch.cuda.Event()
            done.record(d2h)
        if plan.phase == "decode":
            self._graph_output_free = done
        next_token_offsets = [0]
        for source_plan in plan.source_plans:
            next_token_offsets.append(
                next_token_offsets[-1] + int(source_plan.real_size)
            )
        return AfdFmhaFanInFlight(
            step_id=int(plan.step_id),
            sent_ns=int(sent_ns),
            done=done,
            retained_batch=batch,
            next_tokens_cpu=next_tokens_cpu,
            attn_dp_ranks=attn_dp_ranks,
            next_token_offsets=tuple(next_token_offsets),
        )

    @staticmethod
    def _cat_tensors(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)

    def _materialize_model_fanin_decode_batch(
        self,
        plan: AfdModelStepPlan,
        namespaced_plans: tuple[Any, ...],
        source_batches: tuple[Any, ...],
    ) -> Any:
        """Stage one merged decode input mapping for the complete model fan-in."""
        from minisgl.core import Batch

        if str(plan.phase) != "decode" or any(
            str(source_plan.phase) != "decode" for source_plan in namespaced_plans
        ):
            raise RuntimeError("FMHA-only grouped materialization is decode-only")
        if not (
            len(namespaced_plans)
            == len(source_batches)
            == len(self._mapped_attn_dp_ranks)
        ):
            raise RuntimeError("FMHA-only model fan-in materialization is incomplete")

        merged_subs = []
        request_offsets = [0]
        token_offsets = [0]
        source_prefixes_by_mb: list[tuple[int, ...]] = []
        for mb in range(self.num_mb):
            source_req_slices = []
            source_offsets = [0]
            for source_plan, source_batch in zip(
                namespaced_plans,
                source_batches,
                strict=True,
            ):
                req_offsets = tuple(int(value) for value in source_plan.microbatch_offsets)
                if len(req_offsets) != self.num_mb + 1:
                    raise RuntimeError(
                        "FMHA-only source plan has invalid microbatch offsets: "
                        f"offsets={req_offsets} num_mb={self.num_mb}"
                    )
                source_reqs = source_batch.padded_reqs[
                    req_offsets[mb] : req_offsets[mb + 1]
                ]
                source_req_slices.append(source_reqs)
                source_offsets.append(source_offsets[-1] + len(source_reqs))
            source_spans = tuple(
                right - left
                for left, right in zip(source_offsets, source_offsets[1:])
            )
            if len(set(source_spans)) != 1:
                raise RuntimeError(
                    "FMHA-only decode fan-in requires equal source graph spans: "
                    f"mb={mb} spans={source_spans}"
                )
            source_prefixes_by_mb.append(tuple(source_offsets))
            sub_reqs = [req for source_reqs in source_req_slices for req in source_reqs]
            sub = Batch(reqs=sub_reqs, phase="decode")
            sub.padded_reqs = sub_reqs
            sub.afd_fmha_source_token_offsets = tuple(source_offsets)
            sub.afd_fmha_source_token_offsets_device = None
            merged_subs.append(sub)
            request_offsets.append(request_offsets[-1] + len(sub_reqs))
            token_offsets.append(token_offsets[-1] + len(sub_reqs))

        batch = Batch(
            reqs=[req for source in source_batches for req in source.reqs],
            phase="decode",
        )
        batch.padded_reqs = [req for sub in merged_subs for req in sub.padded_reqs]
        batch.afd_microbatch_offsets = tuple(request_offsets)
        batch.afd_microbatch_token_offsets = tuple(token_offsets)
        stage_index, table_indices, pos_i64 = (
            self.worker.runtime._attach_input_mapping(batch, namespaced_plans[0])
        )
        token_pool = self.worker.runtime.token_pool
        if token_pool is None:
            raise RuntimeError("FMHA-only model fan-in has no token pool")
        batch.input_ids = token_pool[table_indices, pos_i64]
        self.worker.runtime._record_input_mapping_stage(stage_index)
        batch.afd_mb_subbatches = merged_subs

        for mb, sub in enumerate(merged_subs):
            start, end = token_offsets[mb], token_offsets[mb + 1]
            sub.input_ids = batch.input_ids[start:end]
            sub.positions = batch.positions[start:end]
            sub.out_loc = batch.out_loc[start:end]

        batch.afd_sampling_plan = plan.sampling
        self._attach_model_fanin_sampling(
            batch,
            namespaced_plans,
            tuple(token_offsets),
            source_prefixes_by_mb,
        )
        return batch

    def _merge_model_fanin_batches(
        self,
        plan: AfdModelStepPlan,
        namespaced_plans: tuple[Any, ...],
        source_batches: tuple[Any, ...],
    ) -> Any:
        from minisgl.core import Batch

        if not (
            len(namespaced_plans)
            == len(source_batches)
            == len(self._mapped_attn_dp_ranks)
        ):
            raise RuntimeError("FMHA-only model fan-in materialization is incomplete")
        merged_subs = []
        request_offsets = [0]
        token_offsets = [0]
        source_prefixes_by_mb: list[tuple[int, ...]] = []
        for mb in range(self.num_mb):
            source_subs = tuple(
                batch.afd_mb_subbatches[mb] for batch in source_batches
            )
            source_offsets = [0]
            for source_sub in source_subs:
                source_offsets.append(
                    source_offsets[-1] + int(source_sub.positions.numel())
                )
            source_prefixes_by_mb.append(tuple(source_offsets))
            sub_reqs = [req for source_sub in source_subs for req in source_sub.reqs]
            sub_padded_reqs = [
                req for source_sub in source_subs for req in source_sub.padded_reqs
            ]
            sub = Batch(reqs=sub_reqs, phase=plan.phase)
            sub.padded_reqs = sub_padded_reqs
            sub.input_ids = self._cat_tensors(
                tuple(source_sub.input_ids for source_sub in source_subs)
            )
            sub.positions = self._cat_tensors(
                tuple(source_sub.positions for source_sub in source_subs)
            )
            sub.out_loc = self._cat_tensors(
                tuple(source_sub.out_loc for source_sub in source_subs)
            )
            sub.afd_fmha_source_token_offsets = tuple(source_offsets)
            # Decode replay uses the immutable source-offset tensor owned by
            # the captured batch. Creating a fresh device tensor here forces a
            # pageable H2D stream synchronization once per microbatch, delaying
            # the model graph and every attention lane's first Q wait. Eager
            # prefill publishes directly from this materialized batch and still
            # needs its own device offsets.
            sub.afd_fmha_source_token_offsets_device = (
                None
                if plan.phase == "decode"
                else torch.tensor(
                    source_offsets,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
            merged_subs.append(sub)
            request_offsets.append(request_offsets[-1] + len(sub_padded_reqs))
            token_offsets.append(token_offsets[-1] + int(sub.positions.numel()))

        batch = Batch(
            reqs=[req for source in source_batches for req in source.reqs],
            phase=plan.phase,
        )
        batch.padded_reqs = [req for sub in merged_subs for req in sub.padded_reqs]
        batch.input_ids = self._cat_tensors(tuple(sub.input_ids for sub in merged_subs))
        batch.positions = self._cat_tensors(tuple(sub.positions for sub in merged_subs))
        batch.out_loc = self._cat_tensors(tuple(sub.out_loc for sub in merged_subs))
        batch.afd_microbatch_offsets = tuple(request_offsets)
        batch.afd_microbatch_token_offsets = tuple(token_offsets)
        batch.afd_mb_subbatches = merged_subs
        batch.afd_sampling_plan = plan.sampling

        self._attach_model_fanin_sampling(
            batch,
            namespaced_plans,
            tuple(token_offsets),
            source_prefixes_by_mb,
        )
        return batch

    def _attach_model_fanin_sampling(
        self,
        batch: Any,
        namespaced_plans: tuple[Any, ...],
        token_offsets: tuple[int, ...],
        source_prefixes_by_mb: list[tuple[int, ...]],
    ) -> None:
        if len(token_offsets) != self.num_mb + 1:
            raise RuntimeError(
                "FMHA-only model sampling has invalid token offsets: "
                f"offsets={token_offsets} num_mb={self.num_mb}"
            )
        if len(source_prefixes_by_mb) != self.num_mb:
            raise RuntimeError(
                "FMHA-only model sampling has invalid source prefixes: "
                f"count={len(source_prefixes_by_mb)} num_mb={self.num_mb}"
            )

        merged_last_positions: list[int] = []
        wb_tables: list[int] = []
        wb_positions: list[int] = []
        wb_batches: list[int] = []
        sample_base = 0
        for source_index, source_plan in enumerate(namespaced_plans):
            last_positions, tables, positions, sample_indices = (
                self.worker.runtime._collect_ag_sample_meta(source_plan)
            )
            source_token_offsets = tuple(
                int(value) for value in source_plan.microbatch_token_offsets
            )
            for local_position in last_positions:
                for mb in range(self.num_mb):
                    source_start = source_token_offsets[mb]
                    source_end = source_token_offsets[mb + 1]
                    if source_start <= int(local_position) < source_end:
                        merged_last_positions.append(
                            token_offsets[mb]
                            + source_prefixes_by_mb[mb][source_index]
                            + int(local_position)
                            - source_start
                        )
                        break
                else:
                    raise RuntimeError(
                        "FMHA-only sampling position is outside its source batch: "
                        f"source={source_index} position={local_position} "
                        f"offsets={source_token_offsets}"
                    )
            wb_tables.extend(tables)
            wb_positions.extend(positions)
            wb_batches.extend(sample_base + int(index) for index in sample_indices)
            sample_base += len(last_positions)
        batch.afd_last_indices, batch.afd_writeback = (
            self.worker.runtime._stage_ag_sample_meta(
                merged_last_positions,
                wb_tables,
                wb_positions,
                wb_batches,
            )
        )
        batch.afd_num_sampling = len(merged_last_positions)

    def _run_attention_step(self, batch: Any, plan: Any) -> None:
        with torch.cuda.stream(self.state.stream):
            self._run_attention_body(
                batch,
                int(plan.step_id) * int(self.model_config.num_layers) * self.num_mb,
            )

    def _run_model_step(self, batch: Any, plan: Any) -> torch.Tensor:
        ctx = self.state.ctx
        with torch.cuda.stream(self.state.stream):
            final = self._run_model_forward(
                batch,
                slot_base=(
                    int(plan.step_id)
                    * int(self.model_config.num_layers)
                    * self.num_mb
                ),
                real_token_counts=tuple(
                    int(x) for x in plan.microbatch_real_token_counts
                ),
                dispatch_bucket=int(plan.dispatch_bucket),
            )
            if int(batch.afd_num_sampling) <= 0:
                return torch.empty((0,), dtype=torch.int32, device=self.device)
            selected = final.index_select(0, batch.afd_last_indices)
            sampling_batch = self.worker.runtime.build_sampling_batch(
                int(batch.afd_num_sampling)
            )
        with ctx.forward_batch(sampling_batch), torch.cuda.stream(self.state.stream):
            logits = self.model.forward_lm_head(selected)
            sampling_args = plan.sampling.to_sampling_args(
                self.device, int(self.model_config.vocab_size)
            )
            next_tokens = self.sampler.sample(logits, sampling_args).to(torch.int32)
        self.worker.runtime.stream.wait_stream(self.state.stream)
        self.worker.runtime.writeback_tokens_ag(batch, next_tokens)
        return next_tokens


__all__ = [
    "AfdFmhaFanInFlight",
    "AfdFmhaInFlight",
    "AfdFmhaRuntime",
]
