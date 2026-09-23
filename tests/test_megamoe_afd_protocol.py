"""CPU checks for the AFD allocation/launch protocol; no GPU kernels are mocked as tests."""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from minisgl.afd_support import runtime_env
from minisgl.kernel import megamoe_mega as mega
from minisgl.moe import megamoe_afd as afd
from minisgl.server.supervisor_ray import _build_ray_runtime_env


class MegaMoEAfdProtocolTest(unittest.TestCase):
    def test_optimization_settings_reach_ray_workers(self):
        settings = {
            "MINISGL_AFD_QKV_EDGE": "1",
            "MINISGL_AFD_QKV_LAMPORT": "1",
            "MINISGL_MEGAMOE_ALL_PUSH_CAPACITY": "1604",
            "MINISGL_MEGAMOE_TRACE_ROOT": "/model-trace",
            "MINISGL_MEGAMOE_RECYCLE_EARLY_PDL": "1",
            "MINISGL_MEGAMOE_RECYCLE_CTAS": "1",
            "MINISGL_MEGAMOE_RECYCLE_AHEAD": "1",
            "MINISGL_MEGAMOE_RECYCLE_TMA": "1",
            "MINISGL_MEGAMOE_NORM_PRELOAD_WEIGHT": "1",
        }
        with patch.dict(os.environ, settings):
            for env in (_build_ray_runtime_env(), runtime_env()):
                for key, value in settings.items():
                    self.assertEqual(env["env_vars"][key], value)

    def test_payload_selection_reaches_ray_workers(self):
        for mode in ("pull", "all_push"):
            with patch.dict(os.environ, {"MINISGL_MEGAMOE_PAYLOAD_MODE": mode}):
                coordinator_env = _build_ray_runtime_env()
                self.assertEqual(coordinator_env["env_vars"]["MINISGL_MEGAMOE_PAYLOAD_MODE"], mode)
                env = runtime_env(node_scoped_cache=True, node_ip="192.0.2.1",
                                  cache_rank_tag="rank1")
                self.assertEqual(env["env_vars"]["MINISGL_MEGAMOE_PAYLOAD_MODE"], mode)

    def test_checkpoint_constructor_and_explicit_buffer_selection(self):
        # Autospec enforces the real checkpoint loader's keyword signature.
        full, arange = torch.full, torch.arange
        def on_cpu(function):
            def call(*args, **kwargs):
                kwargs["device"] = "cpu"
                return function(*args, **kwargs)
            return call

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE": "fp4",
                "MINISGL_MEGAMOE_EXPERT_WEIGHT_PATH": "checkpoint",
                "DG_MEGAMOE_LAMPORT_COMBINE": "0",
            }))
            stack.enter_context(patch.object(afd.dist, "is_initialized", return_value=True))
            stack.enter_context(patch.object(afd.dist, "get_rank", return_value=0))
            stack.enter_context(patch.object(afd.dist, "get_world_size", return_value=2))
            stack.enter_context(patch.object(torch.cuda, "current_device", return_value=0))
            stack.enter_context(patch.object(torch, "full", side_effect=on_cpu(full)))
            stack.enter_context(patch.object(torch, "arange", side_effect=on_cpu(arange)))
            loader = stack.enter_context(patch.object(
                afd._weights, "Mxfp4ExpertWeightLoader", autospec=True))
            allocator = stack.enter_context(patch.object(mega, "MegaMoESymmBuffer", autospec=True))
            allocator.return_value.payload_mode = "pull"
            allocator.return_value.all_push = False
            allocator.return_value.num_max_tokens_per_rank = 8
            adapter = afd.MegaMoEAfdAdapter(
                group=object(), num_experts=8, num_local_experts=4,
                hidden_size=1024, intermediate_size=512, num_layers=1,
                top_k=2, num_max_tokens_per_rank=8, num_lanes=2,
                tp_size=1, tp_rank=0, gate_renormalize=True,
            )
            adapter.allocate_buffers()
            adapter.allocate_buffers()
            loader.assert_called_once()
            self.assertEqual(allocator.call_count, 2)
            self.assertTrue(all(c.kwargs["inband_combine"] is True
                                for c in allocator.call_args_list))
            # Decode storage stays fixed for both small and 8K prompt budgets;
            # only the explicit pull-prefill allocation follows the input limit.
            stack.enter_context(patch.object(torch.cuda, "Stream"))
            stack.enter_context(patch.object(torch.cuda, "Event"))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            stack.enter_context(patch.dict(os.environ, {
                "MINISGL_MEGAMOE_PAYLOAD_MODE": "all_push",
            }))
            allocator.return_value.payload_mode = "all_push"
            allocator.return_value.all_push = True
            allocator.return_value.num_max_tokens_per_rank = 1024
            allocator.return_value.handle = SimpleNamespace(barrier=Mock())
            allocator.return_value.y = torch.empty(8, 1024)
            for prefill_capacity in (8, 8192):
                allocator.reset_mock()
                adapter.buffers = []
                adapter.bucket = prefill_capacity
                adapter.allocate_buffers()
                self.assertEqual(
                    [(c.kwargs["payload_mode"], c.kwargs["num_max_tokens_per_rank"])
                     for c in allocator.call_args_list],
                    [("all_push", 1024)] * 4 + [("pull", prefill_capacity)] * 2,
                )

    def test_padded_empty_rank_participates_with_prepared_metadata(self):
        adapter = object.__new__(afd.MegaMoEAfdAdapter)
        adapter.hidden_size, adapter.bucket, adapter.num_lanes = 1024, 8, 1
        adapter.all_push_reset_rows = 1024
        adapter.prefill_buffers = []
        adapter.top_k, adapter.expert_map = 2, None
        adapter.gate_renormalize = True
        adapter.perfect_routing = adapter.defer_route_dependent_launch = False
        adapter.recycle_ahead = False
        adapter.weights = {0: (object(), object())}
        adapter._run_mega_moe = Mock()
        buffer = SimpleNamespace(
            all_push=False,
            topk_idx=torch.empty(8, 2, dtype=torch.int64),
            topk_weights=torch.empty(8, 2), x=torch.empty(8, 1024),
            x_sf=torch.empty(8, 32), y=torch.empty(8, 1024),
            route_prepare_args=Mock(return_value=("rank-ready",)),
        )
        adapter.buffers = [buffer]
        with patch.object(afd, "gate_topk") as gate:
            output = adapter.forward(0, torch.zeros(1, 1024), torch.zeros(8, 1024),
                                     lane=0, num_token_non_padded=0)
        self.assertEqual(output.shape, (1, 1024))
        self.assertEqual(gate.call_args.kwargs["num_token_non_padded"], 0)
        self.assertEqual(gate.call_args.kwargs["route_prepare"], ("rank-ready",))
        adapter._run_mega_moe.assert_called_once()
        self.assertTrue(all(adapter._run_mega_moe.call_args.kwargs.values()))

        # Alternating layers use distinct storage. Reset and readiness belong
        # to the native specialization, with no per-call events or peer barrier.
        buffer.all_push = True
        alternate = SimpleNamespace(**vars(buffer))
        calls = Mock()
        adapter.alternate_buffers = [alternate]
        adapter.weights[1] = adapter.weights[0]
        adapter._run_mega_moe = calls.moe
        with patch.object(afd, "gate_topk", calls.gate):
            for layer in (0, 1, 0):
                adapter.forward(layer, torch.zeros(1, 1024), torch.zeros(8, 1024),
                                lane=0, num_token_non_padded=0)
        self.assertEqual([c[0] for c in calls.mock_calls], ["gate", "moe"] * 3)
        self.assertEqual([c.args[3] for c in calls.moe.call_args_list],
                         [buffer, alternate, buffer])
        self.assertTrue(all(c.kwargs["recycle_payload"] for c in calls.gate.call_args_list))
        self.assertTrue(all(c.kwargs["reset_rows"] == 1024 for c in calls.moe.call_args_list))

        calls.reset_mock()
        with patch.object(afd, "gate_topk", calls.gate), self.assertRaisesRegex(
            RuntimeError, "exceeds row capacity=1024"
        ):
            adapter.forward(0, torch.zeros(1025, 1024), torch.zeros(8, 1024),
                            lane=0, num_token_non_padded=1025)
        self.assertFalse(calls.mock_calls)

        # Prefill is an explicit phase policy, independent of the decode shape.
        prefill = SimpleNamespace(**{**vars(buffer), "all_push": False})
        adapter.prefill_buffers = [prefill]
        calls.reset_mock()
        with (patch.object(afd, "gate_topk", calls.gate),
              patch.object(torch.cuda, "current_stream", side_effect=AssertionError("decode prep in prefill"))):
            adapter.forward(0, torch.zeros(1, 1024), torch.zeros(8, 1024),
                            lane=0, num_token_non_padded=0, phase="prefill")
        self.assertEqual([c[0] for c in calls.mock_calls], ["gate", "moe"])
        self.assertIs(calls.moe.call_args.args[3], prefill)

    def test_both_weight_wrappers_use_allocation_protocol(self):
        buffer = SimpleNamespace(buffer=object(), buffer_ptrs=[1, 2], rank=0,
                                 num_max_tokens_per_rank=8, num_experts=8, num_topk=2, all_push=False)
        extension = Mock()
        with patch.object(mega, "_ext", return_value=extension):
            for mode, compact in ((False, False), (True, False), (False, True), (True, True)):
                buffer.all_push = compact
                buffer.inband_combine = mode
                with patch.dict(os.environ, {"DG_MEGAMOE_LAMPORT_COMBINE": str(int(not mode))}):
                    for name in ("fp8_fp4_mega_moe", "fp8_fp8_mega_moe"):
                        getattr(mega, name)(object(), (1, 2), (3, 4), buffer,
                                           routes_prepared=True, rank_gated_combine=True,
                                           route_ready_dispatch=True, rank_ready_route_publish=True)
                        self.assertEqual(getattr(extension, name).call_args.args[-2:], (mode, buffer.all_push))


if __name__ == "__main__":
    unittest.main()
