from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class GB200OverlapContractTests(unittest.TestCase):
    def test_bundle_pool_accepts_precision_qualified_case_ids(self) -> None:
        path = (
            ROOT
            / "scripts/experiments/afd/oci_hsg/afd_online_case_pool.py"
        )
        spec = importlib.util.spec_from_file_location("afd_online_case_pool", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for precision in ("fp8", "fp4"):
            case_id = f"i131072-fep8-r7-atp1-b6-{precision}"
            self.assertEqual(module.validate_case_id(case_id), case_id)

    def test_gpu_cleanup_uses_pidfd_events_without_sleep(self) -> None:
        path = (
            ROOT
            / "scripts/experiments/afd/oci_hsg/wait_gpu_processes_exit.py"
        )
        source = path.read_text()
        self.assertIn("os.pidfd_open(pid)", source)
        self.assertIn("poller.poll(timeout_ms)", source)
        self.assertNotIn("time.sleep", source)
        spec = importlib.util.spec_from_file_location("wait_gpu_processes_exit", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(
            module, "query_gpu_pids", side_effect=[{101, 202}, set()]
        ), mock.patch.object(
            module, "wait_for_pid_events", return_value={101}
        ) as wait_events:
            report = module.wait_for_gpu_processes_exit("nvidia-smi", 1.0)
        self.assertEqual(report["status"], "clear")
        self.assertEqual(report["observed_pids"], [101, 202])
        self.assertEqual(report["wait_mechanism"], "linux_pidfd_poll")
        wait_events.assert_called_once()

    def test_bundle_case_cleanup_blocks_before_next_gpu_preflight(self) -> None:
        source = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        cleanup = source.index('ray_cli stop --force >/dev/null 2>&1 || true')
        barrier = source.index('"$PYTHON" "$GPU_PROCESS_EXIT_SCRIPT"')
        snapshot = source.index(
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory"
        )
        self.assertLess(cleanup, barrier)
        self.assertLess(barrier, snapshot)

    def test_bundle_cleanup_uses_stable_case_directory(self) -> None:
        source = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn("CASE_RUN_DIR=$RUN_DIR", source)
        self.assertIn(
            '> "$CASE_RUN_DIR/gpu-snapshots/cleanup-rank-$RANK.json"', source
        )
        self.assertNotIn(
            '> "$RUN_DIR/gpu-snapshots/cleanup-rank-$RANK.json"', source
        )
        self.assertLess(
            source.index("CASE_RUN_DIR=$RUN_DIR"),
            source.index("export RUN_DIR=$EXPERIMENT"),
        )

    def test_bundle_resolves_controls_from_submitted_source_repo(self) -> None:
        source = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd_bundle.sh"
        ).read_text()
        self.assertIn(
            "CONTROL_DIR=$SOURCE_REPO/scripts/experiments/afd/oci_hsg",
            source,
        )
        self.assertIn(
            "Slurm executes a copied spool script",
            source,
        )

    def test_bundle_accepts_slurm_subhour_remaining_time(self) -> None:
        source = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd_bundle.sh"
        ).read_text()
        self.assertIn("2) minutes=${parts[0]}; seconds=${parts[1]}", source)
        self.assertIn(
            'SLURM_REMAINING_SECONDS=$(slurm_duration_seconds "$SLURM_REMAINING")',
            source,
        )
        self.assertNotIn(
            '[[ "$SLURM_REMAINING" =~ ^([0-9]+):',
            source,
        )

    def test_tray_chains_enforce_four_short_jobs_without_sleep(self) -> None:
        source = (
            ROOT
            / "scripts/experiments/afd/oci_hsg/submit_afd_tray_chain.sh"
        ).read_text()
        self.assertIn("#SBATCH --qos=cpu-short", source)
        self.assertIn("--qos=short --time=02:00:00", source)
        self.assertIn('--segment="$trays"', source)
        self.assertIn("[[ \"$MAX_SHORT_JOBS\" == 4 ]]", source)
        self.assertIn("short_jobs >= MAX_SHORT_JOBS", source)
        self.assertIn('--dependency="afterany:$gpu_job_id"', source)
        self.assertIn('validate_completed_group "$previous_trays"', source)
        self.assertIn("finalize_if_ready", source)
        self.assertIn('"$TASK_ROOT/report/terminal-audit.json"', source)
        self.assertIn("FASTAFD_CHAIN_TRAYS_ENCODED", source)
        self.assertIn("encoded_remaining=${encoded_remaining//,/:}", source)
        self.assertIn(
            "MAX_BATCHED_TOKENS=${FASTAFD_AFD_MAX_BATCHED_TOKENS:?}", source
        )
        self.assertEqual(
            source.count("FASTAFD_AFD_MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS"),
            3,
        )
        self.assertIn(
            "CASE_TIMEOUT_SECONDS=${FASTAFD_CASE_TIMEOUT_SECONDS:?}", source
        )
        self.assertEqual(
            source.count("FASTAFD_CASE_TIMEOUT_SECONDS=$CASE_TIMEOUT_SECONDS"),
            3,
        )
        self.assertIn(
            "SOURCE_MANIFEST=${FASTAFD_EXPECTED_SOURCE_MANIFEST:?}", source
        )
        self.assertEqual(
            source.count("FASTAFD_EXPECTED_SOURCE_MANIFEST=$SOURCE_MANIFEST"),
            3,
        )
        self.assertIn("IMAGE=${FASTAFD_IMAGE:?}", source)
        self.assertEqual(source.count("FASTAFD_IMAGE=$IMAGE"), 3)
        self.assertIn(
            'len({int(row["allocated_trays"]) for row in csv.DictReader(stream)})',
            source,
        )
        self.assertNotIn("== 22", source)
        self.assertNotIn("FASTAFD_CHAIN_TRAYS=${remaining:-$trays}", source)
        self.assertNotIn("sleep ", source)

    def test_fmha_bundle_requires_one_nvl72_fabric_block(self) -> None:
        source = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd_bundle.sh"
        ).read_text()
        self.assertIn("mapfile -t ALLOCATED_HOSTS", source)
        self.assertIn("sed -E 's/-T[0-9]+$//' | sort -u", source)
        self.assertIn("FMHA-only allocation must remain in one NVL72 fabric block", source)
        self.assertIn(
            "ALLOW_DIRTY_SOURCE=${FASTAFD_ALLOW_DIRTY_SOURCE:-0}", source
        )
        self.assertEqual(
            source.count('FASTAFD_ALLOW_DIRTY_SOURCE="$ALLOW_DIRTY_SOURCE"'),
            2,
        )

        runner = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn('if [[ "$ALLOW_DIRTY_SOURCE" == 1 ]]; then', runner)
        self.assertIn('SOURCE_VALIDATION_DETAIL="dirty_source_with_manifest"', runner)

    def test_reproduction_window_retains_warmup_and_all_measured_steps(self) -> None:
        source = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd_reproduce.sh"
        ).read_text()
        self.assertIn("TRACE_WARMUP_DECODE_STEPS=1", source)
        self.assertIn(
            "OUTPUT_TOKENS=$((1 + TRACE_WARMUP_DECODE_STEPS + "
            "NSYS_CAPTURE_DECODE_STEPS))",
            source,
        )
        self.assertIn("MAX_TOKENS=$OUTPUT_TOKENS", source)
        self.assertIn("set(lengths) != {output_tokens}", source)
        self.assertIn(
            'r"target_batch_per_dp=(\\d+) warmup_step_id=(\\d+) "', source
        )
        self.assertIn('r"step_ids=([0-9,]+) count=(\\d+) trace_count=(\\d+)"', source)
        self.assertIn("int(trace_count) != 16", source)
        self.assertIn("int(captured_warmup) != steps[0] - 1", source)
        self.assertIn('capture = json.loads((run / "capture-complete.json").read_text())', source)
        self.assertIn('capture.get("trace_decode_step_ids") != expected_trace_steps', source)
        self.assertIn('capture.get("gpu_worker_profiler_stop_logs", -1)', source)
        self.assertIn('worker_log.count("nsys profiler:start sync=1") != 1', source)
        self.assertIn('worker_log.count("nsys profiler:stop sync=1") != 1', source)
        self.assertNotIn("afd_ag_decode_graph:replay step_id=", source)
        self.assertNotIn("MAX_TOKENS=16", source)

    def test_fmha_campaign_uses_launch_safe_prefill_and_shared_builds(self) -> None:
        runner = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn(
            "AFD_MAX_BATCHED_TOKENS=${FASTAFD_AFD_MAX_BATCHED_TOKENS:-512}",
            runner,
        )
        self.assertIn("AFD_MAX_BATCHED_TOKENS <= 8192", runner)
        self.assertIn(
            "RAY_NUM_CPUS_PER_NODE=${FASTAFD_RAY_NUM_CPUS_PER_NODE:-16}",
            runner,
        )
        self.assertNotIn("--object-store-memory", runner)
        self.assertNotIn("--num-cpus=140", runner)

        support = (ROOT / "python/minisgl/afd_support.py").read_text()
        self.assertIn("_LOCKED_BUILD_CACHE_ENV_KEYS", support)
        self.assertIn(
            "if rank_tag and key not in _LOCKED_BUILD_CACHE_ENV_KEYS",
            support,
        )

        bundle = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd_bundle.sh"
        ).read_text()
        self.assertIn("#SBATCH --mem=0", bundle)
        self.assertIn("--gres=gpu:4 --mem=0 --kill-on-bad-exit=1", bundle)
        self.assertIn(
            "CASE_TIMEOUT_SECONDS=${FASTAFD_CASE_TIMEOUT_SECONDS:-1800}",
            bundle,
        )
        self.assertIn("CASE_TIMEOUT_SECONDS <= 3600", bundle)
        self.assertIn(
            'timeout --foreground --signal=TERM --kill-after=60s "$CASE_TIMEOUT_SECONDS"',
            bundle,
        )
        self.assertIn("historical-runtime watchdog", bundle)
        self.assertIn("watchdog_timeout=1", bundle)
        self.assertIn("watchdog_timeout || failures >= MAX_FAILURES", bundle)
        self.assertIn("per-case AFD memory contract is incomplete", bundle)
        self.assertIn(
            "per-case MegaMoE expert weight dtype must be fp8 or fp4", bundle
        )
        self.assertIn(
            "case_id must end with its MegaMoE expert weight dtype", bundle
        )
        self.assertIn('FASTAFD_CASE_ID="$case_id"', bundle)
        self.assertIn(
            'FASTAFD_AFD_KV_CAPACITY_TOKENS="$case_kv_capacity"', bundle
        )
        self.assertIn(
            'MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE="$case_expert_weight_dtype"',
            bundle,
        )

    def test_trtllm_decode_matches_original_source_policy(self) -> None:
        path = ROOT / "python/minisgl/attention/trtllm.py"
        source = path.read_text()
        tree = ast.parse(source)
        self.assertIn(
            "attention_sm_count = 120 if q.shape[0] == 8 else 128",
            source,
        )
        decode_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "trtllm_batch_decode_with_kv_cache"
        )
        self.assertNotIn("backend", {keyword.arg for keyword in decode_call.keywords})

    def test_deepep_comm_supports_adaptive_complete_clusters(self) -> None:
        python_source = (
            ROOT / "python/minisgl/kernel/deepep_moe.py"
        ).read_text()
        self.assertIn("OVERLAP_NUM_SMS = 24", python_source)
        self.assertIn("HIGH_FANIN_OVERLAP_NUM_SMS = 8", python_source)
        self.assertIn("HIGH_FANIN_MIN_SOURCES = 8", python_source)
        self.assertIn("SUPPORTED_OVERLAP_NUM_SMS = (8, 16, 24)", python_source)
        self.assertIn("overlap_num_sms: int = OVERLAP_NUM_SMS", python_source)
        self.assertIn("OVERLAP_CLUSTER_DIM = 8", python_source)
        for relative in (
            "python/minisgl/kernel/csrc/deepep/csrc/kernels/elastic/dispatch.hpp",
            "python/minisgl/kernel/csrc/deepep/csrc/kernels/elastic/combine.hpp",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn("LaunchArgs(num_sms, num_threads, num_smem_bytes, 8, true)", source)
            self.assertIn("num_sms == 8 or num_sms == 16 or num_sms == 24", source)
            self.assertNotIn("launch_barrier(nccl_dev_comm, nccl_window", source)
            self.assertNotIn("kDecodePhaseBarrierMaxTokensPerRank", source)
        buffer_source = (
            ROOT / "python/minisgl/kernel/csrc/deepep/csrc/elastic/buffer.hpp"
        ).read_text()
        self.assertIn(
            "math::ceil_div(num_max_tokens_per_rank, num_sms)", buffer_source
        )
        for relative in (
            "python/minisgl/kernel/csrc/deepep/csrc/kernels/elastic/dispatch.hpp",
            "python/minisgl/kernel/csrc/deepep/csrc/kernels/elastic/combine.hpp",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn(
                "math::ceil_div(num_max_tokens_per_rank, num_sms)", source
            )

        dispatch_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepep/csrc/kernels/elastic/dispatch.hpp"
        ).read_text()
        self.assertIn(
            "max_recv_tokens = num_max_tokens_per_rank * num_scaleout_ranks * num_scaleup_ranks",
            dispatch_source,
        )
        self.assertIn(
            "jit::LaunchArgs(epilogue_num_sms, num_threads",
            dispatch_source,
        )
        self.assertIn("BufferLayout<true>(token_layout, num_warps, 1)", dispatch_source)
        combine_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepep/csrc/kernels/elastic/combine.hpp"
        ).read_text()
        self.assertIn(
            "std::max(1, num_max_tokens_per_rank)",
            combine_source,
        )
        self.assertIn(
            "jit::LaunchArgs(epilogue_num_sms, num_threads",
            combine_source,
        )
        self.assertIn("BufferLayout<false>(token_layout, num_warps, 1)", combine_source)

    def test_deepep_uses_nccl_multimem_lsa_barriers(self) -> None:
        backend_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepep/csrc/kernels/backend/nccl.cu"
        ).read_text()
        self.assertIn("reqs.lsaMultimem = true", backend_source)
        self.assertIn("reqs.lsaBarrierCount = 10", backend_source)

        comm_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepep/deep_ep/include/deep_ep/common/comm.cuh"
        ).read_text()
        for name, value in (
            ("kDeviceBarrierTag", 0),
            ("kKernelBarrierTag", 1),
            ("kDispatchTag0", 2),
            ("kDispatchTag1", 3),
            ("kCombineTag0", 4),
            ("kCombineTag1", 5),
            ("kHybridDispatchTag0", 6),
            ("kHybridDispatchTag1", 7),
            ("kHybridCombineTag0", 8),
            ("kHybridCombineTag1", 9),
        ):
            self.assertIn(f"static constexpr int {name} = {value}", comm_source)
        self.assertIn("ncclLsaBarrierSession<ncclCoopCta> barrier", comm_source)
        self.assertIn("cuda::memory_order_acquire", comm_source)
        self.assertIn("cuda::memory_order_release", comm_source)
        self.assertNotIn("cuda::memory_order_acq_rel", comm_source)
        self.assertNotIn("ptx::red_add_rel_sys(dst_ptr", comm_source)

    def test_deepgemm_uses_sync_deallocation_as_peer_rendezvous(self) -> None:
        source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/sm100_blockscaled_gemm_1d1d.cuh"
        ).read_text()
        teardown = source[source.index("// Finish local TMEM use") :]
        self.assertIn("__syncthreads();", teardown)
        self.assertNotIn("cluster_sync_with_relaxed_arrive()", teardown)

        allocator_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepgemm/third-party/cutlass/include/cute/arch/tmem_allocator_sm100.hpp"
        ).read_text()
        self.assertIn(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32", allocator_source
        )

    def test_fmha_transport_uses_one_signal_after_one_payload_push(self) -> None:
        source = (
            ROOT / "python/minisgl/kernel/csrc/jit/afd_fmha_transport.cu"
        ).read_text()
        self.assertIn("constexpr int kSignalThreads = 32", source)
        self.assertIn("load_acquire_system", source)
        self.assertIn("store_release_system", source)
        self.assertIn("int64_t ready_value", source)
        self.assertIn("int64_t expected_ready", source)
        self.assertIn("static_cast<uint64_t>(expected_ready)", source)
        self.assertIn("reinterpret_cast<const int64_t*>(ready_ptr)", source)
        self.assertIn("finish_payload_publication", source)
        self.assertIn("atomicAdd(completion_counter, 1)", source)
        self.assertIn("atomicExch(completion_counter, 0)", source)
        finish_start = source.index("void finish_payload_publication(")
        finish_end = source.index("__global__ void wait_turn_kernel", finish_start)
        finish_source = source[finish_start:finish_end]
        self.assertIn("__threadfence_system();", finish_source)
        self.assertEqual(finish_source.count("__syncthreads();"), 1)
        self.assertIn("if constexpr (kSingleReady)", finish_source)
        self.assertIn("ready_pointer(ready_desc, slot)", finish_source)
        self.assertIn("__shfl_sync", finish_source)
        self.assertIn("edge = threadIdx.x", finish_source)
        self.assertIn("LaunchKernel(1, kSignalThreads", source)
        self.assertIn("wait_ready_turn", source)
        self.assertIn("publish_o_kernel_release_turn", source)
        self.assertIn("publish_o_fp8_kernel_release_turn", source)
        self.assertIn("const int32_t* scales", source)
        self.assertIn("wait_turn_kernel", source)
        self.assertNotIn("publish_turn_kernel", source)
        self.assertIn("LaunchKernel(1, kThreads", source)
        self.assertNotIn("cg::this_grid().sync()", source)
        self.assertNotIn("kSentinel", source)
        self.assertNotIn("canonicalize", source)
        self.assertNotIn("reset_sentinel", source)
        self.assertNotIn("consumed_", source)
        self.assertNotIn("atomic_ref<", source)
        self.assertNotIn("printf(", source)
        wrapper_source = (
            ROOT / "python/minisgl/kernel/afd_fmha_transport.py"
        ).read_text()
        self.assertIn("wait_ready_turn", wrapper_source)
        self.assertIn("publish_o_release_turn", wrapper_source)
        self.assertIn("publish_o_fp8_release_turn", wrapper_source)
        self.assertIn("def _validate_packed_scales(", wrapper_source)
        self.assertIn("def wait_turn(", wrapper_source)
        self.assertNotIn("def publish_turn(", wrapper_source)
        self.assertIn("def _validate_ready(", wrapper_source)
        self.assertIn("def _validate_completion_counter(", wrapper_source)
        self.assertNotIn("reset_sentinel", wrapper_source)
        self.assertIn("def _validate_ready_value(", wrapper_source)
        self.assertNotIn("consumed_", wrapper_source)
        runtime_source = (ROOT / "python/minisgl/afd_fmha_runtime.py").read_text()
        self.assertIn("self._attention_turn", runtime_source)
        self.assertIn("publish_o_fp8_release_turn(", runtime_source)
        self.assertIn("self._quantize_attention_o", runtime_source)
        self.assertIn("finish_attention_fp8(", runtime_source)
        self.assertNotIn("self_attn.finish_attention(o)", runtime_source)
        self.assertIn("self.num_lanes = min(2, self.num_mb)", runtime_source)
        self.assertIn("self.transport_slots = 2", runtime_source)
        self.assertIn("mb % len(lane_streams)", runtime_source)
        self.assertIn("slot = epoch % self.transport_slots", runtime_source)
        self.assertIn("self.o_slots[slot, :rows]", runtime_source)
        self.assertIn("self.q_ready[slot]", runtime_source)
        self.assertIn("self.o_ready[slot]", runtime_source)
        self.assertIn("self.q_publish_counters[slot : slot + 1]", runtime_source)
        self.assertIn("self.o_publish_counters[slot : slot + 1]", runtime_source)
        self.assertIn("attention_handle = self.q_owner.handle", runtime_source)
        self.assertIn("q_ready_handle=attention_handle", runtime_source)
        self.assertIn("kv_handle=attention_handle", runtime_source)
        self.assertIn("model_handle = self.o_owner.handle", runtime_source)
        self.assertIn("o_scale_handle=model_handle", runtime_source)
        self.assertIn("o_ready_handle=model_handle", runtime_source)
        self.assertIn("self._imported_allocations", runtime_source)
        self.assertIn("fmha_transport_arenas", runtime_source)
        self.assertIn("expected_import_arenas", runtime_source)
        self.assertIn("include_q_ready=source_index == 0", runtime_source)
        self.assertIn("shared_q_ready_dp", runtime_source)
        self.assertIn("expected_ready=epoch + 1", runtime_source)
        self.assertIn("ready_value=epoch + 1", runtime_source)
        self.assertNotIn("consumed_", runtime_source)
        self.assertIn('bootstrap_compute_events=bootstrap_compute_events', runtime_source)
        self.assertIn('lane_stream.wait_event(previous_compute_event)', runtime_source)
        self.assertIn('prepare_and_publish_qkv(next_round, compute_done_event)', runtime_source)
        self.assertIn('compute_done_event.record(lane_stream)', runtime_source)
        self.assertNotIn('pending_odd_combine', runtime_source)
        self.assertNotIn('cleanup_turn', runtime_source)
        self.assertNotIn('wait_turn(', runtime_source)
        self.assertNotIn('cleanup_ready_events', runtime_source)
        self.assertNotIn('reset_sentinel', runtime_source)
        self.assertIn('"mb_exclusive_compute_ping_pong"', runtime_source)
        self.assertNotIn("mb3", runtime_source.lower())
        self.assertNotIn("self.num_mb not in (1, 2)", runtime_source)
        self.assertEqual(runtime_source.count("retained_batch: Any"), 2)
        self.assertEqual(runtime_source.count("retained_batch=batch"), 2)

        worker_source = (ROOT / "python/minisgl/afd_worker_base.py").read_text()
        self.assertIn("for _ in range(min(2, self.afd_num_mb))", worker_source)
        self.assertIn("adapter_lanes = min(2, int(self.afd_num_mb))", worker_source)
        attention_worker_source = (
            ROOT / "python/minisgl/afd_attention_worker.py"
        ).read_text()
        expert_worker_source = (
            ROOT / "python/minisgl/afd_expert_worker.py"
        ).read_text()
        for worker_runtime_source in (
            attention_worker_source,
            expert_worker_source,
        ):
            self.assertIn("required_lanes = min(2, self.num_mb)", worker_runtime_source)
            self.assertIn("mb % len(adapters)", worker_runtime_source)
        self.assertIn(
            "mb % len(state.lane_streams)", attention_worker_source
        )
        self.assertNotIn("mapped_attention_sources", attention_worker_source)
        self.assertNotIn("fmha_source_switch_requires_drain", expert_worker_source)
        self.assertIn("AfdRunModelStepCmd", expert_worker_source)
        self.assertIn("run_fanin_step(cmd.plan", expert_worker_source)
        coordinator_source = (
            ROOT / "python/minisgl/afd_coordinator.py"
        ).read_text()
        self.assertIn("model_source_plans", coordinator_source)
        self.assertIn("AfdModelStepPlan(", coordinator_source)
        self.assertIn("AfdRunModelStepCmd(", coordinator_source)
        self.assertIn("actual_sources != expected_sources", coordinator_source)
        self.assertIn("def run_fanin_step(", runtime_source)
        self.assertIn("if self.fanin == 1:", runtime_source)
        self.assertIn(
            "batch = self.worker.runtime.materialize_ag_plan(namespaced)",
            runtime_source,
        )
        self.assertIn("self.receive_attention_o(sub, epoch)", runtime_source)
        self.assertNotIn("afd_fmha_source_subbatches", runtime_source)
        self.assertNotIn("_activate_transport_attn_dp", runtime_source)
        self.assertIn("mb % len(lane_streams)", expert_worker_source)
        args_source = (ROOT / "python/minisgl/server/args.py").read_text()
        self.assertIn("if num_mb < 1 or num_mb > batch_size:", args_source)

        combine_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepep/deep_ep/include/deep_ep/impls/combine.cuh"
        ).read_text()
        self.assertIn(
            "arrived == static_cast<unsigned long long>(kNumSMs - 1)",
            combine_source,
        )
        self.assertIn("atomicExch(counter, 0ull)", combine_source)
        self.assertIn("ptx::st_release_gpu(release_turn, release_value)", combine_source)
        self.assertNotIn("__threadfence();", combine_source)
        self.assertIn("release_turn", combine_source)
        self.assertIn("constexpr int kTransportSlots = 2", source)
        self.assertIn("slot < kTransportSlots", source)
        self.assertIn("constexpr int64_t kMaxPublicationBlocks = 1024", source)
        self.assertEqual(source.count("publication_blocks(tasks)"), 5)
        self.assertIn("first_flattened_task", source)
        self.assertEqual(source.count("global_task - edge_base"), 2)
        self.assertIn("bool kSingleEdge", source)
        self.assertIn("bool kSingleReady", source)
        self.assertIn("if constexpr (!kSingleEdge)", source)
        self.assertIn("source_offset_count.unwrap() == 2", source)
        self.assertEqual(source.count("ready_edges.unwrap() == 1"), 5)
        self.assertIn("q_edges.unwrap() == 1 and kv_edges.unwrap() == 1", source)
        self.assertEqual(source.count("single_ready and edges.unwrap() == 1"), 4)
        self.assertIn("publish_qkv_kernel<kHeadDim, true, true>", source)
        self.assertIn("publish_qkv_kernel<kHeadDim, false, false>", source)
        self.assertIn("publish_o_kernel<kHeadDim, true, true>", source)
        self.assertIn("publish_o_kernel_release_turn<kHeadDim, true, true>", source)
        self.assertIn("publish_o_fp8_kernel<kHeadDim, true, true>", source)
        self.assertIn(
            "publish_o_fp8_kernel_release_turn<kHeadDim, true, true>", source
        )
        release_start = source.index("void publish_o_kernel_release_turn(")
        release_end = source.index("__global__ void wait_ready_kernel", release_start)
        release_source = source[release_start:release_end]
        self.assertIn("store_release_device(params.turn", release_source)
        self.assertNotIn("__threadfence();", release_source)
        self.assertNotIn("__threadfence_system();", release_source)
        self.assertIn("ld.acquire.gpu.global.u64", source)
        self.assertIn("st.release.gpu.global.u64", source)
        self.assertNotIn("atomicAdd(ticket", source)

    def test_fmha_capture_uses_prewarms_instead_of_fabric_eager_execution(self) -> None:
        source = (ROOT / "python/minisgl/afd_fmha_runtime.py").read_text()
        tree = ast.parse(source)
        capture = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_capture_decode_graph"
        )
        eager_body_calls = [
            node
            for node in ast.walk(capture)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "body"
        ]
        self.assertEqual(eager_body_calls, [])
        self.assertIn("fn=body", ast.get_source_segment(source, capture))

    def test_deepgemm_reserves_the_deepep_communication_sms_once(self) -> None:
        source = (ROOT / "python/minisgl/afd_fmha_runtime.py").read_text()
        create_start = source.index("    def _create_moe_buffer")
        create_end = source.index("    def warmup_decode_graphs", create_start)
        create_source = source[create_start:create_end]
        self.assertIn(
            "self.fanin >= DeepEPMoeElasticBuffer.HIGH_FANIN_MIN_SOURCES",
            create_source,
        )
        self.assertIn("compute_sms = total_sms - communication_sms", create_source)
        self.assertIn("overlap_num_sms=communication_sms", create_source)
        self.assertLess(
            create_source.index('os.environ["MINISGL_MEGAMOE_AG_SMS"] = "0"'),
            create_source.index("deep_gemm.set_num_sms(compute_sms)"),
        )
        self.assertIn("int(deep_gemm.get_num_sms()) != compute_sms", create_source)
        self.assertNotIn("log=lambda", create_source)
        self.assertNotIn("fmha_decode_graph:replay", source)

    def test_deepgemm_psum_tuning_uses_declared_routing_density(self) -> None:
        dispatcher_source = (
            ROOT / "python/minisgl/layers/moe/token_dispatcher/deepep.py"
        ).read_text()
        self.assertIn("def _expected_m_per_expert", dispatcher_source)
        self.assertIn(
            "get_theoretical_mk_alignment_for_contiguous_layout(\n"
            "                int(expected_m_per_expert)",
            dispatcher_source,
        )
        self.assertIn(
            "max_tokens_per_rank = int(ctx.moe_deepep_dispatch_max_tokens_per_rank)",
            dispatcher_source,
        )
        runner_source = (
            ROOT
            / "python/minisgl/layers/moe/moe_runner/deepgemm_grouped.py"
        ).read_text()
        self.assertIn(
            "expected_m = int(dispatch_output.expected_m_per_expert)",
            runner_source,
        )
        self.assertIn("worker_blocks=min(M, expected_m * E)", runner_source)
        self.assertNotIn("expected_m = max(1, int(M) // max(1, E))", runner_source)
        psum_source = (
            ROOT / "python/minisgl/kernel/csrc/jit/psum_silu_mul_fp8_packed.cu"
        ).read_text()
        self.assertIn("static_cast<unsigned>(worker_blocks)", psum_source)
        self.assertNotIn("132 * 32", psum_source)
        router_source = (
            ROOT / "python/minisgl/kernel/csrc/jit/gate_topk_fused.cu"
        ).read_text()
        self.assertIn("const MapT* __restrict__ expert_map", router_source)
        self.assertIn("static_cast<IndexT>(mapped_expert)", router_source)
        self.assertIn("valid_token_count", router_source)
        self.assertIn("std::min(kTokensPerBlock, rows)", router_source)
        self.assertNotIn("dispatch_q", router_source)
        model_source = (ROOT / "python/minisgl/models/utils.py").read_text()
        self.assertIn("self.experts.prepare_deepep_from_gate(", model_source)
        self.assertNotIn(
            "router_logits = self.gate.forward(hidden_states)\n"
            "        return _MoEMLPPrepared(",
            model_source,
        )

    def test_fmha_megamoe_adds_fp8_fp8_without_replacing_fp8_fp4(self) -> None:
        api_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepgemm/csrc/apis/mega.hpp"
        ).read_text()
        self.assertIn('m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe)', api_source)
        self.assertIn('m.def("fp8_fp8_mega_moe", &fp8_fp8_mega_moe)', api_source)

        jit_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepgemm/csrc/jit_kernels/impls/"
            "sm100_fp8_fp4_mega_moe.hpp"
        ).read_text()
        self.assertIn("static void sm100_fp8_fp4_mega_moe(", jit_source)
        self.assertIn("static void sm100_fp8_fp8_mega_moe(", jit_source)
        self.assertIn("/* use_fp8_weights */ false", jit_source)
        self.assertIn("/* use_fp8_weights */ true", jit_source)

        kernel_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/"
            "impls/sm100_fp8_fp4_mega_moe.cuh"
        ).read_text()
        self.assertIn("std::conditional_t<kUseFP8Weights", kernel_source)
        self.assertIn(
            "kUseFP8Weights ? SMEM_B_SIZE_PER_STAGE * 2 : SMEM_B_SIZE_PER_STAGE",
            kernel_source,
        )

        adapter_source = (
            ROOT / "python/minisgl/moe/megamoe_afd.py"
        ).read_text()
        self.assertIn("requant_qwen_fp8_weights_per32(", adapter_source)
        self.assertIn("requant_qwen_fp8_weights_to_fp4(", adapter_source)
        self.assertIn("_mega.fp8_fp8_mega_moe", adapter_source)
        self.assertIn("_mega.fp8_fp4_mega_moe", adapter_source)
        self.assertNotIn("num_concurrent_lanes=self.num_lanes", adapter_source)
        self.assertIn("MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", adapter_source)

        wrapper_source = (
            ROOT / "python/minisgl/kernel/megamoe_mega.py"
        ).read_text()
        self.assertIn("def fp8_fp4_mega_moe(", wrapper_source)
        self.assertIn("_ext().fp8_fp4_mega_moe(", wrapper_source)

        weight_source = (
            ROOT / "python/minisgl/kernel/megamoe_m2n_mega.py"
        ).read_text()
        self.assertIn("def requant_qwen_fp8_weights_to_fp4(", weight_source)

        runtime_source = (ROOT / "python/minisgl/afd_fmha_runtime.py").read_text()
        create_start = runtime_source.index("    def _create_megamoe")
        create_end = runtime_source.index("    def _run_model_moe", create_start)
        create_source = runtime_source[create_start:create_end]
        self.assertIn("reserved_sms = 2", create_source)
        self.assertIn("compute_sms = total_sms - reserved_sms", create_source)
        self.assertIn("compute_sms <= 0 or compute_sms % 2", create_source)
        self.assertIn("deep_gemm.set_num_sms(compute_sms)", create_source)
        self.assertIn("reserved_sms={reserved_sms}", create_source)

    def test_fmha_only_defaults_to_same_rank_megamoe(self) -> None:
        support_source = (ROOT / "python/minisgl/afd_support.py").read_text()
        self.assertIn(
            'default_backend = "megamoe" if placement == "fmha-only" else "deepep"',
            support_source,
        )
        self.assertIn(
            'backend == "megamoe" and placement != "fmha-only"', support_source
        )
        self.assertIn(
            'backend == "megamoe_m2n" and placement != "legacy"', support_source
        )

        runner = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd_reproduce.sh"
        ).read_text()
        self.assertIn("legacy) DEFAULT_AFD_MOE_BACKEND=megamoe_m2n", runner)
        self.assertIn("fmha-only) DEFAULT_AFD_MOE_BACKEND=megamoe", runner)
        self.assertIn(
            "MINISGL_AFD_MOE_BACKEND=${MINISGL_AFD_MOE_BACKEND:-$DEFAULT_AFD_MOE_BACKEND}",
            runner,
        )
        self.assertIn("MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", runner)
        self.assertIn('FASTAFD_JOB_TIME_LIMIT:-01:00:00', runner)

        campaign = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn('if [[ "$AFD_MODEL_PLACEMENT" == fmha-only ]]; then', campaign)
        self.assertIn("DEFAULT_AFD_MOE_BACKEND=megamoe", campaign)
        self.assertIn("DEFAULT_AFD_MOE_BACKEND=deepep", campaign)
        self.assertIn("afd_moe_backend=%s", campaign)
        self.assertIn("MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", campaign)
        self.assertIn("FASTAFD_CASE_ID must identify the shape", campaign)
        self.assertIn('--case-id "$CASE_ID"', campaign)

    def test_model_lazy_setup_finishes_before_transport_ready_barrier(self) -> None:
        runtime_source = (ROOT / "python/minisgl/afd_fmha_runtime.py").read_text()
        init_start = runtime_source.index("    def __init__(self, worker: Any)")
        init_end = runtime_source.index("    def _init_graph_buffers", init_start)
        init_source = runtime_source[init_start:init_end]
        self.assertLess(
            init_source.index("ensure_afd_fmha_transport_built(self.head_dim)"),
            init_source.index("dist.barrier()"),
        )
        self.assertLess(
            init_source.index("self._prewarm_attention_decode_kernels()"),
            init_source.index("dist.barrier()"),
        )
        self.assertLess(
            init_source.index("self._prewarm_model_decode_kernels()"),
            init_source.index("dist.barrier()"),
        )
        self.assertIn(
            "prepare_layer(layer.mlp.experts)",
            runtime_source,
        )
        self.assertIn(
            "layer.mlp.experts.dispatcher.prepare_for_capture(self.device)",
            runtime_source,
        )
        prewarm_start = runtime_source.index(
            "    def _prewarm_model_decode_kernels(self) -> None:"
        )
        prewarm_end = runtime_source.index(
            "    def _create_tp_lane_communicators", prewarm_start
        )
        prewarm_source = runtime_source[prewarm_start:prewarm_end]
        self.assertIn("self.model.forward_lm_head(lm_head_input)", prewarm_source)
        self.assertIn("per_mb_bs * self.num_mb", prewarm_source)
        self.assertIn("backend.forward_prepared(q, 0, sub)", runtime_source)

        deepep_source = (
            ROOT / "python/minisgl/layers/moe/token_dispatcher/deepep.py"
        ).read_text()
        self.assertIn("def prepare_for_capture", deepep_source)
        self.assertIn("self._map_global_to_group_experts(device)", deepep_source)

        deepgemm_source = (
            ROOT
            / "python/minisgl/layers/moe/moe_runner/deepgemm_grouped.py"
        ).read_text()
        self.assertIn("def prepare_layer(self, layer: Any)", deepgemm_source)
        self.assertIn("self._get_fp8_weights_for_deepgemm(layer)", deepgemm_source)

        transport_source = (
            ROOT / "python/minisgl/kernel/csrc/jit/afd_fmha_transport.cu"
        ).read_text()
        self.assertNotIn("printf(", transport_source)


if __name__ == "__main__":
    unittest.main()
