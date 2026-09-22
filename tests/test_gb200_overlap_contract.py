from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class GB200OverlapContractTests(unittest.TestCase):
    def test_fmha_runtime_uniformly_sizes_nondivisible_fanin_buffers(self) -> None:
        runtime = (ROOT / "python/minisgl/afd_fmha_runtime.py").read_text()
        self.assertIn("self.fanin = len(self._mapped_attn_dp_ranks)", runtime)
        self.assertIn("int(topology.attn_fanin_per_mlp_dp)", runtime)
        self.assertIn(
            "self.max_rows = self.source_max_rows * self.fanin_capacity",
            runtime,
        )
        self.assertIn("if self.fanin > self.fanin_capacity:", runtime)

    def test_comprehensive_launcher_supports_explicit_attention_gpu_count(self) -> None:
        launcher = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn(
            "ATTENTION_GPUS_OVERRIDE=${FASTAFD_ATTENTION_GPUS_OVERRIDE:-}",
            launcher,
        )
        self.assertIn(
            '[[ "$SWEEP_CONTRACT" == comprehensive ]]',
            launcher,
        )
        override = launcher.index("if [[ -n \"$ATTENTION_GPUS_OVERRIDE\" ]]")
        tray_limit = launcher.index("if [[ \"$SWEEP_CONTRACT\" == comprehensive ||")
        self.assertLess(override, tray_limit)
        self.assertIn("NODES=$(( (ACTIVE_GPUS + 3) / 4 ))", launcher[override:])
        self.assertIn("ATTENTION_GPUS=$ATTENTION_GPUS_OVERRIDE", launcher[override:])
        self.assertEqual(
            launcher.count(
                "FASTAFD_ATTENTION_GPUS_OVERRIDE=$ATTENTION_GPUS_OVERRIDE"
            ),
            1,
        )
        self.assertEqual(
            launcher.count(
                'FASTAFD_ATTENTION_GPUS_OVERRIDE="$ATTENTION_GPUS_OVERRIDE"'
            ),
            1,
        )
        self.assertEqual(
            launcher.count("FASTAFD_MAX_TOTAL_TRAYS=$MAX_TOTAL_TRAYS"),
            1,
        )
        self.assertEqual(
            launcher.count('FASTAFD_MAX_TOTAL_TRAYS="$MAX_TOTAL_TRAYS"'),
            1,
        )

    def test_perfect_routing_preserves_router_work_and_balances_ids(self) -> None:
        kernel = (
            ROOT / "python/minisgl/kernel/csrc/jit/gate_topk_fused.cu"
        ).read_text()
        select = kernel[kernel.index("gate_topk_select_kernel") :]
        self.assertIn("if constexpr (kPerfectBalance)", select)
        self.assertIn(
            "gate_topk_select_kernel<IndexT, MapT, kPrepareRoutes, kRankReady,",
            select,
        )
        self.assertIn("kPerfect, kRenormalize, kAllPush>", select)
        self.assertIn("if constexpr (!kRenormalize)", select)
        self.assertIn("if (renormalize)", select)
        perfect_start = select.index("if constexpr (kPerfectBalance)")
        normal_start = select.index("float logit[kMaxExpertsPerLane]")
        topk_start = select.index("for (int j = 0; j < topk; ++j)")
        self.assertLess(normal_start, topk_start)
        self.assertLess(topk_start, perfect_start)
        self.assertIn("GEMM, softmax, and top-k work above remains unchanged", select)
        self.assertIn("(num_experts - 1)", select)
        self.assertNotIn("lane) % num_experts", select)
        self.assertIn("perfect balance requires route preparation", kernel)
        self.assertIn(
            "perfect balance requires a power-of-two expert count", kernel
        )
        self.assertIn('#include "gate_topk_fused_common.cuh"', kernel)
        common = (
            ROOT / "python/minisgl/kernel/csrc/jit/gate_topk_fused_common.cuh"
        ).read_text()
        self.assertIn("inline constexpr int kMaxExperts = 512", common)
        self.assertIn("num_mtiles * num_expert_splits", kernel)

        for experts, topk in ((128, 8), (512, 10)):
            for ranks in (4, 8):
                for valid_tokens in range(1, 33):
                    counts = [0] * experts
                    for rank in range(ranks):
                        for token in range(valid_tokens):
                            for slot in range(topk):
                                expert = (
                                    (rank * valid_tokens + token) * topk + slot
                                ) % experts
                                counts[expert] += 1
                    self.assertLessEqual(max(counts) - min(counts), 1)

        adapter = (ROOT / "python/minisgl/moe/megamoe_afd.py").read_text()
        self.assertIn('"MINISGL_AFD_PERFECT_ROUTING", "0"', adapter)
        self.assertIn("perfect_balance=self.perfect_routing", adapter)

        campaign = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn("FASTAFD_PROMPT_BASE_FILE:-$SOURCE_REPO/prompts/", campaign)
        bundle = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd_bundle.sh"
        ).read_text()
        self.assertEqual(bundle.count('FASTAFD_PROMPT_BASE_FILE="$PROMPT_BASE_FILE"'), 2)
        self.assertEqual(bundle.count('FASTAFD_PROMPT_CACHE_ROOT="$PROMPT_CACHE_ROOT"'), 2)

    def test_bundle_pool_accepts_fp4_qualified_case_ids(self) -> None:
        path = (
            ROOT
            / "scripts/experiments/afd/oci_hsg/afd_online_case_pool.py"
        )
        spec = importlib.util.spec_from_file_location("afd_online_case_pool", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        case_id = "i131072-fep8-r7-atp1-b6-fp4"
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
        self.assertIn("attn-only allocation must remain in one NVL72 fabric block", source)
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
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
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
            'r"nsys profiler:target_decode_window target_batch_per_dp=(\\d+) "',
            source,
        )
        self.assertIn('r"warmup_step_id=(\\d+) step_ids=([0-9,]+) "', source)
        self.assertIn("int(trace_count_text) != 16", source)
        self.assertIn('r"(?:real=\\d+ )?bs=(\\d+) num_mb=(\\d+)"', source)
        self.assertIn("trace_step_set = set(trace_steps)", source)
        self.assertIn("steps[0] != warmup_step + 1", source)
        self.assertIn('"trace_decode_step_ids": trace_steps', source)
        self.assertIn("trace first exact full-resident decode as warmup", source)
        self.assertNotIn("MAX_TOKENS=16", source)

    def test_alignment_only_run_does_not_require_trace_artifacts(self) -> None:
        source = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn("NSYS=${NSYS:-1}", source)
        self.assertIn(
            'if [[ "${NSYS:-0}" == 1 ]]; then\n'
            '    [[ -s "$RUN_DIR/ray_logs/coordinator_nvtx_cpu.json" ]]',
            source,
        )
        self.assertIn("if nsys_enabled:\n    if model_placement", source)
        self.assertIn(
            'else "alignment-only; sample shape and token count validated"',
            source,
        )
        self.assertIn(
            '"status": "pending_postprocess" if nsys_enabled else "alignment_only"',
            source,
        )

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
        self.assertIn("per-case AFD perf contract is incomplete", bundle)
        self.assertIn(
            'FASTAFD_AFD_PERF_MODE="$case_perf_mode"',
            bundle,
        )
        self.assertIn(
            'MINISGL_AFD_PERFECT_ROUTING="$case_perfect_routing"',
            bundle,
        )
        self.assertIn(
            'argv[0] not in {"qwen3", "qwen35", "minimax"}', bundle
        )
        self.assertIn(
            'bash "$CASE_RUNNER" "$case_model" "$context"', bundle
        )
        self.assertNotIn(
            'bash "$CASE_RUNNER" qwen3 "$context"', bundle
        )

    def test_qwen35_gdn_fuses_and_overlaps_input_projections(self) -> None:
        model_source = (ROOT / "python/minisgl/models/qwen35_moe.py").read_text()
        project_start = model_source.index("    def _project_inputs(")
        project_end = model_source.index("    def forward(", project_start)
        project_source = model_source[project_start:project_end]
        start_record = project_source.index("start_event.record(main_stream)")
        qkvz_launch = project_source.index(
            "projected_qkvz = self.in_proj_qkvz.forward(hidden_states)",
            start_record,
        )
        ba_launch = project_source.index(
            "projected_ba = self.in_proj_ba.forward(hidden_states)", qkvz_launch
        )
        self.assertLess(start_record, qkvz_launch)
        self.assertLess(qkvz_launch, ba_launch)
        self.assertIn("with torch.cuda.stream(aux_stream):", project_source)
        self.assertIn("aux_stream.wait_event(start_event)", project_source)
        self.assertIn("done_event.record(aux_stream)", project_source)
        gate_launch = project_source.rindex(
            "g, beta = qwen35_gdn_gates(a, b, self.A_log, self.dt_bias)"
        )
        self.assertLess(ba_launch, gate_launch)
        self.assertLess(gate_launch, project_source.index("done_event.record(aux_stream)"))
        self.assertIn("projected_ba.record_stream(main_stream)", project_source)
        self.assertIn("g.record_stream(main_stream)", project_source)
        self.assertIn("beta.record_stream(main_stream)", project_source)
        self.assertNotIn("main_stream.wait_event(done_event)", project_source)

        forward_start = model_source.index("    def forward(", project_start)
        forward_end = model_source.index("    @staticmethod", forward_start)
        forward_source = model_source[forward_start:forward_end]
        self.assertLess(
            forward_source.index("qwen35_causal_conv1d_decode_packed("),
            forward_source.index("wait_event(side_done_event)"),
        )

        weight_source = (ROOT / "python/minisgl/models/weight.py").read_text()
        for old_name, fused_name in (
            ("in_proj_qkv", "in_proj_qkvz"),
            ("in_proj_z", "in_proj_qkvz"),
            ("in_proj_b", "in_proj_ba"),
            ("in_proj_a", "in_proj_ba"),
        ):
            self.assertIn(f'".{old_name}": (".{fused_name}"', weight_source)

        gdn_source = (ROOT / "python/minisgl/kernel/qwen35_gdn.py").read_text()
        decode_start = gdn_source.index("def qwen35_gdn_decode(")
        decode_end = gdn_source.index("\n\n__all__", decode_start)
        decode_source = gdn_source[decode_start:decode_end]
        self.assertNotIn("a = a.contiguous()", decode_source)
        self.assertNotIn("b = b.contiguous()", decode_source)
        self.assertIn("a.stride(0)", decode_source)
        self.assertIn("b.stride(0)", decode_source)
        kernel_start = gdn_source.index("def _qwen35_gdn_decode_kernel(")
        kernel_end = gdn_source.index("\n\ndef qwen35_gdn_decode(", kernel_start)
        kernel_source = gdn_source[kernel_start:kernel_end]
        self.assertIn("token_idx * a_row_stride + value_head", kernel_source)
        self.assertIn("token_idx * b_row_stride + value_head", kernel_source)
        conv_start = gdn_source.index("def qwen35_causal_conv1d_decode_packed(")
        conv_end = gdn_source.index("\n\ndef _qwen35_rmsnorm_gated(", conv_start)
        conv_source = gdn_source[conv_start:conv_end]
        self.assertIn("PACK_QKV=True", conv_source)
        self.assertIn("q = packed[:q_elements]", conv_source)
        self.assertIn("k = packed[q_elements : 2 * q_elements]", conv_source)
        self.assertIn("v = packed[2 * q_elements", conv_source)
        self.assertIn("q_norm_scale = torch.empty(", conv_source)
        self.assertIn("COMPUTE_QK_SCALES=True", conv_source)
        self.assertIn("z.stride(0)", gdn_source)

        self.assertIn("HAS_QK_NORM_SCALES", kernel_source)
        self.assertIn("q_norm_scale_ptr + norm_offset", kernel_source)
        self.assertIn("k_norm_scale_ptr + norm_offset", kernel_source)
        self.assertIn("HAS_PRECOMPUTED_GATES", kernel_source)
        self.assertIn("g = tl.load(g_ptr + gate_offset)", kernel_source)
        self.assertIn("beta = tl.load(beta_ptr + gate_offset)", kernel_source)
        self.assertIn("delta = (v - memory) * beta", kernel_source)

        self.assertIn("qwen35_rmsnorm_gated_fp8(", forward_source)
        self.assertIn("self.out_proj.forward_fp8_prequant(", forward_source)
        linear_source = (ROOT / "python/minisgl/layers/linear.py").read_text()
        o_proj_start = linear_source.index("class LinearOProj(")
        o_proj_end = linear_source.index("\n\nclass LinearRowParallel", o_proj_start)
        o_proj_source = linear_source[o_proj_start:o_proj_end]
        self.assertIn("output: torch.Tensor | None = None", o_proj_source)
        self.assertIn("output=output", o_proj_source)
        fp8_start = gdn_source.index("def _qwen35_rmsnorm_gated_fp8_kernel(")
        fp8_end = gdn_source.index("\n\ndef qwen35_rmsnorm_gated(", fp8_start)
        fp8_source = gdn_source[fp8_start:fp8_end]
        self.assertIn("rounded = (normalized * gate).to(", fp8_source)
        self.assertIn("scale.to(tl.int32, bitcast=True) >> 23", fp8_source)
        self.assertIn("local_row_id.to(tl.int32) * 8", fp8_source)

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
        finish_start = source.index("void finish_payload_publication(")
        finish_end = source.index("__global__ void wait_turn_kernel", finish_start)
        finish_source = source[finish_start:finish_end]
        self.assertEqual(finish_source.count("__syncthreads();"), 1)
        self.assertNotIn("publish_turn_kernel", source)
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
        self.assertNotIn("def publish_turn(", wrapper_source)
        self.assertNotIn("reset_sentinel", wrapper_source)
        self.assertNotIn("consumed_", wrapper_source)
        runtime_source = (ROOT / "python/minisgl/afd_fmha_runtime.py").read_text()
        self.assertNotIn("quantize_publish_o_fp8_release_turn(", runtime_source)
        # The legacy transport remains FP8, while Qwen3.5 hybrid attention
        # intentionally transports BF16 O and finishes its gated output
        # projection on the model worker.
        self.assertNotIn("consumed_", runtime_source)
        self.assertNotIn("MINISGL_AFD_QWEN_SHARED_EXPERT_OVERLAP", runtime_source)
        self.assertNotIn("self._shared_expert_streams", runtime_source)
        self.assertNotIn("extra_streams=preparation_streams", runtime_source)
        moe_start = runtime_source.index("    def _run_model_moe")
        moe_end = runtime_source.index("    def warmup_decode_graphs", moe_start)
        moe_source = runtime_source[moe_start:moe_end]
        shared_wait = moe_source.index("wait_turn(")
        routed_start = moe_source.index("output = mega_moe.forward(")
        self.assertLess(shared_wait, routed_start)
        self.assertNotIn("release_turn(\n                shared_ready_turn", moe_source)
        self.assertNotIn("torch.cuda.Event", moe_source)
        self.assertNotIn("wait_event", moe_source)
        overlap_shared_start = moe_source.index(
            "shared = layer.mlp.shared_expert.forward_fp8_prequant("
        )
        overlap_shared_end = moe_source.index(
            "output = mega_moe.forward(", overlap_shared_start
        )
        self.assertNotIn(
            "qwen35_shared_output(",
            moe_source[overlap_shared_start:overlap_shared_end],
        )
        self.assertNotIn("output = output + shared", moe_source)
        exact_norm = runtime_source.index(
            "layer.post_attention_layernorm.forward(",
            runtime_source.index("def _finish_model_layer("),
        )
        fused_shared_input = runtime_source.index(
            "fp8_quant_shared_gate(", exact_norm
        )
        shared_call = runtime_source.index("self._run_model_moe(", fused_shared_input)
        self.assertLess(exact_norm, fused_shared_input)
        self.assertLess(fused_shared_input, shared_call)
        self.assertNotIn('pending_odd_combine', runtime_source)
        self.assertNotIn('cleanup_turn', runtime_source)
        self.assertNotIn('cleanup_ready_events', runtime_source)
        self.assertNotIn('reset_sentinel', runtime_source)

        qwen35_transport = (
            ROOT / "python/minisgl/kernel/csrc/jit/qwen35_qkv_transport.cu"
        ).read_text()
        norm_quant_source = (
            ROOT / "python/minisgl/kernel/csrc/jit/fp8_quant_shared_gate.cu"
        ).read_text()
        self.assertNotIn("residual", norm_quant_source)
        self.assertNotIn("#pragma unroll 1", norm_quant_source)
        self.assertNotIn("expf(", qwen35_transport)
        self.assertLess(
            norm_quant_source.index('"st.release.gpu.global.u64'),
            norm_quant_source.index("expf(-gate_value)"),
        )
        self.assertNotIn("mb3", runtime_source.lower())
        self.assertNotIn("self.num_mb not in (1, 2)", runtime_source)
        self.assertEqual(runtime_source.count("retained_batch: Any"), 2)
        self.assertEqual(runtime_source.count("retained_batch=batch"), 2)

        attention_worker_source = (
            ROOT / "python/minisgl/afd_attention_worker.py"
        ).read_text()
        expert_worker_source = (
            ROOT / "python/minisgl/afd_expert_worker.py"
        ).read_text()
        self.assertNotIn("mapped_attention_sources", attention_worker_source)
        self.assertNotIn("fmha_source_switch_requires_drain", expert_worker_source)
        self.assertNotIn("afd_fmha_source_subbatches", runtime_source)
        self.assertNotIn("_activate_transport_attn_dp", runtime_source)

        combine_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepep/deep_ep/include/deep_ep/impls/combine.cuh"
        ).read_text()
        self.assertNotIn("__threadfence();", combine_source)
        fused_start = source.index("__global__ void quantize_publish_o_fp8_kernel(")
        fused_end = source.index("__global__ void wait_ready_kernel", fused_start)
        fused_source = source[fused_start:fused_end]
        quantize_at = fused_source.index("quantize_o_fp8_to_stage")
        join_at = fused_source.index("__syncthreads();")
        publish_at = fused_source.index("publish_staged_o_fp8_payload")
        ready_at = fused_source.index("finish_payload_publication")
        self.assertLess(quantize_at, join_at)
        self.assertLess(join_at, publish_at)
        self.assertLess(publish_at, ready_at)
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
        m2n_source = (
            ROOT / "python/minisgl/moe/deepep_m2n_adapter.py"
        ).read_text()
        self.assertIn("expected_m_per_expert: int", m2n_source)
        self.assertIn("def _expected_m_per_expert", m2n_source)
        self.assertIn(
            "get_theoretical_mk_alignment_for_contiguous_layout(\n"
            "                    int(expected_m_per_expert)",
            m2n_source,
        )
        transport_source = (
            ROOT / "python/minisgl/kernel/csrc/jit/afd_fmha_transport.cu"
        ).read_text()
        self.assertIn(
            "const bool no_kv = kv_desc.shape().Product() == 0", transport_source
        )
        self.assertIn("const int64_t kv_edge_count = no_kv ? 0", transport_source)
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
        iket_source = (
            ROOT / "codex_scripts/profile_megamoe_iket.sbatch"
        ).read_text()
        self.assertNotIn("MEGAMOE_SMOKE_PERFECT_ROUTING=", iket_source)
        self.assertNotIn("MEGAMOE_SMOKE_SPECIALIZATION_MODES=", iket_source)
        self.assertNotIn('"comm_bookend_proxy_us"', iket_source)


        kernel_source = (
            ROOT
            / "python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/"
            "impls/sm100_fp8_fp4_mega_moe.cuh"
        ).read_text()
        self.assertNotIn("float2 pair_sums[4];", kernel_source)


        router_source = (
            ROOT / "python/minisgl/kernel/csrc/jit/gate_topk_fused.cu"
        ).read_text()
        self.assertEqual(
            router_source.count("if (!defer_route_dependent_launch)"), 2
        )

        adapter_source = (
            ROOT / "python/minisgl/moe/megamoe_afd.py"
        ).read_text()
        self.assertNotIn("num_concurrent_lanes=self.num_lanes", adapter_source)


        m2n_adapter_source = (
            ROOT / "python/minisgl/moe/megamoe_m2n_afd.py"
        ).read_text()
        self.assertEqual(
            m2n_adapter_source.count("use_fp8_weights=self.use_fp8_weights"), 3
        )


    def test_attn_only_defaults_to_same_rank_megamoe(self) -> None:
        support_source = (ROOT / "python/minisgl/afd_support.py").read_text()
        self.assertIn(
            'default_backend = "megamoe" if placement == "attn-only" else "deepep"',
            support_source,
        )
        self.assertIn(
            'backend == "megamoe" and placement != "attn-only"', support_source
        )
        self.assertIn(
            'backend == "megamoe_m2n" and placement != "legacy"', support_source
        )

        campaign = (
            ROOT / "scripts/experiments/afd/oci_hsg/run_afd.sh"
        ).read_text()
        self.assertIn('if [[ "$AFD_MODEL_PLACEMENT" == attn-only ]]; then', campaign)
        self.assertIn("DEFAULT_AFD_MOE_BACKEND=megamoe", campaign)
        self.assertIn("DEFAULT_AFD_MOE_BACKEND=megamoe_m2n", campaign)
        self.assertIn("DEFAULT_AFD_MOE_BACKEND=deepep", campaign)
        self.assertIn("deepep|megamoe|megamoe_m2n", campaign)
        self.assertIn("megamoe_m2n requires legacy placement", campaign)
        self.assertIn("afd_moe_backend=%s", campaign)
        self.assertIn("MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", campaign)
        self.assertIn("MINISGL_MEGAMOE_EXPERT_WEIGHT_PATH", campaign)
        self.assertIn("Qwen3.5-397B-A17B-FP8-MXFP4-MoE", campaign)
        self.assertIn("attn-only:megamoe", campaign)
        self.assertIn("legacy:megamoe_m2n", campaign)
        self.assertIn("FASTAFD_CASE_ID must identify the shape", campaign)
        self.assertIn('--case-id "$CASE_ID"', campaign)
        self.assertNotIn("CASE_ID=single", campaign)
        self.assertIn("CASE_ID=$DEFAULT_CASE_ID", campaign)

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
        self.assertIn("backend.forward_prepared(q, full_layer, sub)", runtime_source)

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
        self.assertIn("release_turn_kernel", transport_source)
        self.assertIn("__threadfence_system()", transport_source)

        qwen35_transport = (
            ROOT / "python/minisgl/kernel/csrc/jit/qwen35_qkv_transport.cu"
        ).read_text()
        self.assertIn("normalize_and_rope", qwen35_transport)
        self.assertIn("params.q_gate", qwen35_transport)
        self.assertNotIn("qwen35_publish_linear_input_kernel", qwen35_transport)
        self.assertIn("qwen35_shared_add_rmsnorm_kernel", qwen35_transport)
        self.assertIn("params.residual", qwen35_transport)
        self.assertIn("__threadfence_system()", qwen35_transport)
        self.assertNotIn("printf(", qwen35_transport)

        publish_start = runtime_source.index("    def publish_model_qkv(")
        publish_end = runtime_source.index("    def prepare_model_qkv(", publish_start)
        publish_source = runtime_source[publish_start:publish_end]
        self.assertIn("if self.is_hybrid_attention:", publish_source)
        self.assertIn("self.publish_qwen35_full_attention_qkv(", publish_source)
        # Qwen3 retains its established preparation and generic publisher.
        self.assertIn("q, k, v = self.prepare_model_qkv", publish_source)
        self.assertIn("self.publish_model_qkv_payload(", publish_source)



if __name__ == "__main__":
    unittest.main()
