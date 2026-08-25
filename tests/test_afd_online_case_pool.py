from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.afd.oci_hsg import afd_online_case_pool as pool
from scripts.experiments.afd.oci_hsg import extract_cuda_wall


ROOT = Path(__file__).resolve().parents[1]


class AfdOnlineCasePoolMetricTests(unittest.TestCase):
    def test_metric_versions_match_the_extractor(self) -> None:
        self.assertEqual(pool.METRIC_VERSION, extract_cuda_wall.METRIC_VERSION)
        self.assertEqual(
            pool.FMHA_METRIC_VERSION, extract_cuda_wall.FMHA_METRIC_VERSION
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="afd-pool-test-", dir=ROOT
        )
        self.root = Path(self.temporary.name)
        self.result_path = self.root / "afd-result.json"
        self.metric_path = self.root / "metric.json"
        self.case = {
            "case_id": "i131072-fep2-r1-atp1-b3",
            "required_retained_steps_min": "10",
            "required_max_outliers": "5",
            "required_dominant_range_percent_limit": "10.0",
            "required_max_median_diff_percent_limit": "10.0",
            "required_trace_decode_steps": "15",
            "required_target_batch_per_attention_dp_lane": "3",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_result(self, placement: str, attention_dp_size: int = 3) -> None:
        self.result_path.write_text(
            json.dumps(
                {
                    "afd_model_placement": placement,
                    "attention_dp_size": attention_dp_size,
                    "placement": {"mlp_dp_rank_nodes": [["node0"]]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def write_metric(self, **updates: object) -> None:
        decode_steps = list(range(100, 115))
        metric = {
            "case_id": self.case["case_id"],
            "metric_version": pool.FMHA_METRIC_VERSION,
            "result_sha256": pool.sha256(self.result_path),
            "target_batch_per_attention_dp_lane": 3,
            "eligible_target_batch_sample_count": 15,
            "decode_step_ids": decode_steps,
            "warmup_decode_step_id": 99,
            "trace_decode_step_ids": [99, *decode_steps],
            "profiled_roles": {
                "attention:rank1": {
                    "role": "attention",
                    "graph_launches_per_trace_step": 1,
                    "per_trace_step_cuda_graph_launch_ms": {
                        str(step): [1.0] for step in [99, *decode_steps]
                    },
                    "per_trace_step_cuda_graph_ms": {
                        str(step): 1.0 for step in [99, *decode_steps]
                    },
                },
                "model:rank2": {
                    "role": "model",
                    "graph_launches_per_trace_step": 1,
                    "per_trace_step_cuda_graph_launch_ms": {
                        str(step): [1.0] for step in [99, *decode_steps]
                    },
                    "per_trace_step_cuda_graph_ms": {
                        str(step): 1.0 for step in [99, *decode_steps]
                    },
                },
            },
            "num_microbatches": 2,
            "sample_count": 15,
            "outlier_count": 0,
            "dominant_range_percent": 1.0,
            "max_median_diff_percent": 1.0,
        }
        metric.update(updates)
        self.metric_path.write_text(
            json.dumps(metric) + "\n", encoding="utf-8"
        )

    def test_fmha_dual_role_metric_is_accepted(self) -> None:
        self.write_result("fmha-only")
        self.write_metric()
        observed = pool.validate_metric(
            self.metric_path, self.result_path, self.case
        )
        self.assertEqual(observed["metric_version"], pool.FMHA_METRIC_VERSION)

    def test_fmha_metric_requires_both_profiled_roles(self) -> None:
        self.write_result("fmha-only")
        self.write_metric(
            profiled_roles={"attention:rank1": {"role": "attention"}}
        )
        with self.assertRaisesRegex(RuntimeError, "strict corrected metric"):
            pool.validate_metric(self.metric_path, self.result_path, self.case)

    def test_fmha_metric_requires_exact_roles_and_preceding_warmup(self) -> None:
        self.write_result("fmha-only")
        self.write_metric(
            profiled_roles={
                "attention:rank1": {"role": "attention"},
                "attention:rank2": {"role": "attention"},
                "model:rank3": {"role": "model"},
            }
        )
        with self.assertRaisesRegex(RuntimeError, "strict corrected metric"):
            pool.validate_metric(self.metric_path, self.result_path, self.case)

    def test_fmha_metric_requires_one_grouped_model_launch(self) -> None:
        self.write_result("fmha-only")
        self.write_metric()
        metric = json.loads(self.metric_path.read_text(encoding="utf-8"))
        metric["profiled_roles"]["model:rank2"][
            "graph_launches_per_trace_step"
        ] = 2
        self.metric_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "strict corrected metric"):
            pool.validate_metric(self.metric_path, self.result_path, self.case)

        self.write_metric(
            warmup_decode_step_id=98,
            trace_decode_step_ids=[98, *range(100, 115)],
        )
        with self.assertRaisesRegex(RuntimeError, "strict corrected metric"):
            pool.validate_metric(self.metric_path, self.result_path, self.case)

    def test_legacy_metric_contract_remains_accepted(self) -> None:
        self.write_result("legacy")
        self.write_metric(
            metric_version=pool.METRIC_VERSION,
            target_batch_filter_passed=True,
            max_median_stability_check_passed=True,
        )
        observed = pool.validate_metric(
            self.metric_path, self.result_path, self.case
        )
        self.assertEqual(observed["metric_version"], pool.METRIC_VERSION)

    def test_recover_failed_completion_preserves_audit_and_proofs(self) -> None:
        self.write_result("fmha-only")
        self.write_metric()
        state_root = self.root / "state"
        for kind in (*pool.STATE_KINDS, "attempts"):
            (state_root / kind).mkdir(parents=True, exist_ok=True)
        failed_path = pool.state_path(state_root, "failed", self.case["case_id"])
        pool.atomic_write_json(
            failed_path,
            {
                "case_id": self.case["case_id"],
                "job_id": "12345",
                "allocated_trays": 1,
                "rerun_index": 1,
                "claimed_at_utc": "2026-08-22T00:00:00+00:00",
                "case": self.case,
                "failed_at_utc": "2026-08-22T00:01:00+00:00",
                "exit_code": 70,
                "reason": "cleanup bookkeeping failed after durable output",
                "retry_policy": "blocked",
            },
        )

        pool.command_recover_failed_completion(
            [self.case],
            state_root,
            self.case["case_id"],
            "12345",
            self.metric_path,
            self.result_path,
            "strict output revalidated after cleanup-path correction",
        )

        self.assertFalse(failed_path.exists())
        completed = json.loads(
            pool.state_path(
                state_root, "completed", self.case["case_id"]
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(completed["job_id"], "12345")
        self.assertEqual(completed["metric_sha256"], pool.sha256(self.metric_path))
        self.assertEqual(completed["result_sha256"], pool.sha256(self.result_path))
        archive = Path(completed["recovered_from_failed_archive"])
        self.assertTrue(archive.is_file())
        archived = json.loads(archive.read_text(encoding="utf-8"))
        self.assertEqual(archived["exit_code"], 70)
        self.assertEqual(
            archived["completion_recovery_reason"],
            "strict output revalidated after cleanup-path correction",
        )


if __name__ == "__main__":
    unittest.main()
