from __future__ import annotations

import unittest

from scripts.experiments.afd.oci_hsg.extract_cuda_wall import (
    collapse_fmha_graph_spans,
    fmha_profile_launches_per_step,
)


class CollapseFmhaGraphSpansTests(unittest.TestCase):
    def test_grouped_model_has_one_launch_independent_of_fanin(self) -> None:
        self.assertEqual(fmha_profile_launches_per_step(4, 4), (1, 1))
        self.assertEqual(fmha_profile_launches_per_step(8, 4), (1, 1))
        self.assertEqual(fmha_profile_launches_per_step(44, 4), (1, 1))

    def test_grouped_profile_rejects_nonintegral_fanin(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fan-in is not integral"):
            fmha_profile_launches_per_step(7, 4)

    def test_single_launch_per_step_is_unchanged(self) -> None:
        collapsed, node_count, grouped = collapse_fmha_graph_spans(
            [1.0, 2.0], [7, 7], [10, 11], 1
        )
        self.assertEqual(collapsed, [1.0, 2.0])
        self.assertEqual(node_count, 7)
        self.assertEqual(grouped, [[1.0], [2.0]])

    def test_serial_fanin_launches_are_summed_per_step(self) -> None:
        collapsed, node_count, grouped = collapse_fmha_graph_spans(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [9, 9, 9, 9, 9, 9],
            [20, 21],
            3,
        )
        self.assertEqual(collapsed, [6.0, 15.0])
        self.assertEqual(node_count, 9)
        self.assertEqual(grouped, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_launch_count_must_match_trace_topology(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not match trace topology"):
            collapse_fmha_graph_spans([1.0, 2.0], [7, 7], [10, 11], 2)

    def test_launch_node_counts_must_be_stable(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "node counts are not stable"):
            collapse_fmha_graph_spans([1.0, 2.0], [7, 8], [10, 11], 1)


if __name__ == "__main__":
    unittest.main()
