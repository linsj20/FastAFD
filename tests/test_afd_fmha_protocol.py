from __future__ import annotations

import unittest

import torch

from minisgl.afd_fmha_protocol import (
    AfdTpSliceTable,
    build_moe_tp_column_groups,
    fmha_epoch,
    validate_decode_microbatch_token_counts,
)
from minisgl.afd_protocol import (
    AfdAGStepPlan,
    AfdCommand,
    AfdModelStepPlan,
    AfdRunModelStepCmd,
    AfdTokenBlock,
    AfdTopology,
    build_two_lane_pipeline_actions,
)
from minisgl.afd_scheduler import _balanced_microbatch_counts


class AfdFmhaProtocolTest(unittest.TestCase):
    @staticmethod
    def _decode_source_plan(attn_dp_rank: int) -> AfdAGStepPlan:
        return AfdAGStepPlan(
            step_id=7,
            phase="decode",
            real_size=6,
            attn_dp_rank=attn_dp_rank,
            table_indices=tuple(range(6)),
            extend_lens=(1,) * 6,
            microbatch_offsets=(0, 3, 6),
            microbatch_token_offsets=(0, 3, 6),
            microbatch_real_token_counts=(3, 3),
            exec_table_indices=tuple(range(6)),
            exec_start_positions=(10,) * 6,
            exec_extend_lens=(1,) * 6,
            exec_sample_indices=tuple(range(6)),
            exec_writebacks=(True,) * 6,
            dispatch_bucket=3,
            use_decode_graph=True,
        )

    def test_model_dp_fanin_covers_each_attention_lane_once(self) -> None:
        for attn_dp_size, mlp_dp_size in ((1, 1), (2, 1), (3, 1), (15, 1), (8, 4)):
            topology = AfdTopology(
                attn_dp_size=attn_dp_size,
                mlp_dp_size=mlp_dp_size,
                attn_tp_size=1,
                mlp_tp_size=1,
            )
            groups = tuple(
                topology.attn_dps_for_mlp_dp(mlp_dp_rank)
                for mlp_dp_rank in range(mlp_dp_size)
            )
            self.assertEqual(
                tuple(attn_dp for group in groups for attn_dp in group),
                tuple(range(attn_dp_size)),
            )
            for mlp_dp_rank, group in enumerate(groups):
                self.assertTrue(group)
                self.assertTrue(
                    all(
                        topology.mlp_dp_for_attn_dp(attn_dp_rank) == mlp_dp_rank
                        for attn_dp_rank in group
                    )
                )

        topology = AfdTopology(3, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            topology.attn_dps_for_mlp_dp(1)

    def test_model_step_aggregates_sources_without_adding_ffn_rounds(self) -> None:
        sources = tuple(self._decode_source_plan(rank) for rank in range(3))
        plan = AfdModelStepPlan(source_plans=sources, dispatch_bucket=9)

        self.assertEqual(plan.step_id, 7)
        self.assertEqual(plan.num_mb, 2)
        self.assertEqual(plan.real_size, 18)
        self.assertEqual(plan.microbatch_offsets, (0, 9, 18))
        self.assertEqual(plan.microbatch_token_offsets, (0, 9, 18))
        self.assertEqual(plan.microbatch_real_token_counts, (9, 9))
        self.assertEqual(
            94 * plan.num_mb,
            188,
            "FFN rounds are layers times microbatches, independent of fan-in",
        )

        uneven = self._decode_source_plan(3)
        uneven.microbatch_offsets = (0, 2, 6)
        uneven.microbatch_token_offsets = (0, 2, 6)
        uneven.microbatch_real_token_counts = (2, 4)
        with self.assertRaisesRegex(ValueError, "identical microbatch buckets"):
            AfdModelStepPlan(
                source_plans=(*sources, uneven),
                dispatch_bucket=12,
            )

    def test_model_prefill_command_codec_preserves_all_source_blocks(self) -> None:
        source_plans = tuple(
            AfdAGStepPlan(
                step_id=8,
                phase="prefill",
                real_size=1,
                attn_dp_rank=rank,
                table_indices=(0,),
                extend_lens=(2,),
                microbatch_offsets=(0, 1),
                microbatch_token_offsets=(0, 2),
                microbatch_real_token_counts=(2,),
                exec_table_indices=(0,),
                exec_start_positions=(0,),
                exec_extend_lens=(2,),
                exec_sample_indices=(0,),
                exec_writebacks=(False,),
                token_blocks=(
                    AfdTokenBlock(
                        table_idx=0,
                        start_pos=0,
                        tokens=torch.tensor([rank, rank + 1], dtype=torch.int32),
                    ),
                ),
                dispatch_bucket=2,
            )
            for rank in range(2)
        )
        command = AfdRunModelStepCmd(
            plan=AfdModelStepPlan(source_plans=source_plans, dispatch_bucket=4),
            sent_ns=123,
        )

        decoded = AfdCommand.decoder(AfdCommand.encoder(command))

        self.assertIsInstance(decoded, AfdRunModelStepCmd)
        self.assertEqual(decoded.sent_ns, 123)
        self.assertEqual(
            [plan.token_blocks[0].tokens.tolist() for plan in decoded.plan.source_plans],
            [[0, 1], [1, 2]],
        )

    def test_ffn_actions_complete_each_lane_before_reuse(self) -> None:
        for num_mb in range(1, 9):
            actions = build_two_lane_pipeline_actions(
                num_layers=3,
                num_microbatches=num_mb,
            )
            active: dict[int, tuple[int, int]] = {}
            completed: list[tuple[int, int]] = []
            for action, layer, mb in actions:
                lane = mb % min(2, num_mb)
                if action == "dispatch":
                    self.assertNotIn(lane, active)
                    active[lane] = (layer, mb)
                else:
                    self.assertEqual(active.pop(lane), (layer, mb))
                    completed.append((layer, mb))
            self.assertEqual(active, {})
            self.assertEqual(
                sorted(completed),
                [(layer, mb) for layer in range(3) for mb in range(num_mb)],
            )

        self.assertEqual(
            build_two_lane_pipeline_actions(num_layers=2, num_microbatches=2),
            (
                ("dispatch", 0, 0),
                ("dispatch", 0, 1),
                ("complete", 0, 0),
                ("dispatch", 1, 0),
                ("complete", 0, 1),
                ("dispatch", 1, 1),
                ("complete", 1, 0),
                ("complete", 1, 1),
            ),
        )

    def test_decode_remainder_can_bias_the_final_microbatches(self) -> None:
        self.assertEqual(_balanced_microbatch_counts(6, 2), (3, 3))
        self.assertEqual(_balanced_microbatch_counts(7, 2), (4, 3))
        self.assertEqual(
            _balanced_microbatch_counts(7, 2, remainder_to_last=True),
            (3, 4),
        )
        self.assertEqual(
            _balanced_microbatch_counts(8, 2, remainder_to_last=True),
            (4, 4),
        )
        self.assertEqual(
            _balanced_microbatch_counts(7, 3, remainder_to_last=True),
            (2, 2, 3),
        )
        self.assertEqual(
            _balanced_microbatch_counts(6, 6, remainder_to_last=True),
            (1, 1, 1, 1, 1, 1),
        )

    def test_every_microbatch_count_through_batch_size_is_valid(self) -> None:
        for batch_size in range(1, 17):
            for num_mb in range(1, batch_size + 1):
                with self.subTest(batch_size=batch_size, num_mb=num_mb):
                    counts = _balanced_microbatch_counts(
                        batch_size,
                        num_mb,
                        remainder_to_last=True,
                    )
                    self.assertEqual(len(counts), num_mb)
                    self.assertEqual(sum(counts), batch_size)
                    self.assertGreaterEqual(min(counts), 1)
                    self.assertLessEqual(max(counts) - min(counts), 1)
                    self.assertEqual(tuple(sorted(counts)), counts)

    def test_moe_groups_are_dense_tp_columns_for_all_tp_degrees(self) -> None:
        for dp_size in (1, 2, 3, 4, 8):
            for tp_size in (1, 2, 4, 8, 16):
                with self.subTest(dp_size=dp_size, tp_size=tp_size, mode="replicated"):
                    replicated = build_moe_tp_column_groups(
                        mlp_dp_size=dp_size,
                        mlp_tp_size=tp_size,
                        ep_size=tp_size,
                    )
                    if tp_size == 1:
                        self.assertEqual(replicated, ())
                    else:
                        self.assertEqual(len(replicated), dp_size * tp_size)
                        self.assertTrue(all(len(group) == 1 for group in replicated))

                full_ep_size = dp_size * tp_size
                with self.subTest(dp_size=dp_size, tp_size=tp_size, mode="full"):
                    full = build_moe_tp_column_groups(
                        mlp_dp_size=dp_size,
                        mlp_tp_size=tp_size,
                        ep_size=full_ep_size,
                    )
                    if full_ep_size == 1:
                        self.assertEqual(full, ())
                    else:
                        self.assertEqual(len(full), tp_size)
                        for tp_rank, group in enumerate(full):
                            self.assertEqual(
                                group,
                                tuple((dp_rank, tp_rank) for dp_rank in range(dp_size)),
                            )

    def test_tp_bridge_exact_coverage_in_both_directions(self) -> None:
        valid_tp_sizes = (1, 2, 4, 8, 16, 32, 64)
        tp_pairs = (
            (attn_tp, model_tp)
            for attn_tp in valid_tp_sizes
            for model_tp in valid_tp_sizes
            if max(attn_tp, model_tp) % min(attn_tp, model_tp) == 0
        )
        for attn_tp, model_tp in tp_pairs:
            with self.subTest(attn_tp=attn_tp, model_tp=model_tp):
                table = AfdTpSliceTable.build(
                    attn_tp_size=attn_tp,
                    model_tp_size=model_tp,
                    num_qo_heads=64,
                    num_kv_heads=8,
                )

                table.validate_exact_coverage()
                for slices, heads in (
                    (table.q_slices, 64),
                    (table.kv_slices, attn_tp * table.attn_local_kv_heads),
                    (table.o_slices, 64),
                ):
                    self.assertEqual(
                        sum(head_slice.head_count for head_slice in slices),
                        heads,
                    )

    def test_kv_replication_assigns_one_writer_to_every_attention_replica(self) -> None:
        table = AfdTpSliceTable.build(
            attn_tp_size=16,
            model_tp_size=8,
            num_qo_heads=64,
            num_kv_heads=8,
        )

        destinations = [
            (head_slice.destination_tp_rank, head_slice.destination_head_start)
            for head_slice in table.kv_slices
            for _ in range(head_slice.head_count)
        ]
        self.assertEqual(sorted(destinations), [(rank, 0) for rank in range(16)])
        self.assertEqual(
            {head_slice.source_tp_rank for head_slice in table.kv_slices},
            set(range(8)),
        )

    def test_tp_bridge_rejects_unsupported_shards(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually divisible"):
            AfdTpSliceTable.build(
                attn_tp_size=3,
                model_tp_size=4,
                num_qo_heads=64,
                num_kv_heads=8,
            )
        with self.assertRaisesRegex(ValueError, "Q/O heads divisible"):
            AfdTpSliceTable.build(
                attn_tp_size=8,
                model_tp_size=16,
                num_qo_heads=72,
                num_kv_heads=8,
            )

    def test_any_microbatch_count_uses_monotonic_two_slot_epochs(self) -> None:
        for num_mb in (1, 2, 3, 6):
            rounds = 94 * num_mb
            for step_id in (0, 1, 9):
                epochs = [
                    fmha_epoch(
                        step_id=step_id,
                        layer_id=layer_id,
                        microbatch_id=mb,
                        num_layers=94,
                        num_microbatches=num_mb,
                    )
                    for layer_id in range(94)
                    for mb in range(num_mb)
                ]
                self.assertEqual(
                    epochs,
                    list(range(step_id * rounds, (step_id + 1) * rounds)),
                )
                self.assertEqual(
                    [epoch & 1 for epoch in epochs],
                    [
                        index & 1
                        for index in range(
                            step_id * rounds,
                            (step_id + 1) * rounds,
                        )
                    ],
                )

    def test_decode_graph_accepts_per_lane_padding(self) -> None:
        self.assertEqual(
            validate_decode_microbatch_token_counts(
                microbatch_real_token_counts=(3, 3),
                microbatch_token_offsets=(0, 3, 6),
            ),
            (3, 3),
        )
        self.assertEqual(
            validate_decode_microbatch_token_counts(
                microbatch_real_token_counts=(4, 3),
                microbatch_token_offsets=(0, 4, 8),
            ),
            (4, 3),
        )
        self.assertEqual(
            validate_decode_microbatch_token_counts(
                microbatch_real_token_counts=(3, 4),
                microbatch_token_offsets=(0, 4, 8),
            ),
            (3, 4),
        )
        self.assertEqual(
            validate_decode_microbatch_token_counts(
                microbatch_real_token_counts=(3, 2),
                microbatch_token_offsets=(0, 4, 8),
            ),
            (3, 2),
        )

    def test_decode_graph_rejects_a_lane_count_larger_than_its_span(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds its graph span"):
            validate_decode_microbatch_token_counts(
                microbatch_real_token_counts=(5, 2),
                microbatch_token_offsets=(0, 4, 8),
            )

if __name__ == "__main__":
    unittest.main()
