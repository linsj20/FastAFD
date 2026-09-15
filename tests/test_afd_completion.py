from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from minisgl.afd_coordinator import AfdCoordinator, _AfdForwardRemain


class AfdCompletionTest(unittest.TestCase):
    def test_completion_without_profiling_preserves_replies_and_frees(self):
        coordinator = object.__new__(AfdCoordinator)
        coordinator._profile_control_step = lambda step: False
        coordinator._afd_collect_global = Mock(return_value={0: [11, 22]})
        coordinator.send_result = Mock()
        coordinator.send_detokenize_batch = Mock()
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(table_idx=i) for i in range(2)],
            padded_reqs=[object(), object()], size=2,
        )
        completion = SimpleNamespace(
            reply_uids=[100, 101], reply_next_tokens=[11, 22],
            reply_finished=[True, False], freed_table_indices=(0,),
        )
        scheduler = SimpleNamespace(complete_results=Mock(return_value=completion))
        remain = _AfdForwardRemain()
        with patch("minisgl.afd_coordinator.time.perf_counter_ns",
                   side_effect=AssertionError("profiling clock used while disabled")):
            freed = coordinator._afd_complete_global(
                (7, [(0, batch, (0, 0))]), [scheduler], [remain], 1
            )
        scheduler.complete_results.assert_called_once_with(batch, [11, 22])
        coordinator.send_detokenize_batch.assert_called_once_with(
            [100, 101], [11, 22], [True, False]
        )
        self.assertEqual(freed, {0: (0,)})
        self.assertEqual((remain.generation(0), remain.generation(1)), (1, 0))

    def test_complete_reply_preserves_tokens(self):
        batch = SimpleNamespace(reqs=[SimpleNamespace(table_idx=i) for i in range(2)])
        tokens = [11, 22]
        self.assertIs(AfdCoordinator._afd_filter_valid_completion_tokens(
            batch, (0, 0), tokens, _AfdForwardRemain()), tokens)

    def test_dummy_source_has_no_real_reply(self):
        batch = SimpleNamespace(reqs=[], padded_reqs=[object(), object()])
        self.assertEqual(AfdCoordinator._afd_filter_valid_completion_tokens(
            batch, (), [], _AfdForwardRemain()), [])

    def test_stale_generation_discards_only_its_request(self):
        requests = [SimpleNamespace(table_idx=i) for i in range(2)]
        batch = SimpleNamespace(reqs=requests.copy())
        remain = _AfdForwardRemain()
        remain.forget_tables((0,))
        self.assertEqual(AfdCoordinator._afd_filter_valid_completion_tokens(
            batch, (0, 0), [11, 22], remain), [22])
        self.assertEqual(batch.reqs, requests[1:])

    def test_malformed_reply_fails_before_request_mutation(self):
        for stale in (False, True):
            for gens, tokens in [((0, 0), []), ((0, 0), [11]),
                                 ((0, 0), [11, 22, 33]), ((0,), [11, 22]),
                                 ((0, 0, 0), [11, 22])]:
                with self.subTest(stale=stale, gens=gens, tokens=tokens):
                    requests = [SimpleNamespace(table_idx=i) for i in range(2)]
                    batch = SimpleNamespace(reqs=requests.copy())
                    remain = _AfdForwardRemain()
                    if stale:
                        remain.forget_tables((0,))
                    with self.assertRaisesRegex(RuntimeError, "cover every real request"):
                        AfdCoordinator._afd_filter_valid_completion_tokens(
                            batch, gens, tokens, remain)
                    self.assertEqual(batch.reqs, requests)


if __name__ == "__main__":
    unittest.main()
