import unittest
from collections import Counter

import torch

from sglang.srt.speculative.csd_runtime import (
    CSD_KEY_SELECTION_ABOVE_UNIFORM_SHARE,
    CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL,
    CSDDeltaBuffer,
    CSDHashTable,
    CSDMetrics,
    CSDRuntime,
    CSDTableStore,
    pack_csd_pair,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestCSDRuntime(CustomTestCase):
    def test_count_squared_over_total_score(self):
        store = CSDTableStore()
        store.add_pair(1, 10, freq=60)
        store.add_pair(1, 11, freq=40)
        store.add_pair(2, 20, freq=9)
        store.add_pair(2, 21, freq=1)

        self.assertAlmostEqual(store.pair_score(pack_csd_pair(1, 10)), 36.0)
        self.assertAlmostEqual(store.pair_score(pack_csd_pair(2, 20)), 8.1)

    def test_count_squared_over_total_strategy_filters_and_ranks(self):
        store = CSDTableStore()
        store.add_pair(1, 10, freq=60)
        store.add_pair(1, 11, freq=40)
        store.add_pair(2, 20, freq=9)
        store.add_pair(2, 21, freq=1)

        self.assertEqual(
            store.filtered_keys(
                freq_threshold=1,
                key_selection_strategy=CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL,
                score_threshold=10,
            ),
            [pack_csd_pair(1, 10), pack_csd_pair(1, 11)],
        )
        self.assertEqual(
            store.filtered_keys(
                freq_threshold=1,
                top_keep=1.1,
                key_selection_strategy=CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL,
            ),
            [pack_csd_pair(1, 10)],
        )

    def test_count_squared_over_total_strategy_ignores_freq_threshold(self):
        store = CSDTableStore()
        store.add_pair(1, 10, freq=2)
        store.add_pair(1, 11, freq=1)

        self.assertEqual(
            store.filtered_keys(
                freq_threshold=100,
                key_selection_strategy=CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL,
                score_threshold=1,
            ),
            [pack_csd_pair(1, 10)],
        )

    def test_above_uniform_share_strategy_filters_singletons(self):
        store = CSDTableStore()
        store.add_pair(1, 10, freq=1)
        store.add_pair(2, 20, freq=3)
        store.add_pair(2, 21, freq=1)
        store.add_pair(2, 22, freq=1)
        store.add_pair(3, 30, freq=2)
        store.add_pair(3, 31, freq=2)

        self.assertEqual(
            store.filtered_keys(
                freq_threshold=2,
                key_selection_strategy=CSD_KEY_SELECTION_ABOVE_UNIFORM_SHARE,
            ),
            [pack_csd_pair(2, 20)],
        )

    def test_rejects_unknown_key_selection_strategy(self):
        store = CSDTableStore()
        store.add_pair(1, 2, freq=1)

        with self.assertRaisesRegex(
            ValueError, "Unsupported CSD key selection strategy"
        ):
            store.filtered_keys(freq_threshold=1, key_selection_strategy="unknown")

    def test_filtered_keys_top_keep_ratio_and_count(self):
        store = CSDTableStore()
        store.add_pair(1, 2, freq=10)
        store.add_pair(3, 4, freq=7)
        store.add_pair(5, 6, freq=5)
        store.add_pair(7, 8, freq=1)

        self.assertCountEqual(
            store.filtered_keys(freq_threshold=2),
            [pack_csd_pair(1, 2), pack_csd_pair(3, 4), pack_csd_pair(5, 6)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=2, top_keep=0.5),
            [pack_csd_pair(1, 2), pack_csd_pair(3, 4)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=6, top_keep=0.5),
            [pack_csd_pair(1, 2), pack_csd_pair(3, 4)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=2, top_keep=0.05),
            [pack_csd_pair(1, 2)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=2, top_keep=2),
            [pack_csd_pair(1, 2), pack_csd_pair(3, 4)],
        )

    def test_frequency_keys_are_updated_incrementally(self):
        store = CSDTableStore()
        store.add_pair(1, 2, freq=5)
        store.add_pair(3, 4, freq=1)

        self.assertCountEqual(
            store.filtered_keys(freq_threshold=6),
            [],
        )
        store.merge_counts(
            Counter(
                {
                    pack_csd_pair(1, 2): 1,
                    pack_csd_pair(3, 4): 5,
                    pack_csd_pair(5, 6): 6,
                }
            )
        )

        self.assertCountEqual(
            store.filtered_keys(freq_threshold=6),
            [
                pack_csd_pair(1, 2),
                pack_csd_pair(3, 4),
                pack_csd_pair(5, 6),
            ],
        )
        store.add_pair(1, 2, freq=2)
        self.assertCountEqual(
            store.filtered_keys(freq_threshold=6),
            [pack_csd_pair(3, 4), pack_csd_pair(5, 6)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=2, top_keep=10),
            [pack_csd_pair(3, 4), pack_csd_pair(5, 6), pack_csd_pair(1, 2)],
        )

    def test_dynamic_rebuild_uses_top_keep_without_pruning_store(self):
        table_store = CSDTableStore()
        table_store.add_pair(1, 2, freq=1)
        table_store.add_pair(3, 4, freq=1)
        table_store.add_pair(5, 6, freq=1)
        delta_buffer = CSDDeltaBuffer.allocate(device="cpu", capacity=8)
        pairs = [
            pack_csd_pair(1, 2),
            pack_csd_pair(1, 2),
            pack_csd_pair(1, 2),
            pack_csd_pair(3, 4),
            pack_csd_pair(3, 4),
            pack_csd_pair(5, 6),
        ]
        delta_buffer.pairs[: len(pairs)] = torch.tensor(pairs, dtype=torch.int64)
        delta_buffer.counter.fill_(len(pairs))
        runtime = CSDRuntime(
            enabled=True,
            dynamic_update=True,
            force_accept_disabled=False,
            dynamic_update_ignore_prob_ratio=False,
            table=CSDHashTable.empty(device="cpu"),
            metrics=CSDMetrics.allocate(device="cpu"),
            delta_buffer=delta_buffer,
            table_store=table_store,
            online_rebuild_enabled=True,
        )

        started = runtime.maybe_start_async_rebuild(
            freq_threshold=1,
            rebuild_threshold=1,
            top_keep=0.5,
        )
        self.assertTrue(started)
        self.assertIsNotNone(runtime.rebuild_future)
        self.assertEqual(runtime.metrics_snapshot()["csd_rebuild_started_ct"], 1)
        self.assertEqual(runtime.metrics_snapshot()["csd_rebuild_inflight"], 1)
        runtime.rebuild_future.result(timeout=5)
        runtime.maybe_apply_async_rebuild(device="cpu")

        self.assertEqual(runtime.table.num_entries, 2)
        self.assertTrue(runtime.table.contains_for_test(pack_csd_pair(1, 2)))
        self.assertTrue(runtime.table.contains_for_test(pack_csd_pair(3, 4)))
        self.assertFalse(runtime.table.contains_for_test(pack_csd_pair(5, 6)))
        self.assertEqual(len(runtime.table_store.entries), 3)
        snapshot = runtime.metrics_snapshot()
        self.assertEqual(snapshot["csd_rebuild_applied_ct"], 1)
        self.assertEqual(snapshot["csd_rebuild_inflight"], 0)
        self.assertGreater(snapshot["csd_rebuild_lifecycle_wall_sec"], 0.0)

    def test_dynamic_rebuild_uses_pair_score_filter(self):
        table_store = CSDTableStore()
        table_store.add_pair(1, 10, freq=6)
        table_store.add_pair(1, 11, freq=4)
        table_store.add_pair(2, 20, freq=9)
        table_store.add_pair(2, 21, freq=1)
        delta_buffer = CSDDeltaBuffer.allocate(device="cpu", capacity=2)
        delta_buffer.pairs[0] = pack_csd_pair(2, 20)
        delta_buffer.counter.fill_(1)
        runtime = CSDRuntime(
            enabled=True,
            dynamic_update=True,
            force_accept_disabled=False,
            dynamic_update_ignore_prob_ratio=False,
            table=CSDHashTable.empty(device="cpu"),
            metrics=CSDMetrics.allocate(device="cpu"),
            delta_buffer=delta_buffer,
            table_store=table_store,
            online_rebuild_enabled=True,
        )

        started = runtime.maybe_start_async_rebuild(
            freq_threshold=1,
            rebuild_threshold=1,
            key_selection_strategy=CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL,
            score_threshold=3,
        )
        self.assertTrue(started)
        runtime.rebuild_future.result(timeout=5)
        runtime.maybe_apply_async_rebuild(device="cpu")

        self.assertTrue(runtime.table.contains_for_test(pack_csd_pair(1, 10)))
        self.assertFalse(runtime.table.contains_for_test(pack_csd_pair(1, 11)))
        self.assertTrue(runtime.table.contains_for_test(pack_csd_pair(2, 20)))
        self.assertFalse(runtime.table.contains_for_test(pack_csd_pair(2, 21)))


if __name__ == "__main__":
    unittest.main()
