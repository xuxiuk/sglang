import unittest

import torch

from sglang.srt.speculative.csd_runtime import (
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
    def test_filtered_keys_top_freq_ratio(self):
        store = CSDTableStore()
        store.add_pair(1, 2, freq=10)
        store.add_pair(3, 4, freq=7)
        store.add_pair(5, 6, freq=5)
        store.add_pair(7, 8, freq=1)

        self.assertEqual(
            store.filtered_keys(freq_threshold=2),
            [pack_csd_pair(1, 2), pack_csd_pair(3, 4), pack_csd_pair(5, 6)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=2, top_freq_ratio=0.5),
            [pack_csd_pair(1, 2), pack_csd_pair(3, 4)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=6, top_freq_ratio=0.5),
            [pack_csd_pair(1, 2), pack_csd_pair(3, 4)],
        )
        self.assertEqual(
            store.filtered_keys(freq_threshold=2, top_freq_ratio=0.05),
            [pack_csd_pair(1, 2)],
        )

    def test_dynamic_rebuild_uses_top_freq_ratio_without_pruning_store(self):
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
            table=CSDHashTable.empty(device="cpu"),
            metrics=CSDMetrics.allocate(device="cpu"),
            delta_buffer=delta_buffer,
            table_store=table_store,
            online_rebuild_enabled=True,
        )

        started = runtime.maybe_start_async_rebuild(
            freq_threshold=1,
            rebuild_threshold=1,
            top_freq_ratio=0.5,
        )
        self.assertTrue(started)
        self.assertIsNotNone(runtime.rebuild_future)
        runtime.rebuild_future.result(timeout=5)
        runtime.maybe_apply_async_rebuild(device="cpu")

        self.assertEqual(runtime.table.num_entries, 2)
        self.assertTrue(runtime.table.contains_for_test(pack_csd_pair(1, 2)))
        self.assertTrue(runtime.table.contains_for_test(pack_csd_pair(3, 4)))
        self.assertFalse(runtime.table.contains_for_test(pack_csd_pair(5, 6)))
        self.assertEqual(len(runtime.table_store.entries), 3)


if __name__ == "__main__":
    unittest.main()
