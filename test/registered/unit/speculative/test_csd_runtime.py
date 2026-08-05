from collections import Counter

import torch

from sglang.srt.speculative.csd_runtime import (
    CSDTableStore,
    _build_hash_table_cpu,
    pack_csd_pair,
    unpack_csd_pair,
)


def test_csd_pair_table_round_trip(tmp_path):
    first = pack_csd_pair(7, 11)
    second = pack_csd_pair(13, 17)
    assert unpack_csd_pair(first) == (7, 11)

    path = tmp_path / "table.json"
    CSDTableStore(
        counts=Counter({first: 6, second: 2}), metadata={"source": "unit"}
    ).save(str(path))
    loaded = CSDTableStore.load(str(path))
    assert loaded.counts == Counter({first: 6, second: 2})
    assert loaded.metadata["source"] == "unit"

    table, inserted = _build_hash_table_cpu([first])
    assert table.dtype == torch.int64
    assert inserted == 1
    assert (table == first).sum().item() == 1
