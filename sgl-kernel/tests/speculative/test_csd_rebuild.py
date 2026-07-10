import pytest
import torch


CSD_EMPTY_KEY = -1
UINT64_MASK = (1 << 64) - 1


def hash64(key: int) -> int:
    x = key & UINT64_MASK
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & UINT64_MASK
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & UINT64_MASK
    return (x ^ (x >> 31)) & UINT64_MASK


def build_reference(keys, max_probe: int, load_factor: float):
    if not keys:
        return [CSD_EMPTY_KEY]
    capacity = 1
    minimum_capacity = max(2, int(len(keys) / load_factor + 0.999999))
    while capacity < minimum_capacity:
        capacity *= 2
    while True:
        table = [CSD_EMPTY_KEY] * capacity
        for key in keys:
            slot = hash64(key) & (capacity - 1)
            for _ in range(max_probe):
                if table[slot] in (CSD_EMPTY_KEY, key):
                    table[slot] = key
                    break
                slot = (slot + 1) & (capacity - 1)
            else:
                capacity *= 2
                break
        else:
            return table


@pytest.mark.parametrize("num_keys", [0, 1, 17, 1000, 10000])
def test_csd_build_hash_table_cpu(num_keys):
    generator = torch.Generator().manual_seed(7)
    keys = torch.randperm(10_000_000, generator=generator)[:num_keys].to(torch.int64)
    table, num_entries = torch.ops.sgl_kernel.csd_build_hash_table_cpu(
        keys, 16, 0.5
    )

    assert table.tolist() == build_reference(keys.tolist(), 16, 0.5)
    assert num_entries == num_keys


def test_csd_build_hash_table_cpu_deduplicates():
    keys = torch.tensor([1, 2, 1, 3, 2], dtype=torch.int64)
    table, num_entries = torch.ops.sgl_kernel.csd_build_hash_table_cpu(
        keys, 16, 0.5
    )

    assert num_entries == 3
    assert sorted(value for value in table.tolist() if value >= 0) == [1, 2, 3]


@pytest.mark.parametrize(
    "keys,max_probe,load_factor",
    [
        (torch.tensor([-1], dtype=torch.int64), 16, 0.5),
        (torch.tensor([1], dtype=torch.int64), 0, 0.5),
        (torch.tensor([1], dtype=torch.int64), 16, 0.0),
    ],
)
def test_csd_build_hash_table_cpu_rejects_invalid_input(
    keys, max_probe, load_factor
):
    with pytest.raises(RuntimeError):
        torch.ops.sgl_kernel.csd_build_hash_table_cpu(
            keys, max_probe, load_factor
        )


def test_csd_table_builder_updates_and_snapshots_counts():
    keys = torch.tensor([10, 20, 30], dtype=torch.int64)
    freqs = torch.tensor([5, 6, 1], dtype=torch.int64)
    builder = torch.classes.sgl_kernel.CSDTableBuilder(keys, freqs, 6)

    pairs = torch.tensor([10, 10, 30, 40, 40, 40, 40, 40, 40], dtype=torch.int64)
    table, num_entries, store_entries = builder.update_and_build(pairs, 16, 0.5)

    assert num_entries == 3
    assert store_entries == 4
    for key in [10, 20, 40]:
        slot = hash64(key) & (table.numel() - 1)
        assert any(
            int(table[(slot + probe) & (table.numel() - 1)]) == key
            for probe in range(16)
        )

    snapshot_keys, snapshot_freqs = builder.snapshot()
    assert dict(zip(snapshot_keys.tolist(), snapshot_freqs.tolist())) == {
        10: 7,
        20: 6,
        30: 2,
        40: 6,
    }
