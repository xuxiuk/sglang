from __future__ import annotations

import heapq
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch

_CSD_NATIVE_EXTENSION_PATH = os.environ.get("SGLANG_CSD_NATIVE_EXTENSION_PATH")
if _CSD_NATIVE_EXTENSION_PATH:
    torch.ops.load_library(_CSD_NATIVE_EXTENSION_PATH)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

CSD_EMPTY_KEY = -1
CSD_MAX_TOKEN_ID = (1 << 31) - 1
CSD_DEFAULT_LOAD_FACTOR = 0.5
CSD_DEFAULT_MAX_PROBE = 16
CSD_DEFAULT_DELTA_CAPACITY = 1 << 20
CSD_DEFAULT_REBUILD_CHECK_INTERVAL = 1
CSD_KEY_SELECTION_FREQUENCY = "frequency"
CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL = "count_squared_over_total"
CSD_KEY_SELECTION_ABOVE_UNIFORM_SHARE = "above_uniform_share"
CSD_KEY_SELECTION_STRATEGIES = (
    CSD_KEY_SELECTION_FREQUENCY,
    CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL,
    CSD_KEY_SELECTION_ABOVE_UNIFORM_SHARE,
)
_UINT64_MASK = (1 << 64) - 1
_CSD_REBUILD_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="csd-rebuild"
)


def _csd_profile_range(name: str):
    if os.environ.get("SGLANG_CSD_PROFILE_ANNOTATIONS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return torch.profiler.record_function(name)
    return nullcontext()


def pack_csd_pair(lhs_token: int, rhs_token: int) -> int:
    if not 0 <= lhs_token <= CSD_MAX_TOKEN_ID:
        raise ValueError(
            f"lhs_token must be in [0, {CSD_MAX_TOKEN_ID}], got {lhs_token}"
        )
    if not 0 <= rhs_token <= CSD_MAX_TOKEN_ID:
        raise ValueError(
            f"rhs_token must be in [0, {CSD_MAX_TOKEN_ID}], got {rhs_token}"
        )
    return (lhs_token << 32) | rhs_token


def unpack_csd_pair(key: int) -> Tuple[int, int]:
    if key < 0:
        raise ValueError(f"CSD key must be non-negative, got {key}")
    return key >> 32, key & 0xFFFFFFFF


def _hash64(key: int) -> int:
    x = key & _UINT64_MASK
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return (x ^ (x >> 31)) & _UINT64_MASK


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@dataclass
class CSDEntry:
    freq: int = 1
    allow: bool = True

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "CSDEntry":
        return cls(
            freq=int(record.get("freq", 1)),
            allow=bool(record.get("allow", True)),
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "freq": int(self.freq),
            "allow": bool(self.allow),
        }


@dataclass
class CSDTableStore:
    entries: Dict[int, CSDEntry] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _filtered_keys_cache: Dict[
        Tuple[int, Optional[float], str, float, int],
        List[int],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _rank_heap: List[Tuple[int, int, int]] = field(
        default_factory=list, init=False, repr=False
    )
    _key_versions: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _rank_heap_initialized: bool = field(default=False, init=False, repr=False)
    _frequency_key_sets: Dict[int, set[int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _version: int = field(default=0, init=False, repr=False)

    @classmethod
    def load(cls, path: str) -> "CSDTableStore":
        table_path = Path(path)
        if table_path.suffix == ".jsonl":
            return cls._load_jsonl(table_path)
        return cls._load_json(table_path)

    @classmethod
    def _load_json(cls, path: Path) -> "CSDTableStore":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            metadata: Dict[str, Any] = {}
            records = payload
        else:
            metadata = dict(payload.get("metadata", {}))
            records = payload.get("entries", [])

        store = cls(metadata=metadata)
        for record in records:
            key = _record_to_key(record)
            store.entries[key] = CSDEntry.from_record(record)
        return store

    @classmethod
    def _load_jsonl(cls, path: Path) -> "CSDTableStore":
        store = cls()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "metadata":
                    store.metadata.update(record.get("metadata", {}))
                    continue
                key = _record_to_key(record)
                store.entries[key] = CSDEntry.from_record(record)
        return store

    def save(self, path: str) -> None:
        table_path = Path(path)
        if table_path.suffix == ".jsonl":
            self._save_jsonl(table_path)
        else:
            self._save_json(table_path)

    def _save_json(self, path: Path) -> None:
        payload = {
            "metadata": self.metadata,
            "entries": self.to_records(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _save_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            if self.metadata:
                f.write(
                    json.dumps({"type": "metadata", "metadata": self.metadata}) + "\n"
                )
            for record in self.to_records():
                f.write(json.dumps(record) + "\n")

    def _touch_key(self, key: int) -> None:
        entry = self.entries[key]
        version = self._key_versions.get(key, 0) + 1
        self._key_versions[key] = version
        heapq.heappush(self._rank_heap, (-entry.freq, key, version))

    def _ensure_rank_heap(self) -> None:
        if self._rank_heap_initialized:
            return
        self._rank_heap.clear()
        self._key_versions.clear()
        for key in self.entries:
            self._touch_key(key)
        self._rank_heap_initialized = True

    def _top_filtered_keys_from_heap(
        self,
        freq_threshold: int,
        keep_count: int,
    ) -> List[int]:
        self._ensure_rank_heap()
        keys: List[int] = []
        skipped: List[Tuple[int, int, int]] = []
        while self._rank_heap and len(keys) < keep_count:
            item = heapq.heappop(self._rank_heap)
            _, key, version = item
            entry = self.entries.get(key)
            if entry is None or self._key_versions.get(key) != version:
                continue
            if entry.freq < freq_threshold:
                skipped.append(item)
                break
            keys.append(key)
            skipped.append(item)
        for item in skipped:
            heapq.heappush(self._rank_heap, item)
        return keys

    def _lhs_totals(self) -> Counter[int]:
        """Return recorded replacement counts grouped by rejected draft token."""
        totals: Counter[int] = Counter()
        for key, entry in self.entries.items():
            lhs_token, _ = unpack_csd_pair(key)
            totals[lhs_token] += entry.freq
        return totals

    def _lhs_distinct_replacements(self) -> Counter[int]:
        """Return the number of observed replacement tokens per draft token."""
        distinct: Counter[int] = Counter()
        for key in self.entries:
            lhs_token, _ = unpack_csd_pair(key)
            distinct[lhs_token] += 1
        return distinct

    def pair_score(self, key: int, lhs_totals: Optional[Counter[int]] = None) -> float:
        """Return count(draft, replacement)^2 / count(draft, *).

        The score rewards pair support while penalizing rejected draft tokens
        that map to many different replacements. It can also be read as
        pair_frequency * pair_share, where pair_share is only an explanatory
        term and is not exposed as a separate table-selection strategy.
        """
        entry = self.entries.get(key)
        if entry is None:
            return 0.0
        lhs_token, _ = unpack_csd_pair(key)
        totals = lhs_totals if lhs_totals is not None else self._lhs_totals()
        lhs_total = totals[lhs_token]
        return entry.freq * entry.freq / lhs_total if lhs_total > 0 else 0.0

    def add_pair(
        self,
        lhs_token: int,
        rhs_token: int,
        freq: int = 1,
        allow: bool = True,
    ) -> None:
        key = pack_csd_pair(lhs_token, rhs_token)
        old_entry = self.entries.get(key)
        self.entries[key] = CSDEntry(
            freq=freq,
            allow=allow,
        )
        old_freq = old_entry.freq if old_entry is not None else 0
        for threshold, keys in self._frequency_key_sets.items():
            if old_freq < threshold <= freq:
                keys.add(key)
            elif freq < threshold <= old_freq:
                keys.discard(key)
        if self._rank_heap_initialized:
            self._touch_key(key)
        self._version += 1
        self._filtered_keys_cache.clear()

    def merge_counts(self, counts: Counter[int]) -> None:
        if not counts:
            return
        for key, count in counts.items():
            entry = self.entries.get(key)
            old_freq = entry.freq if entry is not None else 0
            if entry is None:
                self.entries[key] = CSDEntry(freq=int(count), allow=True)
            else:
                entry.freq += int(count)
            new_freq = self.entries[key].freq
            for threshold, keys in self._frequency_key_sets.items():
                if old_freq < threshold <= new_freq:
                    keys.add(key)
            if self._rank_heap_initialized:
                self._touch_key(key)
        self._version += 1
        self._filtered_keys_cache.clear()

    def filtered_keys(
        self,
        freq_threshold: int,
        top_keep: Optional[float] = None,
        key_selection_strategy: str = CSD_KEY_SELECTION_FREQUENCY,
        score_threshold: float = 0.0,
    ) -> List[int]:
        if key_selection_strategy not in CSD_KEY_SELECTION_STRATEGIES:
            raise ValueError(
                f"Unsupported CSD key selection strategy: {key_selection_strategy}"
            )
        cache_key = (
            int(freq_threshold),
            top_keep,
            key_selection_strategy,
            float(score_threshold),
            self._version,
        )
        cached_keys = self._filtered_keys_cache.get(cache_key)
        if cached_keys is not None:
            return list(cached_keys)

        if (
            key_selection_strategy == CSD_KEY_SELECTION_FREQUENCY
            and (top_keep is None or top_keep <= 0)
        ):
            keys = self._frequency_key_sets.get(freq_threshold)
            if keys is None:
                keys = {
                    key
                    for key, entry in self.entries.items()
                    if entry.freq >= freq_threshold
                }
                self._frequency_key_sets[freq_threshold] = keys
            result = list(keys)
            self._filtered_keys_cache[cache_key] = result
            return list(result)

        items = self.entries.items()
        use_pair_score = (
            key_selection_strategy == CSD_KEY_SELECTION_COUNT_SQUARED_OVER_TOTAL
        )
        use_above_uniform_share = (
            key_selection_strategy == CSD_KEY_SELECTION_ABOVE_UNIFORM_SHARE
        )
        lhs_totals = (
            self._lhs_totals() if use_pair_score or use_above_uniform_share else None
        )
        lhs_distinct = (
            self._lhs_distinct_replacements() if use_above_uniform_share else None
        )

        def above_uniform_share(key: int, entry: CSDEntry) -> bool:
            lhs_token, _ = unpack_csd_pair(key)
            lhs_total = lhs_totals[lhs_token] if lhs_totals is not None else 0
            num_replacements = (
                lhs_distinct[lhs_token] if lhs_distinct is not None else 0
            )
            if lhs_total <= 0 or num_replacements <= 0:
                return False
            # Keep pairs whose observed replacement share is above the uniform
            # baseline over that draft token's replacement choices:
            # count(d, t) / count(d, *) > 1 / K(d).
            return entry.freq * num_replacements > lhs_total

        def above_uniform_rank(key: int, entry: CSDEntry) -> float:
            lhs_token, _ = unpack_csd_pair(key)
            lhs_total = lhs_totals[lhs_token] if lhs_totals is not None else 0
            num_replacements = (
                lhs_distinct[lhs_token] if lhs_distinct is not None else 0
            )
            if lhs_total <= 0 or num_replacements <= 0:
                return 0.0
            return entry.freq * num_replacements / lhs_total

        def filtered_items() -> List[Tuple[int, CSDEntry]]:
            return [
                (key, entry)
                for key, entry in items
                if (
                    use_pair_score
                    or use_above_uniform_share
                    or entry.freq >= freq_threshold
                )
                and (
                    not use_pair_score
                    or self.pair_score(key, lhs_totals) >= score_threshold
                )
                and (
                    not use_above_uniform_share
                    or (
                        entry.freq >= freq_threshold and above_uniform_share(key, entry)
                    )
                )
            ]

        def sorted_filtered_keys() -> List[int]:
            # Ranking is only needed when top_keep truncates the candidate set.
            # Without top_keep, the hash table only needs membership, so the
            # unsorted filtered key set is equivalent and avoids a full sort.
            if use_pair_score:
                sort_key = lambda item: (
                    -self.pair_score(item[0], lhs_totals),
                    -item[1].freq,
                    item[0],
                )
            elif use_above_uniform_share:
                sort_key = lambda item: (
                    -above_uniform_rank(item[0], item[1]),
                    -item[1].freq,
                    item[0],
                )
            else:
                sort_key = lambda item: (-item[1].freq, item[0])
            return [key for key, entry in sorted(filtered_items(), key=sort_key)]

        if top_keep is None or top_keep <= 0:
            keys = [key for key, entry in filtered_items()]
        else:
            if top_keep <= 1:
                keep_count = max(1, math.ceil(len(self.entries) * top_keep))
            else:
                keep_count = max(1, math.floor(top_keep))
            if (
                use_pair_score
                or use_above_uniform_share
                or keep_count * 2 >= len(self.entries)
            ):
                sorted_keys = sorted_filtered_keys()
                keys = (
                    sorted_keys
                    if len(sorted_keys) <= keep_count
                    else sorted_keys[:keep_count]
                )
            else:
                keys = self._top_filtered_keys_from_heap(freq_threshold, keep_count)

        self._filtered_keys_cache[cache_key] = list(keys)
        return keys

    def build_allow_hash_table(
        self,
        device: torch.device | str,
        freq_threshold: int,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
        load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
        top_keep: Optional[float] = None,
        key_selection_strategy: str = CSD_KEY_SELECTION_FREQUENCY,
        score_threshold: float = 0.0,
    ) -> "CSDHashTable":
        return build_csd_hash_table(
            self.filtered_keys(
                freq_threshold,
                top_keep=top_keep,
                key_selection_strategy=key_selection_strategy,
                score_threshold=score_threshold,
            ),
            device=device,
            max_probe=max_probe,
            load_factor=load_factor,
        )

    def to_records(self) -> List[Dict[str, Any]]:
        records = []
        for key, entry in sorted(self.entries.items()):
            lhs_token, rhs_token = unpack_csd_pair(key)
            record = entry.to_record()
            record.update(
                {
                    "key": int(key),
                    "lhs_token": int(lhs_token),
                    "rhs_token": int(rhs_token),
                }
            )
            records.append(record)
        return records


def _record_to_key(record: Dict[str, Any]) -> int:
    if "key" in record:
        key = int(record["key"])
        if key == CSD_EMPTY_KEY:
            raise ValueError("CSD table entry cannot use the empty-key sentinel")
        return key
    return pack_csd_pair(int(record["lhs_token"]), int(record["rhs_token"]))


@dataclass
class CSDHashTable:
    keys: torch.Tensor
    num_entries: int
    capacity: int
    max_probe: int

    @classmethod
    def empty(
        cls,
        device: torch.device | str,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
    ) -> "CSDHashTable":
        return cls(
            keys=torch.full((1,), CSD_EMPTY_KEY, dtype=torch.int64, device=device),
            num_entries=0,
            capacity=1,
            max_probe=max_probe,
        )

    @property
    def has_entries(self) -> bool:
        return self.num_entries > 0

    def contains_for_test(self, key: int) -> bool:
        slot = _hash64(int(key)) & (self.capacity - 1)
        keys = self.keys.detach().cpu().tolist()
        for _ in range(self.max_probe):
            existing_key = keys[slot]
            if existing_key == int(key):
                return True
            if existing_key == CSD_EMPTY_KEY:
                return False
            slot = (slot + 1) & (self.capacity - 1)
        return False


@dataclass
class CSDHashTablePayload:
    keys: List[int] | torch.Tensor
    num_entries: int
    capacity: int
    max_probe: int
    store_entries: Optional[int] = None


def build_csd_rebuild_payload_native(
    native_builder: Any,
    pairs: torch.Tensor,
    max_probe: int,
    load_factor: float,
) -> CSDHashTablePayload:
    with _csd_profile_range("csd_rebuild_native:update_and_build"):
        hash_keys, num_entries, store_entries = native_builder.update_and_build(
            pairs,
            max_probe,
            load_factor,
        )
    return CSDHashTablePayload(
        keys=hash_keys,
        num_entries=num_entries,
        capacity=hash_keys.numel(),
        max_probe=max_probe,
        store_entries=store_entries,
    )


def build_csd_rebuild_payload_from_counts(
    table_store: CSDTableStore,
    counts: Counter[int],
    freq_threshold: int,
    max_probe: int = CSD_DEFAULT_MAX_PROBE,
    load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
    top_keep: Optional[float] = None,
    key_selection_strategy: str = CSD_KEY_SELECTION_FREQUENCY,
    score_threshold: float = 0.0,
) -> CSDHashTablePayload:
    with _csd_profile_range("csd_rebuild_start:merge_counts"):
        table_store.merge_counts(counts)
    with _csd_profile_range("csd_rebuild_start:filtered_keys"):
        keys = table_store.filtered_keys(
            freq_threshold,
            top_keep=top_keep,
            key_selection_strategy=key_selection_strategy,
            score_threshold=score_threshold,
        )
    with _csd_profile_range("csd_rebuild_start:build_payload"):
        return build_csd_hash_table_payload(
            keys,
            max_probe=max_probe,
            load_factor=load_factor,
        )


def materialize_csd_hash_table_payload(
    payload: CSDHashTablePayload,
    device: torch.device | str,
) -> CSDHashTable:
    keys = payload.keys
    if isinstance(keys, torch.Tensor):
        keys = keys.to(device=device, dtype=torch.int64)
    else:
        keys = torch.tensor(keys, dtype=torch.int64, device=device)
    return CSDHashTable(
        keys=keys,
        num_entries=payload.num_entries,
        capacity=payload.capacity,
        max_probe=payload.max_probe,
    )


def build_csd_hash_table_payload(
    keys: Iterable[int],
    max_probe: int = CSD_DEFAULT_MAX_PROBE,
    load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
) -> CSDHashTablePayload:
    if max_probe < 1:
        raise ValueError("CSD max_probe must be at least 1")
    if not 0 < load_factor <= 1:
        raise ValueError("CSD load_factor must be in the range (0, 1]")

    native_builder = getattr(
        getattr(torch.ops, "sgl_kernel", None),
        "csd_build_hash_table_cpu",
        None,
    )
    if native_builder is not None:
        key_tensor = torch.as_tensor(
            keys if isinstance(keys, (list, torch.Tensor)) else list(keys),
            dtype=torch.int64,
            device="cpu",
        )
        hash_keys, num_entries = native_builder.default(
            key_tensor,
            max_probe,
            load_factor,
        )
        return CSDHashTablePayload(
            keys=hash_keys,
            num_entries=num_entries,
            capacity=hash_keys.numel(),
            max_probe=max_probe,
        )

    key_list = [int(key) for key in keys]
    if not key_list:
        return CSDHashTablePayload(
            keys=[CSD_EMPTY_KEY],
            num_entries=0,
            capacity=1,
            max_probe=max_probe,
        )

    for key in key_list:
        if key < 0:
            raise ValueError(f"CSD keys must be non-negative, got {key}")

    capacity = _next_power_of_two(max(2, math.ceil(len(key_list) / load_factor)))
    while True:
        hash_keys = _try_build_hash_table(key_list, capacity, max_probe)
        if hash_keys is not None:
            return CSDHashTablePayload(
                keys=hash_keys,
                num_entries=len(set(key_list)),
                capacity=capacity,
                max_probe=max_probe,
            )
        capacity *= 2


def build_csd_hash_table(
    keys: Iterable[int],
    device: torch.device | str,
    max_probe: int = CSD_DEFAULT_MAX_PROBE,
    load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
) -> CSDHashTable:
    payload = build_csd_hash_table_payload(
        keys=keys,
        max_probe=max_probe,
        load_factor=load_factor,
    )
    return materialize_csd_hash_table_payload(payload, device)


def _try_build_hash_table(
    keys: List[int],
    capacity: int,
    max_probe: int,
) -> Optional[List[int]]:
    hash_keys = [CSD_EMPTY_KEY] * capacity

    for key in keys:
        slot = _hash64(key) & (capacity - 1)
        inserted = False
        for _ in range(max_probe):
            existing_key = hash_keys[slot]
            if existing_key in (CSD_EMPTY_KEY, key):
                hash_keys[slot] = key
                inserted = True
                break
            slot = (slot + 1) & (capacity - 1)
        if not inserted:
            return None

    return hash_keys


@dataclass
class CSDMetrics:
    lookup_hit_ct: torch.Tensor
    forced_accept_ct: torch.Tensor
    delta_pair_ct: torch.Tensor

    @classmethod
    def allocate(cls, device: torch.device | str) -> "CSDMetrics":
        return cls(
            lookup_hit_ct=torch.zeros((1,), dtype=torch.int64, device=device),
            forced_accept_ct=torch.zeros((1,), dtype=torch.int64, device=device),
            delta_pair_ct=torch.zeros((1,), dtype=torch.int64, device=device),
        )

    def reset(self) -> None:
        self.lookup_hit_ct.zero_()
        self.forced_accept_ct.zero_()
        self.delta_pair_ct.zero_()

    def snapshot(self) -> Dict[str, int]:
        return {
            "csd_lookup_hit_ct": int(self.lookup_hit_ct.item()),
            "csd_forced_accept_ct": int(self.forced_accept_ct.item()),
            "csd_delta_pair_ct": int(self.delta_pair_ct.item()),
        }


@dataclass
class CSDDeltaBuffer:
    pairs: torch.Tensor
    counter: torch.Tensor

    @classmethod
    def allocate(
        cls,
        device: torch.device | str,
        capacity: int = CSD_DEFAULT_DELTA_CAPACITY,
    ) -> "CSDDeltaBuffer":
        return cls(
            pairs=torch.empty((capacity,), dtype=torch.int64, device=device),
            counter=torch.zeros((1,), dtype=torch.int32, device=device),
        )

    @property
    def capacity(self) -> int:
        return self.pairs.shape[0]

    def reset(self) -> None:
        self.counter.zero_()

    def drain_to_counter(self) -> Counter[int]:
        count = min(int(self.counter.item()), self.capacity)
        if count == 0:
            return Counter()
        pairs = self.pairs[:count].detach().cpu().tolist()
        self.reset()
        return Counter(int(pair) for pair in pairs if int(pair) >= 0)

    def drain_to_cpu_tensor(self) -> torch.Tensor:
        count = min(int(self.counter.item()), self.capacity)
        if count == 0:
            return torch.empty((0,), dtype=torch.int64, device="cpu")
        pairs = self.pairs[:count].detach().to(device="cpu", dtype=torch.int64)
        self.reset()
        return pairs


@dataclass
class CSDRuntime:
    enabled: bool
    dynamic_update: bool
    force_accept_disabled: bool
    dynamic_update_ignore_prob_ratio: bool
    table: CSDHashTable
    metrics: CSDMetrics
    delta_buffer: Optional[CSDDeltaBuffer] = None
    table_store: CSDTableStore = field(default_factory=CSDTableStore)
    delta_counts: Counter[int] = field(default_factory=Counter)
    delta_save_path: Optional[str] = None
    online_rebuild_enabled: bool = False
    rebuild_future: Optional[Future] = None
    rebuild_started_ct: int = 0
    rebuild_applied_ct: int = 0
    rebuild_started_at: Optional[float] = None
    rebuild_lifecycle_wall_sec: float = 0.0
    native_table_builder: Optional[Any] = None
    native_table_store_entries: Optional[int] = None
    rebuild_check_interval: int = CSD_DEFAULT_REBUILD_CHECK_INTERVAL
    rebuild_check_countdown: int = 0

    @classmethod
    def from_server_args(
        cls,
        server_args: "ServerArgs",
        device: torch.device | str,
        delta_capacity: int = CSD_DEFAULT_DELTA_CAPACITY,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
        load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
    ) -> "CSDRuntime":
        metrics = CSDMetrics.allocate(device)
        if not server_args.speculative_csd_enabled:
            return cls(
                enabled=False,
                dynamic_update=False,
                force_accept_disabled=False,
                dynamic_update_ignore_prob_ratio=False,
                table=CSDHashTable.empty(device=device, max_probe=max_probe),
                metrics=metrics,
            )

        table_store = CSDTableStore()
        if server_args.speculative_csd_table_path is not None:
            table_store = CSDTableStore.load(server_args.speculative_csd_table_path)

        table = table_store.build_allow_hash_table(
            device=device,
            freq_threshold=server_args.speculative_csd_freq_threshold,
            max_probe=max_probe,
            load_factor=load_factor,
            key_selection_strategy=server_args.speculative_csd_key_selection_strategy,
            score_threshold=server_args.speculative_csd_score_threshold,
        )
        delta_buffer = None
        if server_args.speculative_csd_dynamic_update:
            delta_buffer = CSDDeltaBuffer.allocate(
                device=device,
                capacity=server_args.speculative_csd_delta_capacity,
            )

        native_table_builder = None
        use_native_builder = (
            server_args.speculative_csd_dynamic_update
            and server_args.speculative_csd_key_selection_strategy
            == CSD_KEY_SELECTION_FREQUENCY
            and (
                server_args.speculative_csd_rebuild_top_keep is None
                or server_args.speculative_csd_rebuild_top_keep <= 0
            )
        )
        if use_native_builder:
            try:
                builder_class = torch.classes.sgl_kernel.CSDTableBuilder
            except RuntimeError:
                builder_class = None
            if builder_class is not None:
                initial_keys = torch.tensor(
                    list(table_store.entries), dtype=torch.int64, device="cpu"
                )
                initial_freqs = torch.tensor(
                    [entry.freq for entry in table_store.entries.values()],
                    dtype=torch.int64,
                    device="cpu",
                )
                native_table_builder = builder_class(
                    initial_keys,
                    initial_freqs,
                    server_args.speculative_csd_freq_threshold,
                )

        return cls(
            enabled=True,
            dynamic_update=server_args.speculative_csd_dynamic_update,
            force_accept_disabled=server_args.speculative_csd_force_accept_disabled,
            dynamic_update_ignore_prob_ratio=server_args.speculative_csd_dynamic_update_ignore_prob_ratio,
            table=table,
            metrics=metrics,
            delta_buffer=delta_buffer,
            table_store=table_store,
            delta_save_path=server_args.speculative_csd_delta_save_path,
            online_rebuild_enabled=bool(
                server_args.speculative_csd_dynamic_update
                and server_args.speculative_csd_table_path is not None
            ),
            native_table_builder=native_table_builder,
            native_table_store_entries=(
                len(table_store.entries)
                if native_table_builder is not None
                else None
            ),
        )

    @property
    def has_table(self) -> bool:
        return self.table.has_entries

    def metrics_snapshot(self) -> Dict[str, int | float]:
        snapshot = self.metrics.snapshot()
        snapshot.update(
            {
                "csd_table_num_entries": int(self.table.num_entries),
                "csd_table_capacity": int(self.table.capacity),
                "csd_table_max_probe": int(self.table.max_probe),
                "csd_table_store_entries": int(
                    self.native_table_store_entries
                    if self.native_table_store_entries is not None
                    else len(self.table_store.entries)
                ),
                "csd_rebuild_started_ct": int(self.rebuild_started_ct),
                "csd_rebuild_applied_ct": int(self.rebuild_applied_ct),
                "csd_rebuild_inflight": int(self.rebuild_future is not None),
                "csd_rebuild_lifecycle_wall_sec": float(
                    self.rebuild_lifecycle_wall_sec
                ),
            }
        )
        if self.delta_buffer is not None:
            snapshot["csd_delta_buffer_capacity"] = int(self.delta_buffer.capacity)
        return snapshot

    def flush_delta(self) -> Counter[int]:
        if self.delta_buffer is None:
            return Counter()
        counts = self.delta_buffer.drain_to_counter()
        self.delta_counts.update(counts)
        return counts

    def save_delta(self, path: Optional[str] = None) -> None:
        self.flush_delta()
        save_path = path or self.delta_save_path
        if save_path is None:
            raise ValueError("CSD delta save path is not configured")
        delta_store = CSDTableStore(metadata=dict(self.table_store.metadata))
        delta_store.metadata["content"] = "raw_delta_counts"
        delta_store.merge_counts(self.delta_counts)
        delta_store.save(save_path)

    def save_table(self, path: str) -> None:
        if self.native_table_builder is not None:
            if self.delta_buffer is not None:
                pairs = self.delta_buffer.drain_to_cpu_tensor()
                if pairs.numel() > 0:
                    self.native_table_builder.update(pairs)
            keys, freqs = self.native_table_builder.snapshot()
            self.table_store.entries = {
                int(key): CSDEntry(freq=int(freq), allow=True)
                for key, freq in zip(keys.tolist(), freqs.tolist())
            }
            self.table_store.save(path)
            return
        self.flush_delta()
        self.table_store.merge_counts(self.delta_counts)
        self.delta_counts.clear()
        self.table_store.save(path)

    def should_check_async_rebuild(self) -> bool:
        if not self.online_rebuild_enabled or self.delta_buffer is None:
            return False
        if self.rebuild_future is not None and not self.rebuild_future.done():
            return False
        if self.rebuild_check_interval <= 1:
            return True
        self.rebuild_check_countdown -= 1
        if self.rebuild_check_countdown > 0:
            return False
        self.rebuild_check_countdown = max(1, self.rebuild_check_interval)
        return True

    def maybe_apply_async_rebuild(
        self,
        device: torch.device | str,
    ) -> bool:
        if self.rebuild_future is None or not self.rebuild_future.done():
            return False
        with _csd_profile_range("csd_rebuild_apply:future_result"):
            payload = self.rebuild_future.result()
        self.rebuild_future = None
        with _csd_profile_range(
            "csd_rebuild_apply:materialize_and_swap "
            f"entries={payload.num_entries} capacity={payload.capacity}"
        ):
            self.table = materialize_csd_hash_table_payload(payload, device)
        if payload.store_entries is not None:
            self.native_table_store_entries = payload.store_entries
        self.rebuild_applied_ct += 1
        if self.rebuild_started_at is not None:
            self.rebuild_lifecycle_wall_sec += (
                time.perf_counter() - self.rebuild_started_at
            )
            self.rebuild_started_at = None
        return True

    def maybe_start_async_rebuild(
        self,
        freq_threshold: int,
        rebuild_threshold: int,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
        load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
        top_keep: Optional[float] = None,
        key_selection_strategy: str = CSD_KEY_SELECTION_FREQUENCY,
        score_threshold: float = 0.0,
    ) -> bool:
        if not self.online_rebuild_enabled or self.delta_buffer is None:
            return False
        if rebuild_threshold <= 0:
            return False
        if self.rebuild_future is not None and not self.rebuild_future.done():
            return False
        with _csd_profile_range("csd_rebuild_start:counter_threshold_check"):
            buffered_pair_count = min(
                int(self.delta_buffer.counter.item()), self.delta_buffer.capacity
            )
            effective_rebuild_threshold = min(
                rebuild_threshold, self.delta_buffer.capacity
            )
            if buffered_pair_count < effective_rebuild_threshold:
                return False

        with _csd_profile_range("csd_rebuild_start:flush_delta"):
            if self.native_table_builder is not None:
                rebuild_input = self.delta_buffer.drain_to_cpu_tensor()
            else:
                rebuild_input = self.flush_delta()
        if len(rebuild_input) == 0:
            return False
        if self.native_table_builder is None:
            self.delta_counts.clear()
        with _csd_profile_range("csd_rebuild_start:executor_submit"):
            self.rebuild_started_at = time.perf_counter()
            if self.native_table_builder is not None:
                self.rebuild_future = _CSD_REBUILD_EXECUTOR.submit(
                    build_csd_rebuild_payload_native,
                    self.native_table_builder,
                    rebuild_input,
                    max_probe,
                    load_factor,
                )
            else:
                self.rebuild_future = _CSD_REBUILD_EXECUTOR.submit(
                    build_csd_rebuild_payload_from_counts,
                    self.table_store,
                    rebuild_input,
                    freq_threshold,
                    max_probe,
                    load_factor,
                    top_keep,
                    key_selection_strategy,
                    score_threshold,
                )
            self.rebuild_started_ct += 1
        return True

    def rebuild_table(
        self,
        device: torch.device | str,
        freq_threshold: int,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
        load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
        top_keep: Optional[float] = None,
        key_selection_strategy: str = CSD_KEY_SELECTION_FREQUENCY,
        score_threshold: float = 0.0,
    ) -> None:
        self.flush_delta()
        self.table_store.merge_counts(self.delta_counts)
        self.delta_counts.clear()
        # TODO: Move rebuild off the hot path and swap in a freshly built table at an idle-safe boundary.
        self.table = self.table_store.build_allow_hash_table(
            device=device,
            freq_threshold=freq_threshold,
            max_probe=max_probe,
            load_factor=load_factor,
            top_keep=top_keep,
            key_selection_strategy=key_selection_strategy,
            score_threshold=score_threshold,
        )
