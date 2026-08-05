from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

CSD_EMPTY_KEY = -1
CSD_MAX_TOKEN_ID = (1 << 31) - 1
CSD_DEFAULT_LOAD_FACTOR = 0.5
CSD_DEFAULT_MAX_PROBE = 16
_UINT64_MASK = (1 << 64) - 1
_REBUILD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="csd-rebuild")


def pack_csd_pair(draft_token: int, residual_token: int) -> int:
    if not 0 <= draft_token <= CSD_MAX_TOKEN_ID:
        raise ValueError(f"draft_token is out of range: {draft_token}")
    if not 0 <= residual_token <= CSD_MAX_TOKEN_ID:
        raise ValueError(f"residual_token is out of range: {residual_token}")
    return (draft_token << 32) | residual_token


def unpack_csd_pair(key: int) -> tuple[int, int]:
    if key < 0:
        raise ValueError(f"CSD key must be non-negative, got {key}")
    return key >> 32, key & 0xFFFFFFFF


def _hash64(key: int) -> int:
    value = key & _UINT64_MASK
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def csd_kernel_kwargs(
    runtime: Optional["CSDRuntime"], *, include_entropy: bool
) -> dict[str, Any]:
    """Build the common CUDA verifier arguments for every CSD backend.

    MTP/EAGLE and DSpark use different proposal layouts, but table lookup,
    dynamic pair collection, probability gating, and metrics share the same
    runtime state.  Keeping this adapter here prevents the two backends from
    silently drifting in their CSD semantics.
    """
    if runtime is None or not runtime.enabled:
        return {}
    delta = runtime.delta_buffer
    kwargs: dict[str, Any] = {
        "csd_table_keys": runtime.table.keys,
        "csd_delta_pairs": None if delta is None else delta.pairs,
        "csd_delta_counter": None if delta is None else delta.counter,
        "csd_lookup_hit_ct": runtime.metrics.lookup_hit_ct,
        "csd_forced_accept_ct": runtime.metrics.forced_accept_ct,
        "csd_delta_pair_ct": runtime.metrics.delta_pair_ct,
        "csd_table_capacity": runtime.table.capacity,
        "csd_table_max_probe": runtime.table.max_probe,
        "csd_delta_capacity": 0 if delta is None else delta.capacity,
        "csd_enabled": runtime.has_table,
        "csd_dynamic_update": runtime.dynamic_update and delta is not None,
        "csd_dynamic_update_ignore_prob_ratio": (
            runtime.dynamic_update_ignore_prob_ratio and delta is not None
        ),
        "csd_force_accept_disabled": runtime.force_accept_disabled,
        "csd_logit_margin": math.log(runtime.prob_ratio),
    }
    if include_entropy:
        kwargs["csd_force_accept_entropy_threshold"] = runtime.entropy_threshold
        kwargs["csd_force_accept_entropy_min_threshold"] = (
            runtime.entropy_min_threshold
        )
    return kwargs


@dataclass
class CSDTableStore:
    counts: Counter[int] = field(default_factory=Counter)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "CSDTableStore":
        table_path = Path(path)
        with table_path.open(encoding="utf-8") as file:
            if table_path.suffix == ".jsonl":
                payload: Any = [json.loads(line) for line in file if line.strip()]
            else:
                payload = json.load(file)
        store = cls()
        if isinstance(payload, dict):
            store.metadata.update(payload.get("metadata", {}))
            records = payload.get("entries", [])
        else:
            records = payload
        for record in records:
            if record.get("type") == "metadata":
                store.metadata.update(record.get("metadata", {}))
                continue
            key = record.get("key")
            if key is None:
                key = pack_csd_pair(
                    int(record.get("lhs_token", record.get("draft_token"))),
                    int(record.get("rhs_token", record.get("residual_token"))),
                )
            store.counts[int(key)] += int(record.get("freq", 1))
        return store

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "key": int(key),
                "lhs_token": unpack_csd_pair(key)[0],
                "rhs_token": unpack_csd_pair(key)[1],
                "freq": int(freq),
                "allow": True,
            }
            for key, freq in sorted(self.counts.items())
        ]
        if output.suffix == ".jsonl":
            with output.open("w", encoding="utf-8") as file:
                if self.metadata:
                    file.write(
                        json.dumps({"type": "metadata", "metadata": self.metadata})
                        + "\n"
                    )
                for record in records:
                    file.write(json.dumps(record) + "\n")
        else:
            with output.open("w", encoding="utf-8") as file:
                json.dump(
                    {"metadata": self.metadata, "entries": records},
                    file,
                    ensure_ascii=False,
                )


@dataclass
class CSDHashTable:
    keys: torch.Tensor
    num_entries: int
    max_probe: int

    @property
    def capacity(self) -> int:
        return int(self.keys.numel())

    @property
    def has_entries(self) -> bool:
        return self.num_entries > 0

    @classmethod
    def empty(cls, device, max_probe: int = CSD_DEFAULT_MAX_PROBE) -> "CSDHashTable":
        return cls(
            keys=torch.full((1,), CSD_EMPTY_KEY, dtype=torch.int64, device=device),
            num_entries=0,
            max_probe=max_probe,
        )


def _build_hash_table_cpu(
    keys: list[int],
    *,
    max_probe: int = CSD_DEFAULT_MAX_PROBE,
    load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
) -> tuple[torch.Tensor, int]:
    key_tensor = torch.tensor(keys, dtype=torch.int64, device="cpu")
    try:
        op = torch.ops.sgl_kernel.csd_build_hash_table_cpu
        return op(key_tensor, max_probe, load_factor)
    except (AttributeError, RuntimeError):
        pass
    if not keys:
        return torch.full((1,), CSD_EMPTY_KEY, dtype=torch.int64), 0
    capacity = _next_power_of_two(max(2, int(len(keys) / load_factor + 0.999999)))
    while True:
        table = [CSD_EMPTY_KEY] * capacity
        inserted = 0
        failed = False
        for key in keys:
            slot = _hash64(key) & (capacity - 1)
            for _ in range(max_probe):
                if table[slot] in (CSD_EMPTY_KEY, key):
                    if table[slot] == CSD_EMPTY_KEY:
                        inserted += 1
                    table[slot] = key
                    break
                slot = (slot + 1) & (capacity - 1)
            else:
                failed = True
                break
        if not failed:
            return torch.tensor(table, dtype=torch.int64), inserted
        capacity *= 2


def _materialize_table(
    table_cpu: torch.Tensor, num_entries: int, max_probe: int, device
) -> CSDHashTable:
    return CSDHashTable(
        keys=table_cpu.to(device=device, non_blocking=True),
        num_entries=int(num_entries),
        max_probe=int(max_probe),
    )


@dataclass
class CSDMetrics:
    lookup_hit_ct: torch.Tensor
    forced_accept_ct: torch.Tensor
    delta_pair_ct: torch.Tensor

    @classmethod
    def allocate(cls, device) -> "CSDMetrics":
        def counter() -> torch.Tensor:
            return torch.zeros((1,), dtype=torch.int64, device=device)

        return cls(counter(), counter(), counter())

    def snapshot(self) -> dict[str, int]:
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
    def allocate(cls, device, capacity: int) -> "CSDDeltaBuffer":
        return cls(
            pairs=torch.empty((capacity,), dtype=torch.int64, device=device),
            counter=torch.zeros((1,), dtype=torch.int32, device=device),
        )

    @property
    def capacity(self) -> int:
        return int(self.pairs.numel())

    def drain_cpu(self) -> torch.Tensor:
        count = min(int(self.counter.item()), self.capacity)
        if count == 0:
            return torch.empty((0,), dtype=torch.int64)
        pairs = self.pairs[:count].detach().to(device="cpu", dtype=torch.int64)
        self.counter.zero_()
        return pairs


@dataclass
class _RebuildPayload:
    table_cpu: torch.Tensor
    num_entries: int
    store_entries: int


def _python_rebuild(
    store: CSDTableStore,
    pairs: torch.Tensor,
    freq_threshold: int,
    max_probe: int,
) -> _RebuildPayload:
    store.counts.update(int(value) for value in pairs.tolist() if int(value) >= 0)
    active = [key for key, freq in store.counts.items() if freq >= freq_threshold]
    table, num_entries = _build_hash_table_cpu(active, max_probe=max_probe)
    return _RebuildPayload(table, num_entries, len(store.counts))


def _native_rebuild(builder, pairs: torch.Tensor, max_probe: int) -> _RebuildPayload:
    table, num_entries, store_entries = builder.update_and_build(
        pairs, max_probe, CSD_DEFAULT_LOAD_FACTOR
    )
    return _RebuildPayload(table, num_entries, store_entries)


@dataclass
class CSDRuntime:
    enabled: bool
    dynamic_update: bool
    dynamic_update_ignore_prob_ratio: bool
    force_accept_disabled: bool
    prob_ratio: float
    entropy_threshold: float
    entropy_min_threshold: float
    freq_threshold: int
    rebuild_threshold: int
    table: CSDHashTable
    metrics: CSDMetrics
    table_store: CSDTableStore = field(default_factory=CSDTableStore)
    delta_buffer: Optional[CSDDeltaBuffer] = None
    delta_save_path: Optional[str] = None
    native_builder: Optional[Any] = None
    rebuild_future: Optional[Future] = None
    rebuild_started_ct: int = 0
    rebuild_applied_ct: int = 0
    rebuild_started_at: Optional[float] = None
    rebuild_wall_sec: float = 0.0
    rebuild_check_interval: int = 16
    steps_since_rebuild_check: int = 0
    aggregate_across_dp: bool = False
    dp_aggregate_started_ct: int = 0
    dp_aggregate_pair_ct: int = 0

    @classmethod
    def from_server_args(cls, server_args: "ServerArgs", device) -> "CSDRuntime":
        metrics = CSDMetrics.allocate(device)
        if not server_args.speculative_csd_enabled:
            return cls(
                enabled=False,
                dynamic_update=False,
                dynamic_update_ignore_prob_ratio=False,
                force_accept_disabled=False,
                prob_ratio=1.0,
                entropy_threshold=-1.0,
                entropy_min_threshold=-1.0,
                freq_threshold=1,
                rebuild_threshold=0,
                table=CSDHashTable.empty(device),
                metrics=metrics,
            )
        store = (
            CSDTableStore.load(server_args.speculative_csd_table_path)
            if server_args.speculative_csd_table_path
            else CSDTableStore()
        )
        active = [
            key
            for key, freq in store.counts.items()
            if freq >= server_args.speculative_csd_freq_threshold
        ]
        table_cpu, num_entries = _build_hash_table_cpu(active)
        table = _materialize_table(
            table_cpu, num_entries, CSD_DEFAULT_MAX_PROBE, device
        )
        native_builder = None
        if server_args.speculative_csd_dynamic_update:
            try:
                builder_type = torch.classes.sgl_kernel.CSDTableBuilder
                keys = torch.tensor(list(store.counts), dtype=torch.int64)
                freqs = torch.tensor(list(store.counts.values()), dtype=torch.int64)
                native_builder = builder_type(
                    keys, freqs, server_args.speculative_csd_freq_threshold
                )
            except (AttributeError, RuntimeError):
                native_builder = None
        return cls(
            enabled=True,
            dynamic_update=server_args.speculative_csd_dynamic_update,
            dynamic_update_ignore_prob_ratio=(
                server_args.speculative_csd_dynamic_update_ignore_prob_ratio
            ),
            force_accept_disabled=server_args.speculative_csd_force_accept_disabled,
            prob_ratio=float(server_args.speculative_csd_prob_ratio),
            entropy_threshold=float(
                server_args.speculative_csd_force_accept_entropy_threshold
            ),
            entropy_min_threshold=float(
                server_args.speculative_csd_force_accept_entropy_min_threshold
            ),
            freq_threshold=int(server_args.speculative_csd_freq_threshold),
            rebuild_threshold=int(server_args.speculative_csd_rebuild_threshold),
            table=table,
            metrics=metrics,
            table_store=store,
            delta_buffer=(
                CSDDeltaBuffer.allocate(
                    device, int(server_args.speculative_csd_delta_capacity)
                )
                if server_args.speculative_csd_dynamic_update
                else None
            ),
            delta_save_path=server_args.speculative_csd_delta_save_path,
            native_builder=native_builder,
            aggregate_across_dp=(
                bool(server_args.enable_dp_attention)
                and int(server_args.dp_size) > 1
            ),
        )

    @property
    def has_table(self) -> bool:
        return self.table.has_entries

    def metrics_snapshot(self) -> dict[str, int | float]:
        result: dict[str, int | float] = self.metrics.snapshot()
        result.update(
            {
                "csd_table_num_entries": self.table.num_entries,
                "csd_table_capacity": self.table.capacity,
                "csd_table_store_entries": len(self.table_store.counts),
                "csd_rebuild_started_ct": self.rebuild_started_ct,
                "csd_rebuild_applied_ct": self.rebuild_applied_ct,
                "csd_rebuild_inflight": int(self.rebuild_future is not None),
                "csd_rebuild_wall_sec": self.rebuild_wall_sec,
                "csd_dp_aggregate_enabled": int(self.aggregate_across_dp),
                "csd_dp_aggregate_started_ct": self.dp_aggregate_started_ct,
                "csd_dp_aggregate_pair_ct": self.dp_aggregate_pair_ct,
                "csd_entropy_min_threshold": self.entropy_min_threshold,
                "csd_entropy_max_threshold": self.entropy_threshold,
            }
        )
        return result

    def _dp_rebuild_input(self) -> Optional[torch.Tensor]:
        """Collect one copy of every DP lane's buffered pairs on every TP rank.

        DP-attention embeds its DP lanes in the model TP world.  All model ranks
        execute verification in lockstep, so the full TP group is the safe
        synchronization domain.  When an attention lane itself has TP > 1, its
        ranks observe the same accept/reject pairs; only attention-TP rank zero
        contributes, avoiding duplicate frequency counts.  Every rank receives
        the identical concatenated input and therefore rebuilds the same table.
        """
        from sglang.srt.distributed import get_attn_tp_group, get_tp_group

        delta = self.delta_buffer
        assert delta is not None
        tp_group = get_tp_group()
        attn_tp_group = get_attn_tp_group()
        contributes = attn_tp_group.rank_in_group == 0
        local_count = min(int(delta.counter.item()), delta.capacity)

        # Gather counts and rebuild readiness together.  A rank with an
        # in-flight CPU rebuild still joins this collective, preventing peers
        # from entering an unmatched all-gather and deadlocking.
        state = torch.tensor(
            [local_count if contributes else 0, int(self.rebuild_future is None)],
            dtype=torch.int64,
            device=delta.pairs.device,
        )
        states = tp_group.all_gather(state, dim=0).view(tp_group.world_size, 2)
        states_cpu = states.cpu()
        counts = [int(value) for value in states_cpu[:, 0].tolist()]
        all_ready = all(int(value) == 1 for value in states_cpu[:, 1].tolist())
        total_count = sum(counts)
        if not all_ready or total_count < self.rebuild_threshold:
            return None

        max_count = max(counts, default=0)
        if max_count == 0:
            return None
        padded = torch.full(
            (max_count,),
            CSD_EMPTY_KEY,
            dtype=torch.int64,
            device=delta.pairs.device,
        )
        if contributes and local_count:
            padded[:local_count].copy_(delta.pairs[:local_count])
        gathered = tp_group.all_gather(padded, dim=0).view(
            tp_group.world_size, max_count
        )
        pieces = [gathered[rank, :count] for rank, count in enumerate(counts) if count]
        pairs = torch.cat(pieces).to(device="cpu", dtype=torch.int64)

        # Non-leader attention-TP ranks also collected duplicate local deltas;
        # clear every rank only after the collective has consumed leader data.
        delta.counter.zero_()
        self.dp_aggregate_started_ct += 1
        self.dp_aggregate_pair_ct += int(pairs.numel())
        return pairs

    def maybe_apply_async_rebuild(self, device) -> bool:
        if self.rebuild_future is None or not self.rebuild_future.done():
            return False
        payload = self.rebuild_future.result()
        self.rebuild_future = None
        self.table = _materialize_table(
            payload.table_cpu, payload.num_entries, CSD_DEFAULT_MAX_PROBE, device
        )
        self.rebuild_applied_ct += 1
        if self.rebuild_started_at is not None:
            self.rebuild_wall_sec += time.perf_counter() - self.rebuild_started_at
            self.rebuild_started_at = None
        return True

    def maybe_start_async_rebuild(self) -> bool:
        delta = self.delta_buffer
        if not self.dynamic_update or delta is None or self.rebuild_threshold <= 0:
            return False
        self.steps_since_rebuild_check += 1
        if self.steps_since_rebuild_check < self.rebuild_check_interval:
            return False
        self.steps_since_rebuild_check = 0
        if self.aggregate_across_dp:
            pairs = self._dp_rebuild_input()
            if pairs is None:
                return False
        else:
            if self.rebuild_future is not None:
                return False
            buffered = min(int(delta.counter.item()), delta.capacity)
            if buffered < min(self.rebuild_threshold, delta.capacity):
                return False
            pairs = delta.drain_cpu()
        if pairs.numel() == 0:
            return False
        self.rebuild_started_at = time.perf_counter()
        if self.native_builder is not None:
            self.rebuild_future = _REBUILD_EXECUTOR.submit(
                _native_rebuild, self.native_builder, pairs, CSD_DEFAULT_MAX_PROBE
            )
        else:
            self.rebuild_future = _REBUILD_EXECUTOR.submit(
                _python_rebuild,
                self.table_store,
                pairs,
                self.freq_threshold,
                CSD_DEFAULT_MAX_PROBE,
            )
        self.rebuild_started_ct += 1
        return True

    def save_table(self, path: Optional[str] = None) -> None:
        output = path or self.delta_save_path
        if output is None:
            raise ValueError("No CSD table output path is configured.")
        if self.rebuild_future is not None:
            payload = self.rebuild_future.result()
            self.rebuild_future = None
            self.rebuild_applied_ct += 1
            if self.native_builder is None:
                self.table_store.metadata["active_entries"] = payload.num_entries
        if self.delta_buffer is not None:
            pairs = self.delta_buffer.drain_cpu()
            if self.native_builder is not None:
                if pairs.numel():
                    self.native_builder.update(pairs)
                keys, freqs = self.native_builder.snapshot()
                self.table_store.counts = Counter(
                    {int(k): int(v) for k, v in zip(keys.tolist(), freqs.tolist())}
                )
            else:
                self.table_store.counts.update(
                    int(value) for value in pairs.tolist() if int(value) >= 0
                )
        self.table_store.metadata.update(
            {
                "content": "csd_pair_frequency",
                "freq_threshold": self.freq_threshold,
                "prob_ratio": self.prob_ratio,
                "entropy_threshold": self.entropy_threshold,
                "entropy_min_threshold": self.entropy_min_threshold,
                "pid": os.getpid(),
            }
        )
        self.table_store.save(output)
