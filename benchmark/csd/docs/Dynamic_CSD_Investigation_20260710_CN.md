# Dynamic CSD 继续排查：每轮 verify 成本被 rebuild 抬高

本页续接 `Dynamic_CSD_Debug_State_CN.md`。原文件当前由 root 用户持有，本轮无法直接写入，因此新增独立记录。

## Full run 的成本分解

对 full code OOD huge run 的 request sidecar 按 `总 wall time / 总 spec_verify_ct` 重新计算：

| dataset | method | token / verify | wall ms / verify | 相对 plain 的每 verify 成本 |
|---|---|---:|---:|---:|
| APPS | plain | 4.116 | 1.205 | baseline |
| APPS | dynamic huge | 4.114 | 1.198 | -0.6% |
| APPS | dynamic 4096 | 4.944 | 1.379 | +14.4% |
| TACO | plain | 4.077 | 1.170 | baseline |
| TACO | dynamic huge | 4.055 | 1.159 | -0.9% |
| TACO | dynamic 4096 | 4.691 | 1.289 | +10.2% |

APPS 的 token / verify 提升 20.1%，等价于单位 token 所需 verify 数下降约 16.7%；但每轮 verify 的 wall 成本上涨 14.4%，两者基本抵消。TACO 也有相同模式。因此当前问题不是 accept 指标虚高，而是触发真实 rebuild 后主循环每轮变贵。

固定 `running_reqs=24` 的 decode metrics 也显示 APPS dynamic 的 steady-state decode 吞吐约为 3614 tok/s，plain 约为 3422 tok/s，只提升 5.6%。因此差距不只是请求完成长尾造成的统计口径问题。

## 当前最强根因：后台 Python rebuild 抢占 GIL

`build_csd_rebuild_payload_from_counts()` 虽然提交给 `ThreadPoolExecutor`，但主要工作是 Python `Counter`/`dict` 合并、遍历全部 `table_store.entries`、构造和排序 Python list，以及用 Python 循环构建 hash table。这些操作是 GIL-bound，会和 scheduler 主线程竞争，并不是真正与 Python 调度热路径并行。

APPS128 counter run 在 threshold=4096 时记录了 108637 个 delta pair，即约 26 个 threshold batch；同时 `table_store` 已有约 105 万条历史 entry。当前每次 rebuild 都会使 filtered-key cache 失效并扫描全历史表，而不是只处理约 4096 个增量，所以其复杂度接近：

```text
rebuild 次数 * 全历史 entry 数
约 26 * 1,050,000（仅 128 个请求）
```

full APPS 有约 4786 万 output token，rebuild 次数和累计全表扫描会继续放大。

阈值 64 的单请求 trace 也支持 GIL stall 假设：3 次 rebuild 的显式 `materialize_and_swap` 合计仅约 27.6 ms，但 `post_verify_mamba_update` 相比 huge 对照多约 389 ms，并出现单次 72.9 ms 停顿。额外时间落在与 rebuild 无关的主线程 range 内，说明显式 apply 不是主要成本，后台线程抢占 GIL 更符合 trace。

## 新增验证 counter

`CSDRuntime.metrics_snapshot()` 现增加：

- `csd_rebuild_started_ct`
- `csd_rebuild_applied_ct`
- `csd_rebuild_inflight`
- `csd_rebuild_lifecycle_wall_sec`

下一次 APPS/TACO 4096 full run 可以直接得到真实 rebuild 次数和累计生命周期，不再用 `delta_pair_ct / threshold` 间接估算。

## 下一步优化验证优先级

1. 先做禁用 online rebuild、离线预合并相同 dynamic table 的对照。如果 accept 保持而吞吐恢复，可直接确认收益损失来自 rebuild，而非大表 lookup。
2. 将 rebuild 计算移出 scheduler 进程，或者把 merge/filter/hash build 改成释放 GIL 的原生实现。单纯增加 thread executor 数量不会解决 GIL 竞争。
3. 避免每 4096 delta 对 105 万历史 entry 做全量过滤；维护增量频次和候选 key，只在 key 跨过 threshold 时更新 membership。
4. rebuild 隔离后，再区分 static/dynamic-added 命中来源，评估新增 key 的纯收益。

## 增量过滤修复与实测

已对 frequency 策略实现增量 key membership：首次构建时记录达到 `freq_threshold` 的 key 集合，后续 `merge_counts()` 只把本批次中新跨过阈值的 key 加入集合。这样 rebuild 不再反复扫描全部历史 entries。

使用实际 1052300-entry table、每轮 4096 个增量的 CPU 微基准：

| 阶段 | 修复前 | 修复后 |
|---|---:|---:|
| frequency 全表过滤 | 约 44.7 ms | 约 0.6 ms |
| merge 4096 增量 | 约 3.6 ms | 约 3.6 ms |
| 构建约 3.4 万 key hash payload | 约 25 ms | 约 25 ms |

threshold=64、同一 APPS prompt、1098 output token 的 GPU profile 对比：

| 指标 | 修复前 dynamic | 修复后 dynamic | huge 对照 |
|---|---:|---:|---:|
| output tok/s（含 profile） | 36.91 | 37.05 | 37.03 |
| `post_verify_mamba_update` 累计 | 576.7 ms | 283.1 ms | 184.9 ms |
| 上述 range 最大单次停顿 | 72.9 ms | 19.9 ms | 0.9 ms |
| `materialize_and_swap` 累计 | 27.6 ms | 27.4 ms | 0 ms |

修复消除了 dynamic 相对 huge 约 75% 的非 apply 异常停顿；显式 apply 成本基本不变。说明 apply 确实会短暂停顿，但后台 Python 全表过滤抢占 GIL 是 threshold=64 实验的主要额外成本。

## 对 full-run 14% 每 verify 成本差异的修正解释

上述修复不能单独解释 full APPS 推导出的 14.4% `wall / verify` 增长。full run 中不同方法产生了不同输出，`wall / verify` 同时混入了上下文长度和连续批处理形态变化。

按每个请求的 `spec_verify_ct` 加权，并用 `prompt_tokens + completion_tokens / 2` 近似该请求 verify 时的平均上下文：

| dataset | plain | dynamic huge | dynamic 4096 | dynamic vs plain |
|---|---:|---:|---:|---:|
| APPS | 17.6k | 17.2k | 24.1k | +37.0% |
| TACO | 15.0k | 14.4k | 18.6k | +24.0% |

dynamic APPS 的平均输出也从 20809 增至 23931，max-new hit 从 62 增至 286。更高 accept 让请求更快前进，但这些 run 同时生成了更多长输出，并在更长 KV context 上执行 verify。Qwen3.5 的 attention 层 decode 成本会随上下文增长，这比大表 lookup 更能解释 full-run 中单位 verify 变贵。

因此下一项严格对照应固定每个请求的生成长度并忽略 EOS，例如同一批 APPS prompt 全部生成固定 16k token。这样 plain/dynamic 的 context-length 轨迹一致，才能测出 accept 提升扣除 rebuild 后的纯系统加速。

## 固定长度 `ignore_eos` 对照

profile 脚本已增加 `--ignore-eos`。使用 4 个相同 APPS prompts，每个方法固定生成 4096 token，总 token 均为 16384，batch 上限为 4：

| method | threshold | tok/s（含 profile） | verify range 次数 |
|---|---:|---:|---:|
| plain | 无 rebuild | 123.63 | 2244 |
| dynamic huge | 基本不 rebuild | 123.66 | 2244 |
| dynamic incremental | 64 | 122.69 | 2216 |

plain 与 dynamic huge 只差 0.02%，进一步排除大表 lookup 和 dynamic 常驻热路径是显著成本。threshold=64 的 dynamic verify 次数只下降约 1.25%，但触发约 55 次实际 apply；trace 中 `materialize_and_swap` 合计约 512 ms，且残余 Python hash payload 构建仍造成 GIL stall，因此最终比 huge 慢约 0.78%。

threshold=64 是刻意放大 rebuild 的设置。正常 threshold=4096 的触发频率约低 64 倍，因此增量过滤修复后，4096 场景的 rebuild 残余成本预计很小。full APPS 中 accept 大涨但加速有限的主抵消项，应优先归因于生成长度分布改变和更长 context 上的 attention decode，而不是 hash-table lookup。

## 直接 CPU/GIL profile 与 C++ hash build

新增工具：

- `benchmark/csd/eval/profile_csd_rebuild_cpu.py`
- `benchmark/csd/eval/run_profile_csd_rebuild_gil.sh`

工具通过 `py-spy record --gil --threads --format chrometrace` 运行生产函数 `build_csd_rebuild_payload_from_counts()`。生成的 Chrome trace 可以直接在 Perfetto 打开。因为使用 `--gil`，`csd-rebuild_0` 线程上显示的 stack 正是后台线程持有 GIL、scheduler 无法执行 Python 的时间。

实际 1052300-entry table、200 次 rebuild 的聚焦 profile：

| Python range | inclusive total | per rebuild |
|---|---:|---:|
| `rebuild_once` | 6435.3 ms | 32.18 ms |
| `merge_counts` | 250.8 ms | 1.25 ms |
| `filtered_keys` | 144.0 ms | 0.72 ms |
| `build_csd_hash_table_payload` | 6010.9 ms | 30.05 ms |
| `_try_build_hash_table` | 4658.6 ms | 23.29 ms |
| `_hash64` | 3043.6 ms | 15.22 ms |

以上为嵌套 inclusive 时间，不能相加。trace 位于 `benchmark/csd/runs/csd_rebuild_cpu_profile/csd_rebuild_gil.chrometrace.json`。

第一阶段已新增 C++ CPU Torch op `csd_build_hash_table_cpu`，只迁移 hash payload build，保留 Python fallback。实际 34305 keys、capacity=131072 时：

- Python hash build：约 28.4 ms
- C++ CPU hash build：约 0.257 ms
- 加速约 110 倍
- C++ 与 Python table 逐槽一致，随机 0/1/17/1000/10000 keys 和重复 key 已验证

加载 C++ op 后，1000 次 rebuild 的 GIL profile：

| Python range | per rebuild |
|---|---:|
| `rebuild_once` | 5.32 ms |
| `merge_counts` | 1.30 ms |
| `filtered_keys` | 0.58 ms |
| `build_csd_hash_table_payload` wrapper | 3.36 ms |
| `_try_build_hash_table` | 0 ms |
| `_hash64` | 0 ms |

完整 rebuild 的 Python/GIL 时间约降低 6 倍。C++ 对照 trace 位于 `benchmark/csd/runs/csd_rebuild_cpu_profile_cpp_v2/csd_rebuild_gil.chrometrace.json`。

第二阶段若要基本消除 GIL，应实现 stateful C++ builder，让 C++ 持有 frequency map 和 threshold membership，并直接接收 drain 后的 CPU pair tensor。CUDA 继续负责 delta append 与 decode lookup；rebuild 的低频控制面、变长 capacity retry 和 open-addressing 构建保留在 C++ CPU 更合适。
