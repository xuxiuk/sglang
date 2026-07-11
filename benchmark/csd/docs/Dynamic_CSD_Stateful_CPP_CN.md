# Dynamic CSD Stateful C++ Rebuild 技术说明

## 1. 背景与目标

Dynamic CSD 在 speculative decode 过程中收集新的 token replacement pair。累计到 rebuild threshold 后，系统将增量频次合并到历史 CSD table，重新筛选达到频次阈值的 key，构建新的 open-addressing hash table，并替换 GPU 上供 speculative sampling kernel 查询的 table。

此前 APPS/TACO 观察到 dynamic accept length 明显提高，但吞吐提升有限。排查确认有两个相互独立的问题：

1. online rebuild 的 Python/GIL 开销。后台 `ThreadPoolExecutor` 执行的是纯 Python dict/set/list/hash 循环，仍会持有 GIL 并阻塞 scheduler Python 主线程。
2. full run 的 workload 发生改变。dynamic 生成更长输出、更多 max-new hit，使剩余 verify 运行在更长 context 上。这部分不是 lookup/rebuild 优化能消除的。

本轮实现针对第一个问题，目标是：

- 在线 rebuild 的频次状态和 hash build 不再由 Python 执行。
- 保留现有 CUDA delta append 与 CUDA lookup kernel。
- 保持 Python fallback，未编译新 kernel 或使用非 frequency 策略时行为不变。
- 保持 table snapshot/save 能力。
- 用直接 GIL profile、逐轮一致性、真实 TP4 profile 和 full coding benchmark 验证。

## 2. 原始数据路径

旧路径如下：

```text
speculative sampling CUDA kernel
    |
    | atomic append packed int64 pair
    v
GPU CSDDeltaBuffer[pairs, counter]
    |
    | counter.item() + pairs.cpu().tolist()
    v
Python Counter
    |
    | merge into ~1.05M-entry Python dict
    v
Python CSDTableStore
    |
    | scan/filter/sort or membership materialization
    v
Python list of ~34K active keys
    |
    | Python splitmix64 + open addressing
    v
Python list hash payload
    |
    | torch.tensor(..., device=cuda)
    v
GPU hash table swap
```

`ThreadPoolExecutor` 只能把 rebuild 放到另一个 OS thread，不能使纯 Python 代码绕过 GIL。worker 持有 GIL 时，scheduler thread 即使 runnable 也不能执行 Python。

## 3. 如何直接 profile GIL

### 3.1 为什么 PyTorch operator trace 看不到扫表

扫表和 Python hash build 使用：

- `dict.items()`
- Python `for` 循环
- Python integer/hash/compare
- list/set 构造
- dataclass attribute access

它们不是 ATen operator，也不启动 CUDA kernel。因此 CUDA/ATen operator 总时间很小并不能说明 CPU 开销小。

当前 torch profiler 从 scheduler thread 启动，`ThreadPoolExecutor` 新线程上的 `record_function` 也没有完整进入同一个 profiler context。结果是在 Perfetto 中看到 scheduler 某个 range 被异常拉长，却看不到对应后台 Python stack。

### 3.2 py-spy GIL Chrome trace

新增：

- `benchmark/csd/eval/profile_csd_rebuild_cpu.py`
- `benchmark/csd/eval/run_profile_csd_rebuild_gil.sh`

核心命令：

```bash
py-spy record \
  --gil \
  --threads \
  --rate 100 \
  --format chrometrace \
  --output csd_rebuild_gil.chrometrace.json \
  -- python benchmark/csd/eval/profile_csd_rebuild_cpu.py ...
```

`--gil` 只记录持有 GIL 的 Python thread。生成文件可直接在 `https://ui.perfetto.dev/` 打开。展开 `csd-rebuild_0` 即可看到后台线程阻止 scheduler 执行 Python 的 stack。

Python 版本 trace：

```text
benchmark/csd/runs/csd_rebuild_cpu_profile/csd_rebuild_gil.chrometrace.json
```

关键 inclusive 时间：

| range | 200 次 total | 每次 rebuild |
|---|---:|---:|
| `rebuild_once` | 6435.3 ms | 32.18 ms |
| `merge_counts` | 250.8 ms | 1.25 ms |
| `filtered_keys` | 144.0 ms | 0.72 ms |
| `build_csd_hash_table_payload` | 6010.9 ms | 30.05 ms |
| `_try_build_hash_table` | 4658.6 ms | 23.29 ms |
| `_hash64` | 3043.6 ms | 15.22 ms |

这些是嵌套 inclusive 时间，不能相加。主线程 0.5 ms heartbeat 的 p99 gap 为 6.7 ms，最大 gap 为 8.0 ms，直接显示了 GIL 抢占。

## 4. 历史中间阶段一：Python 增量 membership（当前 fast path 已不使用）

> 本节记录优化演进过程，不描述当前最终运行路径。当前 frequency/no-top-keep fast path 以第 6 节 Stateful C++ builder 为准。

最初每次 merge 都让 filtered cache 失效，随后重新扫描约 105 万历史 entry。frequency threshold 只会随计数增加而从 false 变为 true，因此无需重复全表扫描。

Python 侧增加：

```python
_frequency_key_sets: Dict[int, set[int]]
```

merge 时仅检查本批发生变化的 key：

```python
old_freq = entry.freq
entry.freq += count
new_freq = entry.freq
if old_freq < threshold <= new_freq:
    active_keys.add(key)
```

实际 table 微基准：

| 阶段 | 原始实现 | 增量实现 |
|---|---:|---:|
| 过滤约 105 万 entries | 44.7 ms | 不再执行 |
| 增量 membership | 无 | 约 0.6 ms |
| Python hash payload build | 约 25-30 ms | 约 25-30 ms |

该阶段消除了全表扫描，但 Python hash build 仍是主要 GIL 热点。

## 5. 历史中间阶段二：Stateless C++ CPU hash build（当前 fast path 已不使用）

> 本节记录只把 hash build 移入 C++ 的中间版本。下文所述 Python `Counter` merge 是这个 stateless 版本的限制，已由第 6 节 Stateful C++ builder 消除。

新增 `sgl-kernel/csrc/speculative/csd_rebuild.cpp`，注册 Torch CPU operator：

```text
sgl_kernel::csd_build_hash_table_cpu(
    Tensor keys,
    int max_probe,
    float load_factor
) -> (Tensor table, int num_entries)
```

C++ 实现与 Python reference 使用相同算法：

1. splitmix64 hash。
2. capacity 取满足 load factor 的下一个 2 次幂。
3. linear probing，最大 probe 次数由 `max_probe` 指定。
4. 某个 key 无法在 probe 限制内插入时 capacity 翻倍并重试。
5. `-1` 为 empty sentinel。
6. 重复 key 不重复计数。

实际 34305 active keys、capacity=131072：

| 实现 | hash build |
|---|---:|
| Python | 约 28.4 ms |
| C++ CPU | 约 0.257 ms |

C++ 输出与 Python reference 逐槽一致，纯 hash build 加速约 110 倍。完整 rebuild 的 GIL 时间从约 32.18 ms 降到约 5.32 ms。

但 stateless 版本仍需要 Python 完成：

- `Counter` merge。
- threshold set 更新。
- active set 转 list。
- list 转 CPU Tensor。

因此还没有完全消除在线 GIL gap。

## 6. 最终实现：Stateful C++ CPU builder

**当前实验运行的是本节的最终实现。它不只是重建 hash table；历史频次状态、delta merge、threshold membership 和 hash build 均由 C++ 完成。第 4、5 节仅用于说明演进过程。**

### 6.1 C++ 状态

注册 TorchBind class：

```text
torch.classes.sgl_kernel.CSDTableBuilder
```

内部持有：

```cpp
int64_t freq_threshold_;
std::unordered_map<int64_t, int64_t> counts_;
std::unordered_set<int64_t> active_keys_;
```

构造函数一次性接收：

- 全量 packed pair keys CPU int64 Tensor。
- 对应 frequency CPU int64 Tensor。
- frequency threshold。

初始化只发生在 server/model 启动期间，不在 decode 热路径。

### 6.2 在线 update/build

核心接口：

```text
update_and_build(
    Tensor delta_pairs_cpu,
    int max_probe,
    float load_factor
) -> (
    Tensor hash_table_cpu,
    int num_hash_entries,
    int num_store_entries
)
```

每个 delta pair：

```cpp
int64_t& count = counts_[key];
++count;
if (count == freq_threshold_) {
    active_keys_.insert(key);
}
```

输入是尚未在 Python 聚合的 CPU `int64` delta-pair Tensor。C++ 对每个 pair 执行一次 `counts_[key]++`，其频次合并语义等价于旧路径的 `Counter.update(delta_pairs)`，但不会创建 Python `Counter`，也不会执行 Python dict merge。

因为 count 单调增加，只有等于 threshold 的瞬间需要修改 membership。随后直接从 `active_keys_` 构建 open-addressing table。

当前 frequency/no-top-keep fast path 的职责边界如下：

| 阶段 | 当前执行位置 |
| --- | --- |
| GPU delta pair append | CUDA |
| 读取 delta 长度、触发异步任务 | Python scheduler |
| delta pairs D2H | PyTorch/CUDA copy |
| delta frequency merge | **Stateful C++ `counts_`** |
| threshold membership 更新 | **Stateful C++ `active_keys_`** |
| open-addressing table build | **Stateful C++** |
| CPU table H2D 与引用替换 | Python/PyTorch |
| speculative lookup | CUDA |

### 6.3 新数据路径

```text
speculative sampling CUDA kernel
    |
    v
GPU delta pair Tensor
    |
    | D2H Tensor copy（不再 .tolist()/Counter）
    v
CPU int64 pair Tensor
    |
    | ThreadPoolExecutor submits TorchBind method
    v
C++ unordered_map frequency update
    |
    v
C++ threshold membership
    |
    v
C++ open-addressing hash build
    |
    v
CPU int64 hash Tensor
    |
    | H2D in apply boundary
    v
GPU hash Tensor swap
    |
    v
CUDA speculative lookup
```

在线 C++ 计算通过 Torch dispatcher/TorchBind 执行，不持有 Python GIL。Python worker 只做一次 native method 调用和 payload 包装。

### 6.4 apply 边界

apply 仍在 scheduler thread：

```python
payload = rebuild_future.result()
self.table = materialize_csd_hash_table_payload(payload, device)
```

这一步包含 CPU Tensor 到 GPU Tensor 的传输以及旧 table Python reference 替换。它是显式短暂停顿，不属于后台 GIL 问题。后续可用 pinned memory、non-blocking H2D 和 CUDA event安全 swap 进一步优化，但当前 threshold=4096 下频率较低，不是主要瓶颈。

## 7. 为什么不是全 CUDA rebuild

CUDA 适合当前已经在 GPU 上的高频数据面：

- speculative sampling。
- delta append。
- 每轮 hash lookup。

rebuild 是低频控制面，包含：

- 动态增长 frequency map。
- 动态 active membership。
- capacity 估算。
- probe failure 后扩容重试。
- 新 table 生命周期和安全替换。

对约 4096 delta 和约 3.4 万 active keys，C++ CPU 已能在数毫秒内完成完整 update/build。CUDA dynamic hashmap、扩容和生命周期管理会显著增加实现复杂度，也可能引入额外同步。当前合理边界是：CUDA 负责 append/lookup，C++ CPU 负责 rebuild。

## 8. Fallback 与兼容性

只有满足以下条件才使用 stateful builder：

- dynamic update 开启。
- key selection strategy 为 `frequency`。
- rebuild top-keep 未启用。
- `torch.classes.sgl_kernel.CSDTableBuilder` 已注册。

否则继续使用 Python rebuild。这样 pair-score、above-uniform-share、top-keep 等需要全局排序/统计的策略不会被错误地套用 frequency-only 增量逻辑。

因此，在代码中仍能看到 `flush_delta()`、`Counter` 和 `merge_counts()`，并不表示当前 Stateful fast path 会执行它们。它们服务于 fallback、显式 delta 保存以及不满足上述条件的策略。`maybe_start_async_rebuild()` 在 `native_table_builder` 存在时调用 `drain_to_cpu_tensor()`，随后直接提交 `build_csd_rebuild_payload_native()`；只有 native builder 不存在时才调用 `flush_delta()` 进入 Python Counter 路径。

当前测试环境通过：

```bash
export SGLANG_CSD_NATIVE_EXTENSION_PATH=/path/to/csd_rebuild_cpu_ext.so
```

让每个 TP server 子进程在加载 `csd_runtime.py` 时调用 `torch.ops.load_library()`。正式重新构建 `sgl-kernel` 后不需要该环境变量，因为 CMake 已包含 `csrc/speculative/csd_rebuild.cpp`。

## 9. Snapshot 与持久化

TorchBind class 提供：

```text
snapshot() -> (Tensor keys, Tensor frequencies)
num_store_entries() -> int
```

正常 decode 不把 native counts 同步回 Python dict。metrics 使用 native `num_store_entries`。调用 `save_table()` 时才导出完整 snapshot，重建 Python `CSDTableStore.entries` 并使用原有 JSON/JSONL 保存逻辑。

这种设计避免在每轮 rebuild 中重复维护两份 105 万 entry 状态，同时保留显式持久化能力。

## 10. 正确性与性能测试

### 10.1 Hash parity

已验证：

- 34305 production keys 与 Python table 逐槽一致。
- 0/1/17/1000/10000 随机 keys。
- 重复 key 去重。
- negative key、非法 max probe/load factor 拒绝。

### 10.2 Stateful 逐轮 parity

使用 production 1052300-entry 初始 table，连续 20 批 delta；每批包含：

- 3500 个已有 keys。
- 596 个新 keys，包含重复。

每轮检查：

- native store entry 数与 Python dict 一致。
- native active entry 数与 Python `freq >= threshold` 一致。
- 随机 active key 均能在 native hash table 的 max-probe 范围内找到。
- 最终 snapshot 的完整 key/frequency dict 与 Python reference 完全一致。

结果：

```text
initial entries: 1,052,300
native initialization: 89.5 ms
mean update+build: 4.25 ms
max update+build: 9.61 ms
parity: OK
```

### 10.3 GIL profile

1000 次 stateful native rebuild：

```text
mean rebuild: 2.80 ms
max rebuild: 4.09 ms
main heartbeat p99 gap: 0.563 ms
main heartbeat max gap: 1.057 ms
```

Python rebuild 对照：

```text
mean rebuild: 29-32 ms
main heartbeat p99 gap: 6.7 ms
main heartbeat max gap: 8.0 ms
```

stateful trace：

```text
benchmark/csd/runs/csd_rebuild_cpu_profile_stateful_cpp/
    csd_rebuild_gil.chrometrace.json
```

### 10.4 真实 TP4 threshold=64

同一 APPS prompt、1098 output tokens：

| 实现 | tok/s（含 profiler） |
|---|---:|
| Python rebuild | 36.91 |
| stateless C++ hash | 37.05 |
| stateful C++ rebuild | 37.12 |

threshold=64 是故意放大 rebuild 频率的压力测试。stateful extension 已在 TP4 多进程、GPU D2H、后台 future、H2D apply 全链路运行通过。

## 11. Full coding benchmark 协议

正式测试方法：

1. `eagle`
2. `dynamic_ignore_ratio`，stateful C++，threshold=4096
3. `dynamic_entropy_p20_ignore_ratio`，stateful C++，threshold=4096

不运行 plain 或 huge。

数据集：

- APPS 2000
- TACO 1000
- LiveCodeBench v6 175

共同参数：

```text
model: Qwen3.5-35B-A3B
TP: 4
parallel/max_running_requests: 24
max_new_tokens: 81920
temperature: 1.0
top_p: 0.95
top_k: 20
presence_penalty: 1.5
tree: steps=5, topk=1, draft_tokens=5
CSD frequency threshold: 6
CSD rebuild threshold: 4096
P20 entropy threshold: 1.5638477802276611
```

GPU 分组：

- GPU 0-3：APPS 三方法。
- GPU 4-7：TACO+LCB 三方法。

输出路径：

```text
benchmark/csd/runs/20260710_stateful_cpp_full_apps
benchmark/csd/runs/20260710_stateful_cpp_full_taco_lcb
```

## 12. Full benchmark 结果

### 12.1 TACO

| method | tok/s | vs eagle | accept | spec success | avg output | max-new hit |
|---|---:|---:|---:|---:|---:|---:|
| eagle | 2696.3 | baseline | 3.331 | 46.62% | 20858.8 | 16/1000 |
| dynamic stateful C++ | 3741.1 | +38.75% | 4.689 | 73.78% | 18670.5 | 74/1000 |
| P20 stateful C++ | 3574.0 | +32.55% | 4.397 | 67.94% | 18889.6 | 52/1000 |

dynamic counters：

```text
delta pairs: 2,736,334
rebuild started/applied: 666/666
final hash entries/capacity: 88,159 / 262,144
native store entries: 1,473,192
rebuild lifecycle wall: 36.35 s
```

P20 counters：

```text
delta pairs: 3,154,117
rebuild started/applied: 768/768
final hash entries/capacity: 95,698 / 524,288
native store entries: 1,525,306
rebuild lifecycle wall: 41.43 s
```

`rebuild_lifecycle_wall` 是 submit 到 apply 的生命周期，可与 decode 重叠，不能直接当成阻塞时间。

### 12.2 LiveCodeBench v6

| method | tok/s | vs eagle | accept | spec success | avg output | max-new hit |
|---|---:|---:|---:|---:|---:|---:|
| eagle | 2436.0 | baseline | 3.361 | 47.22% | 23924.2 | 7/175 |
| dynamic stateful C++ | 3245.7 | +33.24% | 4.434 | 68.68% | 18498.2 | 11/175 |
| P20 stateful C++ | 3052.8 | +25.32% | 4.131 | 62.62% | 18586.1 | 6/175 |

### 12.3 APPS

APPS eagle 完成：

```text
tok/s: 2711.7
accept: 3.346
avg output: 23812.8
max-new hit: 61/2000
```

APPS dynamic 与 P20 在运行中被用户要求停止，以优先进行 verify 成本定位。两项均未完成，因此不报告或外推 partial-run 吞吐。

### 12.4 TACO accept 到吞吐的闭环

speculative success 定义：

```text
success = (accept_length - 1) / speculative_num_steps
accept_length = 1 + 5 * success
```

因此 success 从 46.62% 到 73.78% 不是 58.3% 的 token-yield 提升。真实 accept ratio：

```text
4.689 / 3.331 = 1.4077
```

新 run 的摊销 wall/verify：

```text
eagle:   1.235 ms
dynamic: 1.253 ms
```

预计吞吐比：

```text
(4.689 / 3.331) / (1.253 / 1.235) = 1.388
```

实际吞吐比：

```text
3741.1 / 2696.3 = 1.388
```

相对 eagle，accept 收益几乎完整转化为吞吐。

历史 plain 对照：

```text
plain accept: 4.077
plain tok/s: 3485.7
plain wall/verify: 1.170 ms
plain verify-weighted context: 14.99k

dynamic accept: 4.689
dynamic tok/s: 3741.1
dynamic wall/verify: 1.253 ms
dynamic verify-weighted context: 18.94k
```

预计 dynamic/plain：

```text
(4.689 / 4.077) / (1.253 / 1.170) = 1.074
```

实际：

```text
3741.1 / 3485.7 = 1.073
```

相对 plain 转化较少，完全由每轮成本增加解释。下一节进一步定位这部分成本。

## 13. Verify 成本受控定位

### 13.1 为什么 `wall/verify` 不能直接定位算子

此前使用：

```text
wall_ms_per_verify = benchmark wall time / sum(spec_verify_ct)
```

这是摊销系统指标，包含 target、draft、post-verify、CPU sync、prefill、scheduler、batch tail 和 rebuild/apply。它不是 `verify()` 或某个 CUDA kernel 的测量值。

要区分“dynamic 实现更贵”和“dynamic 改变 workload”，必须固定：

- 相同 prompt。
- 相同 batch size。
- 相同 output token 数。
- 相同 context-length 轨迹。
- 忽略 EOS，避免方法改变停止位置。

### 13.2 等 context 方法对照

实验：

```text
prompt: 14,930 tokens
output: fixed 4,096 tokens, ignore_eos
batch/max_running: 1
methods: eagle, plain, stateful dynamic
context trajectory: approximately 14.9k -> 19.0k
```

CPU ranges：

| range, 每轮 | eagle | plain | dynamic |
|---|---:|---:|---:|
| target_forward_verify | 5260 us | 5233 us | 5196 us |
| verify_total | 3444 us | 3425 us | 3421 us |
| draft_total | 2199 us | 2208 us | 2206 us |
| accept_index CPU sync | 2677 us | 2656 us | 2669 us |
| tree sampling | 87.7 us | 100.2 us | 81.5 us |

GPU user-annotation ranges：

| range, 每轮 | eagle | plain | dynamic |
|---|---:|---:|---:|
| target_forward_verify | 6517 us | 6485 us | 6456 us |
| draft_total | 2684 us | 2694 us | 2692 us |
| draft_extend | 834 us | 856 us | 809 us |
| target softmax/topk/topp | 282 us | 283 us | 282 us |
| tree sampling | 48.9 us | 72.3 us | 70.1 us |

结论：在相同 context 与 shape 下，dynamic 的 target forward、verify total 和 draft 均没有额外成本。target GPU range 甚至比 eagle 小约 0.94%，属于正常 run noise。CSD lookup/sampling 增量只有约 21 us/range，不可能解释历史 plain→dynamic 的 83 us `wall/verify` 差异。

trace：

```text
benchmark/csd/runs/verify_context_profile/
    20260711_verify_context15k_fixed4k/
```

### 13.3 短/长 context 对照

为了只测 context 效应，使用相同 eagle、batch=1、固定 4096 output：

- short：约 212-token prompt，context 约 0.2k→4.3k。
- long：14930-token prompt，context 约 14.9k→19.0k。

每轮 CPU range：

| range | short | long | delta |
|---|---:|---:|---:|
| target_forward_verify | 5034 us | 5260 us | +4.49% |
| verify_total | 3295 us | 3444 us | +4.52% |
| draft_total | 2170 us | 2199 us | +1.35% |
| draft_extend | 933 us | 933 us | 0.0% |
| accept_index CPU sync | 2524 us | 2677 us | +6.08% |

GPU range：

| range | short | long | delta |
|---|---:|---:|---:|
| target_forward_verify | 6208 us | 6517 us | +4.97% |
| draft_total | 2660 us | 2684 us | +0.91% |
| draft_extend | 830 us | 834 us | +0.57% |

增长集中在 target verify。`accept_index_predict_to_cpu_sync` 自身 GPU annotation 只有约 39 us；其 CPU range 增长是因为 `.to(cpu)` 是等待此前 target GPU 工作完成的同步点，并不表示 index copy kernel 本身耗时 2.7 ms。

### 13.4 CUDA kernel 归属方法

torch trace 中 `gpu_user_annotation` 给出 GPU timeline 上的 `csd_worker:target_forward_verify` 区间。分析脚本按 `(pid, tid/stream)` 建立有序区间，并将时间戳落在 target 区间内的 `cat=kernel` events归属到 target forward；随后按 target range 次数归一化。

主要正增量：

| target kernel，每个 GPU target range | short | long | delta |
|---|---:|---:|---:|
| FlashAttention main | 79.65 us | 159.01 us | +79.36 us |
| FlashAttention combine | 26.81 us | 38.13 us | +11.33 us |
| TP cross-device reduce | 548.22 us | 581.33 us | +33.11 us |
| fused MoE | 888.67 us | 908.85 us | +20.18 us |
| all target kernels | 3411 us | 3556 us | +145 us |

最大、最符合 context scaling 的增量来自 FlashAttention。cross-device reduce/MoE 理论上不随 KV context 线性增长，其小幅变化主要视为运行噪声或频率/调度差异。

### 13.5 最终定位

可以将历史 TACO plain→dynamic 的单位 verify 成本增加定位为：

```text
dynamic 改变输出/停止分布
    -> verify-weighted context 14.99k 增至 18.94k
    -> target verify FlashAttention 读取更长 KV cache
    -> target GPU range 变长
    -> accept_index CPU sync 等待时间增加
    -> 摊销 wall/verify 上升
```

明确排除或弱化：

- dynamic CSD lookup：等 context 下无 target/verify 回归。
- stateful C++ rebuild GIL：GIL profile 已基本消除，且等 context dynamic 不慢。
- draft model context cost：短/长仅约 +1%。
- tree sampling kernel：绝对时间几十微秒，远小于 target forward。

## 14. 已知限制与后续工作

1. 当前 C++ stateful fast path 仅支持 frequency/no-top-keep。
2. D2H drain 仍通过 `counter.item()` 确定长度，会产生 CPU/GPU sync。
3. apply 仍在 scheduler thread执行 CPU-to-GPU materialize/swap。
4. 每个 TP rank 当前各自维护相同 native builder并执行相同 rebuild，计算有重复；后续可只由 TP0 rebuild 后广播 CPU/GPU table，但需要保证 collective 不引入更大同步。
5. full run 中不同方法输出分布不同。端到端 tok/s 必须同时结合固定 decode window、平均输出长度、max-new hit 和 verify 加权 context解释。
6. 当前工作区 native `.so` 是独立 JIT build，用于无需修改 root-owned build tree 的验证。正式部署应重新构建并安装完整 `sgl-kernel` wheel。
