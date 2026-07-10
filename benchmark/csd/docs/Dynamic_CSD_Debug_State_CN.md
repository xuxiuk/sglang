# Dynamic CSD Debug State

更新时间：2026-07-10

## 当前工作方向

我们在排查一个核心问题：dynamic CSD 明显提高 speculative success / accept length，但端到端吞吐提升很小，甚至在部分设置下变慢。

当前不能再简单用“accept 提高就应该线性提速”来解释，因为实际系统里还包含 target verify、draft extend、CPU sync、online rebuild、连续批处理调度、输出长度分布和 max_new hit 等因素。现在的目标是把这些因素拆开，明确 dynamic 的额外开销到底在哪里，以及为什么 4096 rebuild threshold 下投机成功率提升很大但吞吐只提升一点。

## 工作逻辑与修改脉络

这轮工作的主线不是单纯“多跑几个数据集”，而是逐步排除 dynamic 变慢的来源。最初观察是：`dynamic` 相比 `plain` 的 speculative success / accept length 明显提高，但端到端 tok/s 没有按比例提高，甚至部分设置更慢。因此我们把问题拆成三类：

1. dynamic 每步热路径是否变慢。

   这里要看开启 dynamic 以后，即使不真正 rebuild，仅仅因为 kernel 分支、delta append、runtime 状态维护，是否已经造成主循环变慢。为此加入了 `dynamic_ignore_ratio_huge` / `p20_no_ratio_huge` 这类方法：dynamic update 和 delta append 都打开，但 rebuild threshold 设置得极大，基本不触发真实 rebuild。这个对照用于回答“dynamic hot path 本身是不是主因”。

2. online rebuild 是否造成累计开销。

   如果 `dynamic_huge` 接近 `plain`，但普通 `dynamic` 明显慢，那么问题更可能来自 rebuild。于是我们比较了 `CSD_REBUILD_THRESHOLD=512` 和 `4096`。512 用于放大 rebuild 问题，4096 用于验证降低 rebuild 频率后是否恢复吞吐。现在的结论是：512 确实会严重拖慢；4096 明显缓解，但 full run 里仍需要继续 profile 残余成本。

3. accept 提升为什么没有转化成吞吐提升。

   这里不能只看总 tok/s。我们补了 request sidecar、server decode metrics、profile ranges 和 accept simulation。request sidecar 用于看每个请求的输出长度、latency、accept、max_new hit、CSD counter；server decode metrics 用于看真实 decode 窗口吞吐，避免只被完成请求窗口或长尾样本误导；accept simulation 用于估计“如果 accept 提升没有 dynamic/rebuild 副作用，理论上应该提速多少”。

当前修改大致对应这些验证目的：

- `python/sglang/srt/speculative/eagle_worker.py`
  - 增加 draft、target verify、post-verify、rebuild start/apply 等 profile range。
  - 目的：拆出每轮 speculative decode 中 target forward、draft forward、verify、rebuild 检查/应用分别花了多少。

- `python/sglang/srt/speculative/eagle_info.py`
  - 标记 verify 内部的 softmax/topk/topp、tree sampling kernel、accept index CPU sync、KV cache 更新、draft input 构建等范围。
  - 目的：确认 entropy / tree sampling / CPU sync 是否是 dynamic 或 p20 的额外瓶颈。

- `python/sglang/srt/speculative/csd_runtime.py`
  - 增加 `_csd_profile_range` 和 `metrics_snapshot()`。
  - 导出 `csd_lookup_hit_ct`、`csd_forced_accept_ct`、`csd_delta_pair_ct`、`csd_table_num_entries`、`csd_table_capacity`、`csd_delta_buffer_capacity` 等。
  - 标记 rebuild start / merge / filter / build / future_result / materialize_and_swap。
  - 目的：证明 dynamic 是否真的写入 delta、是否触发 rebuild、table 是否增长，以及 rebuild 是否被后台线程覆盖。

- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
  - 把 CSD runtime counters 放进 response `meta_info`。
  - 目的：让 benchmark 的每个请求都能记录 CSD 行为，而不是只依赖服务端日志。

- `benchmark/csd/eval/bench_sglang_chat_generation.py`
  - 写出 `*.request_metrics.jsonl` 和 completed-window 统计。
  - 目的：分析每个请求的 completion tokens、latency、request throughput、accept length、max_new hit 和 CSD counters。

- `benchmark/csd/eval/scrape_sglang_decode_metrics.py`
  - 从 SGLang `/metrics` 抓取 decode token counter、running reqs、queue reqs。
  - 目的：计算固定时间窗口的 decode 吞吐，区分“真实 decode 变慢”和“请求完成分布/长尾导致统计口径变差”。

- `benchmark/csd/eval/profile_csd_dynamic_overhead.py`
  - 支持 `dynamic_no_ratio_huge`、`p20_no_ratio_huge`、`--prompt-file`、`--prompt-indices`。
  - 目的：对同一批 prompt 做 profile，对比 plain / dynamic_huge / dynamic / p20，避免不同样本导致结论不稳定。

- `benchmark/csd/eval/run_domain_ood_csd_515.sh`
  - 统一封装 domain OOD 测试方法，包括 plain、dynamic ignore ratio、dynamic huge、p20 等。
  - 目的：可重复地跑 APPS / TACO / LCB 等 OOD code 数据集，并确保生成参数、树结构、ratio 设置一致。

所以新对话接手时，应该按这个顺序理解：先看 `dynamic_huge` 判断热路径，再看 512 vs 4096 判断 rebuild，再看 request sidecar / decode window 判断吞吐口径和长尾，最后用 profile range 定位 verify、draft、rebuild 的真实开销。当前仍缺的是把 static 表命中和 dynamic 新增表命中拆开，否则只能知道总 forced accept 增加，不能确认新增表项的实际成本收益比。

## 已确认结论

1. 静态 CSD 的收益是真实的。

   在已有主报告中，`plain` 相对 `eagle` 明显更快。例如 LCB 主表里 `eagle` 约 2502 tok/s，`plain` 约 3208 tok/s。新补的 APPS128 counter 对照也显示：

   - `eagle`: 3183.9 tok/s, accept 3.251
   - `plain`: 3680.3 tok/s, accept 3.903

   所以不能说 CSD force accept 本身没有收益。

2. dynamic hot path / delta append 本身不是主要问题。

   `dynamic_ignore_ratio_huge` 开启 dynamic update 和 delta append，但把 rebuild threshold 设成极大，基本不触发真实 rebuild。它和 `plain` 几乎一样快：

   - `plain`: 3680.3 tok/s
   - `dynamic_huge`: 3695.3 tok/s

   这说明每步写 delta buffer、dynamic flag、kernel 中的动态分支，不是吞吐损失主因。

3. 低 rebuild threshold 会造成非常大的端到端损失。

   APPS128，`CSD_REBUILD_THRESHOLD=512`：

   - `plain`: 3680.3 tok/s, accept 3.903
   - `dynamic`: 2953.9 tok/s, accept 3.971

   accept 提高了，但吞吐低 19.7%。这说明 512 下真实 rebuild/flush/apply 的累计成本很大。

4. 4096 threshold 能大幅缓解 rebuild 成本，但 dynamic 仍只小幅慢或小幅快。

   APPS128，`CSD_REBUILD_THRESHOLD=4096`：

   - `plain`: 3740.8 tok/s, accept 3.897
   - `dynamic_huge`: 3710.2 tok/s, accept 3.884
   - `dynamic`: 3629.8 tok/s, accept 3.982

   4096 下 dynamic 只慢 2.97%，比 512 好很多。但 accept 增益只有 3.897 -> 3.982，这个小增益不足以覆盖残余 rebuild 成本。

5. 在更大的 full run 中，确实存在 accept/spec success 提升很大但吞吐只提升一点的情况。

   典型例子是 `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260708_code_ood_huge_g1`：

   - APPS: `plain` 3416.4 tok/s, accept 4.116；`dynamic` 3584.9 tok/s, accept 4.944
   - TACO: `plain` 3485.7 tok/s, accept 4.077；`dynamic` 3639.2 tok/s, accept 4.691
   - LCB: `plain` 3092.6 tok/s, accept 4.147；`dynamic` 3133.7 tok/s, accept 4.489

   这里 APPS / TACO 的 accept 提升约 15%-20%，吞吐只提升 4%-5%；LCB 的 accept 也明显提升，但吞吐只提升约 1.3%。这是下一步需要重点 profile 的对象。

6. `dynamic_entropy_p20_ignore_ratio` 是当前最后/最终候选方法。

   这个方法是在 `dynamic_ignore_ratio` 基础上加 P20 entropy gate。目标不是最大化 accept，而是抑制高 entropy 场景下的错误 force accept，减少长尾和 max_new hit。它通常会降低 accept 和吞吐，但能明显减少部分数据集上的 hit。

   在 code OOD huge run 中：

   - APPS: `dynamic` hit 286/2000；`p20` hit 167/2000
   - TACO: `dynamic` hit 72/1000；`p20` hit 50/1000
   - LCB: `dynamic` hit 13/175；`p20` hit 8/175

   所以 p20 是“更保守、更稳”的最终候选方法，需要和 raw dynamic 一起 profile。不能只看 raw dynamic。

## 当前使用的数据集

目前最重要的是三个 code OOD 数据集，它们都来自 `/root/sglang/benchmark/csd/runs/code_ood_data`，并被转换成 Alpaca-style chat generation 输入：

1. APPS

   - 数据文件：`/root/sglang/benchmark/csd/runs/code_ood_data/apps_2000_code_ood_alpaca.json`
   - 样本数：2000
   - 任务类型：代码生成 / 编程题解。
   - 当前观察：dynamic 的 accept 提升最明显之一，但吞吐提升不成比例。
   - full run 结果：
     - `plain`: 3416.4 tok/s, accept 4.116, avg out 20808.7, hit 62/2000
     - `dynamic`: 3584.9 tok/s, accept 4.944, avg out 23930.8, hit 286/2000
     - `p20`: 3364.6 tok/s, accept 4.524, avg out 23043.7, hit 167/2000
     - accept +20.1%，tok/s 只 +4.9%，同时 avg out 和 max_new hit 明显增加。
     - p20 把 hit 从 286 降到 167，但吞吐低于 plain，说明 entropy gate 能抑制长尾，但也付出了 accept/吞吐代价。

2. TACO

   - 数据文件：`/root/sglang/benchmark/csd/runs/code_ood_data/taco_1000_code_ood_alpaca.json`
   - 样本数：1000
   - 任务类型：代码生成 / 算法题。
   - 当前观察：和 APPS 类似，dynamic 的 accept 提升明显，但吞吐提升只有几百分点。
   - full run 结果：
     - `plain`: 3485.7 tok/s, accept 4.077, avg out 16963.2, hit 20/1000
     - `dynamic`: 3639.2 tok/s, accept 4.691, avg out 18671.8, hit 72/1000
     - `p20`: 3432.9 tok/s, accept 4.402, avg out 18923.3, hit 50/1000
     - accept +15.1%，tok/s 只 +4.4%，hit 增加。
     - p20 把 hit 从 72 降到 50，但吞吐仍低于 plain。

3. LCB / LiveCodeBench v6

   - 数据文件：`/root/sglang/benchmark/csd/runs/code_ood_data/lcb_v6_175_code_ood_alpaca.json`
   - 样本数：175
   - 任务类型：LiveCodeBench code generation。
   - 当前观察：投机成功率也有明显提升，但吞吐提升最小，而且样本数较少、长尾更明显。
   - full run 结果：
     - `plain`: 3092.6 tok/s, accept 4.147, avg out 19677.0, hit 7/175
     - `dynamic`: 3133.7 tok/s, accept 4.489, avg out 20101.2, hit 13/175
     - `p20`: 3122.8 tok/s, accept 4.167, avg out 20922.7, hit 8/175
     - accept +8.2%，tok/s 只 +1.3%，hit 增加。
     - p20 基本把 hit 拉回 plain 附近，吞吐略高于 plain，但 accept 提升也明显变小。

这三个数据集目前都支持“dynamic 的 speculative success/accept length 明显上涨，但端到端吞吐上涨很少”的现象。APPS 和 TACO 是下一步最优先 profile 对象，因为样本数更多、accept 提升更大；LCB 可以作为补充验证，但 175 条样本更容易受长尾和个别样本影响。p20 是当前最后候选方法，profile 时必须和 `plain`、`dynamic_huge`、raw `dynamic` 一起看。

另外，之前也跑过 Legal Case Summary、BillSum、FLARE、PubMed、GovReport、arXiv、LongBench、医学/法律/金融 OOD 等数据集。它们主要用于寻找 OOD 场景和确认 dynamic 是否会增加长尾；当前 profile 主线优先聚焦 APPS / TACO / LCB。

## 关键实验路径

主结果报告：

- `/root/sglang/benchmark/csd/docs/Result_CSD_515_P20_NO_RATIO_CN.md`

full code OOD huge 对照：

- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260708_code_ood_huge_g1`

APPS128 counter 对照，threshold=512：

- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260710_csd_counter_quality_apps128`

APPS128 counter 对照，threshold=4096：

- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260710_csd_counter_quality_apps128_rebuild4096`

accept 模拟上限实验：

- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260710_accept_sim_apps128_acc4.1`
- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260710_accept_sim_apps128_acc4.9`
- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260710_accept_sim_apps24_max81920_acc4.1`
- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260710_accept_sim_apps24_max81920_acc4.9`

profile 结果：

- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/profiles/20260710_draft_cost_profile`
- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/profiles/20260710_verify_s_drop_apps_profile_max81920`
- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/profiles/20260710_verify_s_drop_force_rebuild64_profile`

## 当前代码改动状态

为了排查问题，当前代码已经增加了 profiling/counter 相关能力：

- `python/sglang/srt/speculative/eagle_worker.py`
  - 增加 draft、target verify、post-verify 等 profile range。

- `python/sglang/srt/speculative/eagle_info.py`
  - tree sampling、softmax、CPU sync、KV cache、draft input 构建等路径已有 CSD profile range。

- `python/sglang/srt/speculative/csd_runtime.py`
  - 增加 `metrics_snapshot()`，能导出 `csd_lookup_hit_ct`、`csd_forced_accept_ct`、`csd_delta_pair_ct`、table entries/capacity 等。

- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
  - 将 CSD runtime metrics 写入 response meta info。

- `benchmark/csd/eval/bench_sglang_chat_generation.py`
  - request sidecar 会写出 `csd_*` metrics。

- `benchmark/csd/eval/profile_csd_dynamic_overhead.py`
  - 支持 `dynamic_no_ratio_huge`、`p20_no_ratio_huge`、prompt file 和 prompt indices。

注意：部分 benchmark 脚本可能是 untracked 文件，继续前应先看 `git status --short`。

## 已排除或弱化的假设

1. “dynamic kernel 分支本身很慢”

   不成立或不是主因。`dynamic_huge` 和 `plain` 接近。

2. “delta append atomic 是主因”

   不成立或不是主因。`dynamic_huge` 同样有大量 `csd_delta_pair_ct`，但不慢。

3. “单次 rebuild apply 很大”

   profile 中单次 apply 不大，但端到端看低 threshold 会非常慢。因此问题更像频繁 rebuild/flush/apply 的累计成本，以及它对主循环 sync/调度的影响。

4. “长上下文导致 accept 提升无法转化”

   只解释一部分。accept 模拟实验显示，即使 max_new=81920，accept 4.1 -> 4.9 仍能带来约 19.5% 吞吐提升。所以真实 dynamic 中 accept 提高但吞吐只小涨，不能只归因于长上下文。

## 当前最可能的问题

4096 场景下仍然需要继续查：

1. dynamic 的真实 rebuild 残余成本。

   512 下很明显，4096 下小很多，但 full run 中可能仍然累计可见。需要 profile full run 中投机成功率大幅提升的 APPS/TACO 场景，直接比较：

   - `plain`
   - `dynamic_ignore_ratio_huge`
   - `dynamic_ignore_ratio` with threshold 4096

2. dynamic 新增表项的成本收益比。

   full run 中 dynamic accept 提升很多，说明表确实生效；但新增 accept 是否带来更长输出、更多 max_new hit、或者更差 batch 结束形态，需要继续量化。

3. accept 指标本身可能过于粗。

   需要区分：

   - normal accept
   - static CSD force accept
   - dynamic-added table force accept

   目前只有总 `csd_forced_accept_ct`，还不能区分 force accept 来自原始静态表还是 online rebuild 后新增表项。这个是下一步最值得加的 instrumentation。

## 下一步 profile 计划

目标：针对“4096 下 accept 提升很大但吞吐只提升一点”的 full run 做 profile，而不是再看 512。

建议 profile APPS 或 TACO，使用相同 prompt 子集，并保持长输出：

- dataset: APPS 或 TACO
- max_new_tokens: 81920
- parallel/max_running_requests: 尽量接近 full run，至少不要只用 batch=1
- methods:
  - `plain`
  - `dynamic_ignore_ratio_huge`
  - `dynamic_ignore_ratio` with `CSD_REBUILD_THRESHOLD=4096`
  - `dynamic_entropy_p20_ignore_ratio` with `CSD_REBUILD_THRESHOLD=4096`

重点采集：

- PyTorch trace profile ranges:
  - `csd_worker:target_forward_verify`
  - `csd_worker:verify_total`
  - `csd_worker:draft_total`
  - `csd_worker:draft_extend_after_decode_total`
  - `csd_worker:maybe_start_async_rebuild_total`
  - `csd_worker:maybe_apply_async_rebuild`
  - `csd_rebuild_start:*`
  - `csd_rebuild_apply:*`
  - `csd_verify:tree_spec_sampling_kernel`
  - `csd_verify:accept_index_predict_to_cpu_sync`

- request sidecar:
  - `completion_tokens`
  - `request_latency`
  - `request_throughput`
  - `accept_length`
  - `csd_lookup_hit_ct`
  - `csd_forced_accept_ct`
  - `csd_delta_pair_ct`
  - `csd_table_num_entries`
  - `csd_table_capacity`

- server decode metrics:
  - 30s/60s decode token window
  - running reqs
  - queue reqs

## 需要补的 instrumentation

最有价值的新增 counter：

1. `csd_forced_accept_static_ct`
2. `csd_forced_accept_dynamic_added_ct`
3. `csd_lookup_hit_static_ct`
4. `csd_lookup_hit_dynamic_added_ct`

实现思路：

- 给 hash table value 加来源标记，或者维护两个 hash table：
  - initial static table
  - dynamic added table
- kernel 查表时区分命中来源。

这样才能回答：dynamic 新增的表项到底贡献了多少 forced accept，以及这些 forced accept 是否真的值得。

## 当前一句话结论

静态 CSD 是有效的；dynamic hot path 不是主要问题；低 rebuild threshold 会显著拖慢；4096 后 rebuild 成本大幅缓解，但 full run 中“accept 大涨、吞吐小涨”的真正原因还需要继续 profile 4096 场景，并区分静态表 force accept 与 dynamic 新增表项 force accept 的贡献。
