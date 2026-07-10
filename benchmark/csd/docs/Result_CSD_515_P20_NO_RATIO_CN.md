# CSD 515 P20 No-Ratio 实验报告

## 策略标识说明

后续表格中的列名保留实验短标识，便于和实验目录、日志、脚本参数对应。`auto` 是非投机基线，不启用 EAGLE/CSD；`eagle` 只启用 MTP 投机解码，不启用 CSD force accept；`plain` 是静态 CSD，使用离线 calibration 得到的 CSD table，不做在线更新。

`dynamic` 是在 `plain` 基础上开启 online dynamic update。这里的 dynamic 会在生成过程中收集新的 token pair，累计到阈值后在线 rebuild CSD table。`dynamic_no_ratio` 也是 online dynamic update，但 **no_ratio 只作用在 dynamic delta pair 的收集阶段**：收集新 pair 时不再要求该 pair 通过 prob-ratio gate，因此会记录更多候选 pair，也更容易提高后续 table hit / force accept 机会。

需要注意，`no_ratio` 不是把最终 CSD force accept 的所有约束都关掉。最终是否 force accept 仍然要走 CSD table 命中、采样验证和其他策略逻辑；`no_ratio` 只是说 dynamic 更新时“哪些 pair 可以被记录进 delta buffer”不再受 prob-ratio 条件限制。`p20_no_ratio` 则是在 `dynamic_no_ratio` 基础上额外增加 P20 entropy gate，用 entropy 抑制高不确定场景下的 force accept，目标是减少 raw dynamic 带来的噪声。

补跑的 `topkeep15000_no_ratio` / `topkeep15000_p20_no_ratio` 都是在 no-ratio dynamic update 基础上设置 `CSD_REBUILD_TOP_KEEP=15000`，即 online rebuild 后只保留频次排序靠前的 15000 个 CSD keys；后者额外叠加 P20 entropy gate。

## 结论摘要与阅读顺序

这份报告现在按“主实验结果 -> 生成长度/长尾解释 -> single-batch 对照 -> OOD 探索 -> profile 证据”的顺序组织。主结论如下：

- 静态 `plain` 仍然是当前最强 Pareto 点：精度最高，Batch=48 平均吞吐也最高。
- `dynamic` / `dynamic_no_ratio` 能稳定提高投机成功率，但 Batch=48 下没有稳定转化成端到端吞吐提升。
- single-batch 下 dynamic 可以加速，说明 dynamic table 不是无效；Batch=48 收益消失更像是连续批处理形态、长尾输出和请求结束时机抵消了 accept 收益。
- long-output OOD 上，`CSD_REBUILD_THRESHOLD=512` 会让 dynamic 慢 20%-27%；调到 4096 后 rebuild 开销大幅缓解，但端到端加速仍然不稳定。
- profile 显示 rebuild/apply 和 entropy 直接计算都不是唯一主因；更应该关注 dynamic 如何改变生成长度分布、max_new hit、verify 次数和 post-verify/scheduler 行为。

## 核心表：macro 指标（含生成长度）

| method | strategy | acc % | Δacc vs plain | tok/s | Δtok/s vs plain | spec success % | Δspec vs plain | accept len | saved % | avg gen len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto` | Auto / 非投机基线 | 84.091 | -1.815 pp | 2466.2 | -35.41% | 0.000 | -67.698 pp | 0.000 | 0.000 | 17796.9 |
| `eagle` | EAGLE baseline | 83.876 | -2.030 pp | 3302.6 | -13.51% | 54.340 | -13.358 pp | 3.885 | 73.015 | 18000.1 |
| `plain` | Static CSD / 静态 CSD | 85.906 | +0.000 pp | 3818.4 | +0.00% | 67.698 | +0.000 pp | 4.458 | 77.175 | 17062.0 |
| `dynamic` | Dynamic CSD / 动态 CSD | 83.933 | -1.973 pp | 3638.0 | -4.72% | 72.115 | +4.417 pp | 4.621 | 78.282 | 17744.3 |
| `dynamic_no_ratio` | Dynamic CSD no-ratio / 动态 CSD 无 ratio 收集门控 | 84.090 | -1.816 pp | 3762.6 | -1.46% | 72.728 | +5.030 pp | 4.646 | 78.418 | 17080.0 |
| `p20_no_ratio` | Dynamic CSD + P20 entropy no-ratio / 动态 CSD + P20 entropy | 85.128 | -0.778 pp | 3645.8 | -4.52% | 68.897 | +1.199 pp | 4.516 | 77.485 | 17667.8 |
| `topkeep15000_no_ratio` | Dynamic CSD no-ratio + topkeep15000 | 83.907 | -1.999 pp | 3655.6 | -4.26% | 69.500 | +1.802 pp | 4.519 | 77.645 | 17006.7 |
| `topkeep15000_p20_no_ratio` | Dynamic CSD + P20 entropy no-ratio + topkeep15000 | 84.857 | -1.049 pp | 3653.6 | -4.32% | 66.405 | -1.293 pp | 4.411 | 76.832 | 16680.2 |

> `avg gen len` 是平均生成 token 数/request。吞吐对比需要同时看这个数：如果某个方法生成更长或更短，端到端 tok/s 和总耗时都会受影响。

## 吞吐与长尾指标口径

后续 OOD 脚本新增了请求级 sidecar：每个 answer 文件旁边会写出 `*.request_metrics.jsonl`，逐请求记录 `prompt_tokens`、`completion_tokens`、`request_latency`、`request_end_offset`、`request_throughput`、`spec_verify_ct` 和逐请求 `accept_length`。这比只看总 tok/s 更适合定位 dynamic 的损失来源，因为可以直接看到尾部请求是不是更长、更慢、是否更容易 hit `max_new_tokens`。

需要特别说明 `peak` 的计算口径。当前 `peak_completed_30s_throughput` / `peak_completed_60s_throughput` 是 **client 侧完成事件窗口吞吐**：按请求完成时间排序，在最近 30s/60s 完成的请求中累加 completion tokens，再除以完整窗口长度。它不是服务端 decode loop 的瞬时 GPU 吞吐，也不是 CUDA kernel peak。旧实现曾经用窗口内首尾完成时间差做除数，当多个请求几乎同时返回时会产生不真实的超大 peak；这个已经改成除以完整窗口长度或启动初期已运行时间。

`peak_prefix_throughput` 是从开跑到某个完成前缀的累计 tokens / elapsed time，并且至少等 10% 请求完成后才开始取最大值。它比完成窗口 peak 稳定，但如果最高点出现在最后一个请求完成时，就会等于总 tok/s。因此它只能说明“完成曲线前段是否更快”，不能单独证明 dynamic 的真实峰值 decode 能力。

新增的 `head90_throughput`、`tail10_token_share`、`tail10_time_share` 用来量化长尾影响：`head90` 只看前 90% 完成请求的 tokens / time；`tail10_token_share` 是最后 10% 完成请求贡献的 token 比例；`tail10_time_share` 是从第一个 tail 请求完成到全局结束这段时间占总 wall time 的比例。如果 dynamic 的 accept 明显更高但总 tok/s 没涨，需要同时检查这些 tail 指标、`max_new_token_hits`、p95/p99 输出长度和 request latency。真正要把“长尾拖慢”与“rebuild/verify 开销”拆开，还需要对同一数据集补三组对照：`plain`、`dynamic` with `CSD_REBUILD_THRESHOLD=4096`、以及 `dynamic` with 极大 rebuild threshold 近似禁用 rebuild。前两者差值包含 dynamic 全部影响，后两者差值更接近真实 rebuild/apply/flush 开销。

## 精度分任务表（%）

| task | auto | eagle | plain | dynamic | dynamic_no_ratio | p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 74.286 | 71.429 | 73.714 | 71.429 | 68.571 | 74.286 | 69.143 | 72.000 |
| AIME25 | 86.667 | 90.000 | 93.333 | 90.000 | 93.333 | 90.000 | 93.333 | 93.333 |
| Math500 | 85.800 | 83.400 | 85.600 | 83.400 | 83.400 | 85.400 | 82.400 | 83.800 |
| GSM8K | 89.613 | 90.675 | 90.978 | 90.902 | 91.054 | 90.826 | 90.751 | 90.296 |
| Macro | 84.091 | 83.876 | 85.906 | 83.933 | 84.090 | 85.128 | 83.907 | 84.857 |

## 吞吐分任务表（tok/s）

| task | auto | eagle | plain | dynamic | dynamic_no_ratio | p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 1647.8 | 2502.2 | 3208.0 | 3095.2 | 3222.1 | 2989.9 | 3297.7 | 2987.8 |
| AIME25 | 1274.2 | 2132.1 | 2555.7 | 2489.1 | 2859.1 | 2504.3 | 2537.6 | 2481.7 |
| Math500 | 3175.4 | 4189.0 | 4854.2 | 4520.6 | 4445.4 | 4617.5 | 4198.9 | 4656.1 |
| GSM8K | 3767.5 | 4387.0 | 4655.6 | 4447.0 | 4523.6 | 4471.4 | 4588.2 | 4489.0 |
| Avg | 2466.2 | 3302.6 | 3818.4 | 3638.0 | 3762.6 | 3645.8 | 3655.6 | 3653.6 |

## 吞吐加速比分任务表（vs auto）

| task | auto | eagle | plain | dynamic | dynamic_no_ratio | p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 1.000x | 1.519x | 1.947x | 1.878x | 1.955x | 1.814x | 2.001x | 1.813x |
| AIME25 | 1.000x | 1.673x | 2.006x | 1.953x | 2.244x | 1.965x | 1.992x | 1.948x |
| Math500 | 1.000x | 1.319x | 1.529x | 1.424x | 1.400x | 1.454x | 1.322x | 1.466x |
| GSM8K | 1.000x | 1.164x | 1.236x | 1.180x | 1.201x | 1.187x | 1.218x | 1.192x |
| Avg | 1.000x | 1.339x | 1.548x | 1.475x | 1.526x | 1.478x | 1.482x | 1.481x |

## 投机成功率分任务表（%）

| task | auto | eagle | plain | dynamic | dynamic_no_ratio | p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 0.000 | 47.640 | 62.920 | 72.590 | 73.430 | 64.550 | 67.380 | 61.540 |
| AIME25 | 0.000 | 54.790 | 69.980 | 72.160 | 71.680 | 70.920 | 70.160 | 67.780 |
| Math500 | 0.000 | 56.150 | 70.060 | 74.940 | 76.620 | 71.810 | 72.950 | 69.310 |
| GSM8K | 0.000 | 58.780 | 67.830 | 68.770 | 69.180 | 68.310 | 67.510 | 66.990 |
| Macro | 0.000 | 54.340 | 67.698 | 72.115 | 72.728 | 68.897 | 69.500 | 66.405 |

## 平均生成长度分任务表（tokens/request）

| task | auto | eagle | plain | dynamic | dynamic_no_ratio | p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 32561.4 | 33682.7 | 31447.4 | 36518.8 | 35193.6 | 32727.9 | 32694.5 | 31447.7 |
| AIME25 | 28628.8 | 28481.4 | 27978.9 | 25033.3 | 23022.8 | 28829.1 | 25875.2 | 25997.7 |
| Math500 | 8566.5 | 8398.1 | 7500.5 | 8063.4 | 8750.9 | 7774.2 | 8087.1 | 7902.3 |
| GSM8K | 1431.1 | 1438.1 | 1321.1 | 1361.6 | 1352.7 | 1339.8 | 1370.2 | 1373.1 |
| Macro | 17796.9 | 18000.1 | 17062.0 | 17744.3 | 17080.0 | 17667.8 | 17006.7 | 16680.2 |

## Batch=48 fair rerun：plain entropy 与 dynamic no-ratio 复跑

这组补跑使用同一脚本和同一并发设置（`MAX_RUNNING_REQUESTS=48`），补充 `plain_entropy_p20` 作为公平 baseline：静态 CSD + P20 entropy gate，不开启 dynamic update。`dynamic_no_ratio_rerun` 和 `p20_no_ratio_rerun` 是对主表中 `dynamic_no_ratio` / `p20_no_ratio` 的复跑，用于确认动态更新改动后的稳定性；它们不是新策略。

| method | acc % | tok/s | spec success % | accept len | saved % | avg gen len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `plain_entropy_p20` | 85.644 | 3736.7 | 65.380 | 4.381 | 76.543 | 16280.7 |
| `dynamic_no_ratio_rerun` | 84.233 | 3755.7 | 72.728 | 4.647 | 78.418 | 17080.0 |
| `p20_no_ratio_rerun` | 85.128 | 3644.6 | 68.898 | 4.516 | 77.485 | 17667.8 |

| task | plain_entropy_p20 acc % | plain_entropy_p20 tok/s | dynamic_no_ratio_rerun acc % | dynamic_no_ratio_rerun tok/s | p20_no_ratio_rerun acc % | p20_no_ratio_rerun tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 73.714 | 2994.5 | 69.143 | 3229.4 | 74.286 | 3006.4 |
| AIME25 | 93.333 | 2681.0 | 93.333 | 2857.7 | 90.000 | 2506.1 |
| Math500 | 84.400 | 4654.8 | 83.400 | 4406.6 | 85.400 | 4583.2 |
| GSM8K | 91.130 | 4616.4 | 91.054 | 4529.1 | 90.826 | 4483.0 |
| Macro | 85.644 | 3736.7 | 84.233 | 3755.7 | 85.128 | 3644.6 |

结论：`plain_entropy_p20` 比 `plain` 精度低 0.262 pp、吞吐低约 2.14%，说明 P20 entropy gate 本身不是免费增益；`p20_no_ratio_rerun` 与主表 `p20_no_ratio` 基本一致，动态更新改动后没有看到明显精度漂移。

## LCB 长尾与 max_new hit

LCB 上 dynamic 的核心问题不只是平均生成长度变长，而是更容易产生撞到 `max_new_tokens` 的长尾样本。`max_new_tokens=81920` 的旧主表里，`plain` 有 17/175 个样本 hit max_new，`dynamic` 增加到 34/175，`dynamic_no_ratio` 增加到 30/175；`p20_no_ratio` 回到 17/175，说明 P20 entropy gate 对抑制长尾是有效的。

把 LCB 单独放长到 `max_new_tokens=102400` 后，这个现象仍然存在：`plain` 是 15/175，`dynamic` 是 30/175，`dynamic_no_ratio` 是 26/175。也就是说，更长的 max_new 没有消除 dynamic 的长尾倾向，dynamic 仍然更容易让一批样本继续生成到上限附近。

| setting | auto | eagle | plain | dynamic | dynamic_no_ratio | p20 / entropy dynamic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB, `max_new=81920` | N/A | N/A | 17/175 | 34/175 | 30/175 | 17/175 (`p20_no_ratio`) |
| LCB, `max_new=102400` | 7/175 | 10/175 | 15/175 | 30/175 | 26/175 | 9/175 (`dynamic_entropy_p20_no_ratio`) |

`max_new=81920` 旧主表没有保留 auto/eagle 的逐请求 metrics artifact，因此这里不填它们的 max_new hit，避免用无法复算的数据。`max_new=102400` 这轮保留了逐请求 metrics，所以可以精确统计 auto/eagle/plain/dynamic 的 hit 数。

结论是：dynamic 确实提高了投机成功率，但在 LCB 上也改变了生成分布，显著增加长输出/撞长度上限的概率。这可以解释 Batch=48 主表里 dynamic 吞吐没有随投机成功率上涨而明显提升的现象：连续批处理不会像静态 batch 那样被单个请求严格卡死，但长尾请求变多会让队列尾部更长、有效 batch 形态更差、请求结束时机更分散，最终抵消一部分 accept length 收益。端到端速度和精度不能只看 spec success，需要同时看生成长度分布、max_new hit 和最终代码抽取是否被长输出污染。

## Single-batch 结果：`MAX_RUNNING_REQUESTS=1`

这组结果来自 batch=1 稳定精度实验，路径为 `/root/sglang/benchmark/csd/runs/lighteval_csd_515_calib1024_batch1_all_methods/20260702_010858_515_calib1024_batch1_all_methods/results/batch1_all_methods.jsonl`。生成配置仍是 thinking 评测配置：`max_new_tokens=81920`，`temperature=1.0`，`top_p=0.95`，`top_k=20`，`presence_penalty=1.5`。这里的 `Avg` / `Macro` 与 Batch=48 主表一致，按 LCB / AIME25 / Math500 / GSM8K 四个任务做 task-level 平均。

### Single-batch macro 指标（含生成长度）

| method | acc % | Δacc vs plain | tok/s | Δtok/s vs plain | spec success % | Δspec vs plain | accept len | saved % | avg gen len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto` | 85.180 | +0.359 pp | 206.4 | -51.13% | 0.000 | -67.580 pp | 0.000 | 0.000 | 17127.0 |
| `eagle` | 84.639 | -0.182 pp | 361.0 | -14.55% | 54.510 | -13.070 pp | 3.887 | 73.068 | 18039.0 |
| `plain` | 84.821 | +0.000 pp | 422.4 | +0.00% | 67.580 | +0.000 pp | 4.466 | 77.148 | 16505.2 |
| `plain_entropy_p20` | 83.036 | -1.785 pp | 413.9 | -2.02% | 65.722 | -1.858 pp | 4.378 | 76.630 | 17771.8 |
| `dynamic` | 81.853 | -2.968 pp | 443.8 | +5.06% | 71.817 | +4.237 pp | 4.616 | 78.210 | 17791.1 |
| `dynamic_no_ratio` | 82.103 | -2.718 pp | 441.7 | +4.55% | 72.442 | +4.862 pp | 4.654 | 78.360 | 17779.4 |
| `dynamic_entropy_p20_no_ratio` | 84.818 | -0.003 pp | 421.6 | -0.19% | 68.790 | +1.210 pp | 4.523 | 77.460 | 17479.4 |
| `topkeep15000_no_ratio` | 85.135 | +0.314 pp | 424.5 | +0.49% | 68.750 | +1.170 pp | 4.516 | 77.455 | 16568.8 |
| `topkeep15000_p20_no_ratio` | 84.954 | +0.133 pp | 414.7 | -1.83% | 66.310 | -1.270 pp | 4.410 | 76.803 | 17188.2 |

Single-batch 下 dynamic 的速度收益更明显：`dynamic` 比 `plain` 快 5.06%，`dynamic_no_ratio` 快 4.55%，同时投机成功率分别提高 4.237 pp 和 4.862 pp。这个现象说明 dynamic update 本身在低并发场景是能转化成吞吐收益的；Batch=48 主表中收益消失，更可能与连续批处理、长尾样本、请求结束时机和调度形态有关，而不是 dynamic table 完全无效。LCB 上所有 dynamic / topkeep 类方法仍然相对 `auto` 或 `plain` 存在不同程度的精度下降，但 single-batch 的主要观察目标是速度和 speculative 行为，这个精度问题不影响“长尾导致 batch 场景收益被稀释”的判断。

### Single-batch 平均生成长度分任务表（tokens/request）

| task | auto | eagle | plain | plain_entropy_p20 | dynamic | dynamic_no_ratio | dynamic_entropy_p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 31982.8 | 32076.5 | 30538.8 | 32274.3 | 34141.6 | 35670.9 | 33188.2 | 31481.3 | 32078.0 |
| AIME25 | 26249.0 | 30146.9 | 26842.2 | 29490.0 | 27447.7 | 26475.5 | 27454.1 | 25591.2 | 27552.7 |
| Math500 | 8807.9 | 8515.3 | 7259.2 | 7965.9 | 8212.4 | 7609.5 | 7890.7 | 7799.4 | 7766.1 |
| GSM8K | 1468.4 | 1417.2 | 1380.6 | 1356.9 | 1362.7 | 1361.7 | 1384.4 | 1403.5 | 1356.1 |
| Macro | 17127.0 | 18039.0 | 16505.2 | 17771.8 | 17791.1 | 17779.4 | 17479.4 | 16568.8 | 17188.2 |

### Single-batch max_new hit 分任务表

这里的 hit 数由各方法的 lighteval metrics artifact 中逐请求 `completion_tokens == 81920` 复算得到。和 Batch=48 主表一致，single-batch 下 LCB 仍然是长尾主要来源：`plain` 为 13/175，`dynamic` 增加到 28/175，`dynamic_no_ratio` 增加到 30/175；P20 entropy / topkeep 能缓解一部分，但不能完全消除长尾。

| task | auto | eagle | plain | plain_entropy_p20 | dynamic | dynamic_no_ratio | dynamic_entropy_p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 12/175 | 16/175 | 13/175 | 12/175 | 28/175 | 30/175 | 18/175 | 17/175 | 14/175 |
| AIME25 | 0/30 | 2/30 | 0/30 | 0/30 | 1/30 | 2/30 | 0/30 | 0/30 | 0/30 |
| Math500 | 2/500 | 1/500 | 0/500 | 2/500 | 7/500 | 4/500 | 2/500 | 4/500 | 2/500 |
| GSM8K | 0/1319 | 0/1319 | 0/1319 | 0/1319 | 0/1319 | 0/1319 | 0/1319 | 0/1319 | 0/1319 |
| Total | 14/2024 | 19/2024 | 13/2024 | 14/2024 | 36/2024 | 36/2024 | 20/2024 | 21/2024 | 16/2024 |

### Single-batch 精度分任务表（%）

| task | auto | eagle | plain | plain_entropy_p20 | dynamic | dynamic_no_ratio | dynamic_entropy_p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 77.714 | 76.571 | 72.000 | 71.429 | 69.143 | 65.714 | 73.143 | 73.714 | 74.857 |
| AIME25 | 90.000 | 86.667 | 93.333 | 86.667 | 83.333 | 86.667 | 90.000 | 90.000 | 90.000 |
| Math500 | 84.000 | 85.400 | 83.200 | 83.600 | 83.200 | 84.600 | 85.000 | 86.000 | 83.600 |
| GSM8K | 89.007 | 89.917 | 90.751 | 90.447 | 91.736 | 91.433 | 91.130 | 90.826 | 91.357 |
| Macro | 85.180 | 84.639 | 84.821 | 83.036 | 81.853 | 82.103 | 84.818 | 85.135 | 84.954 |

### Single-batch 吞吐分任务表（tok/s）

| task | auto | eagle | plain | plain_entropy_p20 | dynamic | dynamic_no_ratio | dynamic_entropy_p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 200.9 | 326.8 | 408.2 | 385.3 | 437.1 | 445.8 | 394.4 | 414.3 | 394.4 |
| AIME25 | 204.8 | 369.3 | 434.2 | 424.4 | 443.0 | 443.1 | 433.7 | 427.4 | 422.5 |
| Math500 | 209.3 | 368.1 | 435.2 | 434.7 | 473.5 | 455.5 | 442.6 | 443.2 | 428.5 |
| GSM8K | 210.9 | 379.7 | 412.1 | 411.2 | 421.6 | 422.2 | 415.9 | 413.1 | 413.4 |
| Avg | 206.4 | 361.0 | 422.4 | 413.9 | 443.8 | 441.7 | 421.6 | 424.5 | 414.7 |

### Single-batch 吞吐加速比分任务表（vs auto）

| task | auto | eagle | plain | plain_entropy_p20 | dynamic | dynamic_no_ratio | dynamic_entropy_p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 1.000x | 1.627x | 2.032x | 1.918x | 2.176x | 2.219x | 1.963x | 2.062x | 1.963x |
| AIME25 | 1.000x | 1.804x | 2.121x | 2.073x | 2.164x | 2.164x | 2.118x | 2.087x | 2.063x |
| Math500 | 1.000x | 1.759x | 2.080x | 2.077x | 2.263x | 2.177x | 2.115x | 2.118x | 2.048x |
| GSM8K | 1.000x | 1.800x | 1.954x | 1.950x | 1.999x | 2.002x | 1.972x | 1.959x | 1.961x |
| Avg | 1.000x | 1.748x | 2.046x | 2.005x | 2.150x | 2.139x | 2.042x | 2.056x | 2.009x |

### Single-batch 投机成功率分任务表（%）

| task | auto | eagle | plain | plain_entropy_p20 | dynamic | dynamic_no_ratio | dynamic_entropy_p20_no_ratio | topkeep15000_no_ratio | topkeep15000_p20_no_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCB | 0.000 | 47.500 | 63.410 | 59.520 | 71.030 | 73.150 | 64.730 | 66.260 | 61.140 |
| AIME25 | 0.000 | 55.520 | 70.410 | 68.440 | 72.690 | 72.570 | 70.490 | 69.330 | 68.040 |
| Math500 | 0.000 | 55.650 | 69.660 | 68.580 | 74.810 | 74.800 | 71.830 | 72.230 | 68.890 |
| GSM8K | 0.000 | 59.370 | 66.840 | 66.350 | 68.740 | 69.250 | 68.110 | 67.180 | 67.170 |
| Macro | 0.000 | 54.510 | 67.580 | 65.722 | 71.817 | 72.442 | 68.790 | 68.750 | 66.310 |

## Long-output OOD summarization 补充

这组实验专门寻找更长输出、更偏分布外的 summarization / legal summarization 场景。和前面的 short domain OOD 不同，这里使用 thinking 配置，目标是让请求长度足够长，确保 dynamic update 有机会真正触发。

统一配置：Qwen3.5-35B-A3B，树结构 `515:5:1:5`，RedPajama calibration table，`enable_thinking=true`，`temperature=1.0`，`top_p=0.95`，`top_k=20`，`presence_penalty=1.5`。除特别说明外，`MAX_RUNNING_REQUESTS=24` / `PARALLEL=24`。这里统计 generation 吞吐和 speculative 指标，不报告 summarization 官方质量分。

先用 `max_new_tokens=4096`、`CSD_REBUILD_THRESHOLD=512` 在 1000 条样本上做压力测试。结果显示 dynamic 的 accept 确实提高，但吞吐下降非常大，说明过低 rebuild 阈值会带来明显累计开销。

| dataset | samples | method | tok/s | Δtok/s vs plain | accept | spec% | avg out | hit |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BillSum | 1000 | `eagle` | 3034.3 | -9.06% | 3.170 | 43.40 | 2587.1 | 4/1000 |
| BillSum | 1000 | `plain` | 3336.8 | +0.00% | 3.498 | 49.96 | 2479.5 | 1/1000 |
| BillSum | 1000 | `dynamic_ignore_ratio` | 2538.0 | -23.94% | 3.613 | 52.26 | 2467.1 | 3/1000 |
| BillSum | 1000 | `dynamic_entropy_p20_ignore_ratio` | 2510.3 | -24.77% | 3.564 | 51.28 | 2468.2 | 3/1000 |
| FLARE-EDTSum | 1000 | `eagle` | 3084.7 | -7.55% | 3.298 | 45.96 | 2520.2 | 23/1000 |
| FLARE-EDTSum | 1000 | `plain` | 3336.7 | +0.00% | 3.611 | 52.22 | 2559.2 | 38/1000 |
| FLARE-EDTSum | 1000 | `dynamic_ignore_ratio` | 2648.4 | -20.63% | 3.753 | 55.06 | 2547.2 | 41/1000 |
| FLARE-EDTSum | 1000 | `dynamic_entropy_p20_ignore_ratio` | 2617.9 | -21.54% | 3.709 | 54.18 | 2546.7 | 39/1000 |
| Legal Case Summary | 1000 | `eagle` | 2864.6 | -12.01% | 3.064 | 41.28 | 3493.3 | 359/1000 |
| Legal Case Summary | 1000 | `plain` | 3255.7 | +0.00% | 3.502 | 50.04 | 3329.4 | 281/1000 |
| Legal Case Summary | 1000 | `dynamic_ignore_ratio` | 2379.8 | -26.90% | 3.708 | 54.16 | 3318.0 | 278/1000 |
| Legal Case Summary | 1000 | `dynamic_entropy_p20_ignore_ratio` | 2419.8 | -25.68% | 3.591 | 51.82 | 3342.1 | 271/1000 |

把 rebuild 阈值调到 `CSD_REBUILD_THRESHOLD=4096` 后，显式 rebuild 管理开销明显下降。下面是 full / larger run，`max_new_tokens=8192`。

| dataset | samples | method | tok/s | Δtok/s vs plain | accept | spec% | avg out | max out | hit |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BillSum | 3269 | `eagle` | 3068.4 | -7.87% | 3.175 | 43.50 | 2578.7 | 5632 | 0/3269 |
| BillSum | 3269 | `plain` | 3330.4 | +0.00% | 3.501 | 50.02 | 2467.3 | 4467 | 0/3269 |
| BillSum | 3269 | `dynamic_ignore_ratio` | 3315.0 | -0.46% | 3.700 | 54.00 | 2466.8 | 5840 | 0/3269 |
| BillSum | 3269 | `dynamic_entropy_p20_ignore_ratio` | 3250.7 | -2.39% | 3.634 | 52.68 | 2464.4 | 4503 | 0/3269 |
| FLARE-EDTSum | 2000 | `eagle` | 3177.6 | -5.84% | 3.295 | 45.90 | 2533.8 | 6787 | 0/2000 |
| FLARE-EDTSum | 2000 | `plain` | 3374.7 | +0.00% | 3.607 | 52.14 | 2576.9 | 7327 | 0/2000 |
| FLARE-EDTSum | 2000 | `dynamic_ignore_ratio` | 3349.0 | -0.76% | 3.816 | 56.32 | 2615.8 | 8192 | 5/2000 |
| FLARE-EDTSum | 2000 | `dynamic_entropy_p20_ignore_ratio` | 3324.9 | -1.47% | 3.764 | 55.28 | 2614.5 | 8192 | 3/2000 |
| Legal Case Summary | 7773 | `eagle` | 2922.2 | -11.36% | 3.048 | 40.96 | 3908.1 | 8192 | 18/7773 |
| Legal Case Summary | 7773 | `plain` | 3296.7 | +0.00% | 3.482 | 49.64 | 3619.1 | 8192 | 18/7773 |
| Legal Case Summary | 7773 | `dynamic_ignore_ratio` | 3251.8 | -1.36% | 3.913 | 58.26 | 3705.1 | 8192 | 189/7773 |
| Legal Case Summary | 7773 | `dynamic_entropy_p20_ignore_ratio` | 3080.4 | -6.56% | 3.709 | 54.18 | 3673.7 | 8192 | 65/7773 |
| PubMed Summarization | 2000 | `plain` | 3177.2 | +0.00% | 3.499 | 49.98 | 2494.7 | 6899 | 0/2000 |
| PubMed Summarization | 2000 | `dynamic_ignore_ratio` | 3159.2 | -0.57% | 3.660 | 53.20 | 2505.3 | 8192 | 1/2000 |
| PubMed Summarization | 2000 | `dynamic_entropy_p20_ignore_ratio` | 3124.0 | -1.67% | 3.604 | 52.08 | 2504.9 | 7015 | 0/2000 |
| GovReport | 973 | `plain` | 3031.2 | +0.00% | 3.312 | 46.24 | 2713.0 | 4118 | 0/973 |
| GovReport | 973 | `dynamic_ignore_ratio` | 3007.6 | -0.78% | 3.420 | 48.40 | 2687.5 | 4546 | 0/973 |
| GovReport | 973 | `dynamic_entropy_p20_ignore_ratio` | 2993.1 | -1.26% | 3.367 | 47.34 | 2711.5 | 4723 | 0/973 |
| arXiv Summarization | 2000 | `plain` | 2973.6 | +0.00% | 3.356 | 47.12 | 2375.1 | 7309 | 0/2000 |
| arXiv Summarization | 2000 | `dynamic_ignore_ratio` | 2950.4 | -0.78% | 3.513 | 50.26 | 2424.0 | 8192 | 2/2000 |
| arXiv Summarization | 2000 | `dynamic_entropy_p20_ignore_ratio` | 2917.7 | -1.88% | 3.445 | 48.90 | 2413.5 | 5698 | 0/2000 |

这组结果说明两件事。第一，`CSD_REBUILD_THRESHOLD=512` 时 dynamic 慢 20%-27%，这个是 rebuild 频率过高导致的系统性开销；阈值调到 4096 后，dynamic 在 BillSum / FLARE / PubMed / GovReport / arXiv 上只慢约 0.5%-0.8%，说明 rebuild 优化和阈值调整有效。第二，即使 accept/spec% 明显提高，端到端吞吐仍然没有稳定提升；Legal Case Summary 最典型，`dynamic_ignore_ratio` 的 spec% 从 49.64 提高到 58.26，但 max_new hit 从 18 增到 189，吞吐仍低 1.36%。`dynamic_entropy_p20_ignore_ratio` 能把 hit 从 189 降到 65，但 accept 回落且额外逻辑存在，最终吞吐低 6.56%。

因此，当前更准确的结论是：dynamic update 可以提高投机接受率，但端到端加速比非常不稳定。低 rebuild 阈值会造成显著额外开销；阈值调大后，剩余问题主要来自生成分布和连续批处理形态，尤其是长尾输出、max_new hit、请求结束时机分散、后半段有效 batch size 下降。只报告平均输出长度不够，必须同时报告 `hit`、`max out`、最好再补 p95/p99 输出长度。

当前还有一轮 `legal_case_summary_7773_long_alpaca` 的 `max_new_tokens=4096`、`CSD_REBUILD_THRESHOLD=4096` 正在运行，用来进一步确认截断长尾后 dynamic 是否还能接近或超过 plain；这轮尚未产出完整结果，因此没有写入上表。

## 知名 domain OOD 数据集补充：MedMCQA / PubMedQA / FinQA / LegalBench-CUAD

为了避免只用不够 canonical 的探索数据集做结论，这里补了一组更知名的医学、金融和法律数据集。医学使用 `MedMCQA` 和 `PubMedQA`，金融使用 `FinQA`，法律使用 `LegalBench-CUAD`。这些任务来自医学 QA、生物医学文献 QA、财报数值推理和合同条款理解，领域上比 RedPajama calibration table 更明显 OOD。

这组结果来自：

- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260705_domain_ood_med_pubmed_512/results/domain_ood_csd_515.jsonl`
- `/root/sglang/benchmark/csd/runs/domain_ood_csd_515/20260705_domain_ood_legal_fin_512/results/domain_ood_csd_515.jsonl`

统一生成参数：`max_new_tokens=384`，`temperature=0.7`，`top_p=0.8`，`top_k=20`，`presence_penalty=1.5`，`enable_thinking=false`，空 system prompt，树结构 `515:5:1:5`，`MAX_RUNNING_REQUESTS=24` / `PARALLEL=24`。每个数据集取 512 条样本。这里统计 generation 吞吐和 speculative 指标，不报告各 benchmark 官方 accuracy；其中 MedMCQA / PubMedQA / FinQA / CUAD 都被转成 Alpaca-style generation 格式，用来观察动态 CSD 的接受率和吞吐行为。这组同样是 non-thinking 历史结果，后续如果要和 RedPajama calibration 对齐，应使用 `enable_thinking=true`、`temperature=1.0`、`top_p=0.95` 复跑。

| dataset | method | tok/s | Δtok/s vs plain | accept len | spec success % | avg gen len | max gen len | max_new hit | avg prompt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MedMCQA | `plain` | 1468.0 | +0.00% | 3.308 | 46.16 | 158.9 | 384 | 31/512 | 106.0 |
| MedMCQA | `dynamic_no_ratio` | 1299.0 | -11.51% | 3.314 | 46.28 | 163.1 | 384 | 34/512 | 106.0 |
| MedMCQA | `dynamic_entropy_p20_no_ratio` | 1378.6 | -6.09% | 3.323 | 46.46 | 159.8 | 384 | 29/512 | 106.0 |
| PubMedQA | `plain` | 910.8 | +0.00% | 3.282 | 45.64 | 61.7 | 204 | 0/512 | 399.9 |
| PubMedQA | `dynamic_no_ratio` | 841.8 | -7.57% | 3.325 | 46.50 | 63.0 | 384 | 2/512 | 399.9 |
| PubMedQA | `dynamic_entropy_p20_no_ratio` | 870.6 | -4.41% | 3.305 | 46.10 | 61.8 | 384 | 1/512 | 399.9 |
| FinQA | `plain` | 1743.7 | +0.00% | 4.656 | 73.12 | 181.5 | 384 | 22/512 | 1068.5 |
| FinQA | `dynamic_no_ratio` | 1634.1 | -6.29% | 4.683 | 73.66 | 182.9 | 384 | 20/512 | 1068.5 |
| FinQA | `dynamic_entropy_p20_no_ratio` | 1689.4 | -3.12% | 4.681 | 73.62 | 182.5 | 384 | 25/512 | 1068.5 |
| LegalBench-CUAD | `plain` | 896.1 | +0.00% | 3.439 | 48.78 | 41.7 | 131 | 0/512 | 126.7 |
| LegalBench-CUAD | `dynamic_no_ratio` | 789.3 | -11.92% | 3.474 | 49.48 | 42.2 | 152 | 0/512 | 126.7 |
| LegalBench-CUAD | `dynamic_entropy_p20_no_ratio` | 838.5 | -6.43% | 3.443 | 48.86 | 41.5 | 92 | 0/512 | 126.7 |

这组结果没有找到合适的 dynamic showcase。虽然这些数据集领域更 OOD，也更适合写报告，但 dynamic 的 accept 提升仍然很小：MedMCQA 只有约 +0.006 到 +0.015，PubMedQA 约 +0.023 到 +0.043，FinQA 约 +0.025 到 +0.027，LegalBench-CUAD 约 +0.004 到 +0.035。对应 spec success 最多只提升不到 1 pp，不能抵消 dynamic 带来的调度形态变化和额外管理成本。

另一个观察是任务形态本身不理想。PubMedQA 和 LegalBench-CUAD 输出太短，平均只有约 62 和 42 tokens/request，动态更新机会有限；FinQA 的静态 `plain` 已经有 73.12% spec success，calibration table 对这类结构化推理并不差；MedMCQA 有一定输出长度，但 max_new hit 已经有 6% 左右，dynamic 反而略微增加长尾。结论是：这些知名 domain OOD 数据集适合说明“dynamic 不一定在领域迁移后自动变快”，但仍不是我们要找的明显正收益样例。

## Chat 数据集补充：MT-Bench / AlpacaEval / Arena-Hard

这组补充实验用于观察 daily chat 场景下 CSD dynamic 的速度表现。Chat 任务更接近开放式生成，投机成功率和生成长度分布都可能不同于数学/代码任务。

这组历史补跑使用的是 Qwen non-thinking 配置：`max_new_tokens=32768`，`temperature=0.7`，`top_p=0.8`，`top_k=20`，`presence_penalty=1.5`，`enable_thinking=false`，不额外添加 system prompt。服务端使用同一张 RedPajama calibration table 和同一棵树 `515:5:1:5`。这里统计的是生成吞吐和 speculative 指标，不是 MT-Bench / AlpacaEval / Arena-Hard 的官方 judge 质量分。

需要注意：RedPajama calibration table 和主 LightEval thinking 评测的分布更接近 thinking mode。所以下面这组 chat 结果只能作为 non-thinking 场景观察，不能直接作为 dynamic 在 RedPajama calibration 分布下的最终证据；后续 chat / OOD 测试脚本默认应使用 `enable_thinking=true`、`temperature=1.0`、`top_p=0.95`。

| task | method | tok/s | Δtok/s vs plain | accept len | spec success % | avg gen len | max gen len | max_new hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT-Bench | `auto` | 1210.2 | -47.00% | 1.000 | 0.00 | 666.3 | 10078 | 0/80 |
| MT-Bench | `eagle` | 2250.2 | -1.44% | 3.213 | 44.26 | 644.7 | 6162 | 0/80 |
| MT-Bench | `plain` | 2283.1 | +0.00% | 3.494 | 49.88 | 688.4 | 8023 | 0/80 |
| MT-Bench | `dynamic_no_ratio` | 2729.9 | +19.57% | 3.430 | 48.60 | 613.8 | 2227 | 0/80 |
| MT-Bench | `dynamic_entropy_p20_no_ratio` | 2205.2 | -3.41% | 3.520 | 50.40 | 723.1 | 8996 | 0/80 |
| AlpacaEval | `auto` | 1794.4 | -41.70% | 1.000 | 0.00 | 854.8 | 32768 | 2/805 |
| AlpacaEval | `eagle` | 2803.9 | -8.90% | 3.142 | 42.84 | 870.5 | 32768 | 3/805 |
| AlpacaEval | `plain` | 3077.8 | +0.00% | 3.362 | 47.24 | 869.2 | 32768 | 3/805 |
| AlpacaEval | `dynamic_no_ratio` | 2762.3 | -10.25% | 3.335 | 46.70 | 852.7 | 32768 | 3/805 |
| AlpacaEval | `dynamic_entropy_p20_no_ratio` | 2667.1 | -13.35% | 3.215 | 44.30 | 785.1 | 32768 | 1/805 |
| Arena-Hard v0.1 | `auto` | 2208.4 | -38.73% | 1.000 | 0.00 | 1663.1 | 32768 | 2/500 |
| Arena-Hard v0.1 | `eagle` | 3486.5 | -3.28% | 3.534 | 50.68 | 1768.0 | 32768 | 5/500 |
| Arena-Hard v0.1 | `plain` | 3604.6 | +0.00% | 3.716 | 54.32 | 1670.5 | 32768 | 4/500 |
| Arena-Hard v0.1 | `dynamic_no_ratio` | 3467.2 | -3.81% | 3.788 | 55.76 | 1754.8 | 32768 | 5/500 |
| Arena-Hard v0.1 | `dynamic_entropy_p20_no_ratio` | 3491.6 | -3.13% | 3.778 | 55.56 | 1759.6 | 32768 | 4/500 |

这里的 `spec success %` 由 `accept_length` 近似换算得到：`(accept_length - 1) / 5`，因为这组树的 draft tokens 是 5。观察结果比较明确：dynamic 在 chat 上仍然不稳定，而且多数情况下还是慢。MT-Bench 上 `dynamic_no_ratio` 最快，吞吐比 `plain` 高 19.57%，但这组的平均生成长度也明显更短，不能单独归因于 dynamic 更新本身。AlpacaEval 上 `dynamic_no_ratio` 比 `plain` 慢 10.25%，`dynamic_entropy_p20_no_ratio` 慢 13.35%，并且投机成功率没有超过 `plain`。Arena-Hard v0.1 上 dynamic 的投机成功率略高于 `plain`，但平均生成长度也更长，最终吞吐仍然低 3% 到 4%。

所以当前结论不是“dynamic rebuild 开销拖慢了一切”，而是 dynamic 带来的投机接受率提升并没有稳定转化为端到端吞吐收益。对于 chat 任务，生成长度分布、长尾样本、请求结束时机和调度形态会抵消一部分 accept length 收益；在 AlpacaEval / Arena-Hard 上，即使 dynamic 的 speculative 指标看起来不差，最终速度仍然弱于静态 `plain`。

## BFCL function-calling 补充

BFCL 这轮用于观察 function-calling / tool-call 风格的分布外场景。结果来自 `/root/sglang/benchmark/csd/runs/bfcl_csd_515/20260705_bfcl_single_turn_all_rerun/results/bfcl_csd_515.jsonl`，共 15 个 single-turn category。结果文件中每个 category 同时写了原始 task 行和展开行，统计时只保留不含 `parent_task` 的原始 60 行，避免重复计数。

统一生成参数：`max_new_tokens=1024`，`temperature=0.7`，`top_p=0.8`，`top_k=20`，`presence_penalty=1.5`，`enable_thinking=false`，空 system prompt，树结构 `515:5:1:5`，`MAX_RUNNING_REQUESTS=24` / `PARALLEL=24`。这里统计的是 generation 吞吐和 speculative 指标，没有跑 BFCL 官方 function-call judge，因此不报告 accuracy。这也是一组 non-thinking 历史结果，后续复跑应切到 `enable_thinking=true`、`temperature=1.0`、`top_p=0.95`。

| method | requests | output tokens | elapsed s | tok/s | Δtok/s vs plain | accept len | spec success % | avg gen len | max gen len | max_new hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eagle` | 3811 | 352204 | 281.518 | 1251.1 | -3.14% | 4.904 | 78.08 | 92.4 | 1024 | 15/3811 |
| `plain` | 3811 | 345028 | 267.117 | 1291.7 | +0.00% | 5.005 | 80.11 | 90.5 | 1024 | 13/3811 |
| `dynamic_no_ratio` | 3811 | 346248 | 270.514 | 1280.0 | -0.91% | 5.026 | 80.53 | 90.9 | 1024 | 15/3811 |
| `dynamic_entropy_p20_no_ratio` | 3811 | 344856 | 261.486 | 1318.8 | +2.10% | 5.032 | 80.64 | 90.5 | 1024 | 16/3811 |

| category | requests | plain tok/s | dynamic tok/s | Δdyn vs plain | dyn+p20 tok/s | Δdyn+p20 vs plain | plain accept | dynamic accept | dyn+p20 accept | avg gen len plain/dyn/dyn+p20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `irrelevance` | 240 | 1542.5 | 1384.7 | -10.23% | 1422.4 | -7.79% | 3.587 | 3.628 | 3.602 | 167.6/165.8/168.6 |
| `java` | 100 | 1367.0 | 1460.9 | +6.87% | 1523.4 | +11.44% | 5.647 | 5.656 | 5.690 | 60.3/59.8/60.1 |
| `javascript` | 50 | 1570.3 | 1611.7 | +2.63% | 1502.8 | -4.30% | 5.622 | 5.695 | 5.636 | 57.5/58.3/58.2 |
| `live_irrelevance` | 882 | 1222.2 | 1138.0 | -6.89% | 1159.0 | -5.17% | 3.601 | 3.610 | 3.609 | 110.8/111.3/110.2 |
| `live_multiple` | 1053 | 917.8 | 980.3 | +6.81% | 1008.3 | +9.86% | 5.479 | 5.517 | 5.522 | 61.0/60.6/60.4 |
| `live_parallel` | 16 | 1474.1 | 1447.1 | -1.83% | 1735.3 | +17.72% | 5.888 | 5.587 | 5.436 | 104.9/100.6/89.7 |
| `live_parallel_multiple` | 24 | 2228.3 | 2059.6 | -7.57% | 2113.9 | -5.13% | 5.849 | 5.822 | 5.812 | 95.5/97.0/96.9 |
| `live_relevance` | 18 | 1328.3 | 1389.0 | +4.57% | 1576.5 | +18.69% | 4.499 | 4.584 | 4.548 | 102.7/93.7/101.1 |
| `live_simple` | 258 | 1278.3 | 1247.1 | -2.44% | 1268.3 | -0.78% | 5.505 | 5.540 | 5.580 | 56.7/56.2/55.4 |
| `multiple` | 200 | 1177.9 | 1150.1 | -2.37% | 1229.3 | +4.36% | 5.657 | 5.733 | 5.705 | 56.3/55.0/56.0 |
| `parallel` | 200 | 2033.1 | 2128.4 | +4.69% | 2137.6 | +5.14% | 5.799 | 5.806 | 5.800 | 159.5/164.4/166.8 |
| `parallel_multiple` | 200 | 2060.3 | 2098.9 | +1.87% | 2102.2 | +2.03% | 5.854 | 5.803 | 5.862 | 163.9/169.1/165.5 |
| `rest` | 70 | 1220.9 | 1169.6 | -4.20% | 1564.0 | +28.10% | 5.135 | 5.091 | 5.434 | 101.4/102.8/92.2 |
| `simple` | 400 | 1263.2 | 1302.2 | +3.09% | 1344.1 | +6.40% | 5.739 | 5.755 | 5.728 | 54.8/54.6/54.9 |
| `sql` | 100 | 1535.6 | 1621.8 | +5.61% | 1776.1 | +15.66% | 5.701 | 5.736 | 5.714 | 86.3/86.8/86.6 |

BFCL 的输出长度整体较短到中等，平均约 90 tokens/request，少量样本会 hit `max_new_tokens=1024`。这组结果里 dynamic 的 accept 提升非常小：`plain` request-weighted accept 是 5.005，`dynamic_no_ratio` 是 5.026，`dynamic_entropy_p20_no_ratio` 是 5.032；对应 spec success 只提升约 0.4-0.5 pp。因此 BFCL 目前不是一个能明显展示 dynamic table 收益的 OOD 样例。`dynamic_entropy_p20_no_ratio` 的总吞吐比 `plain` 高 2.10%，但提升幅度小，且分 category 波动较大，更像生成长度和 category 形态共同作用的结果，不能作为强结论。

# 性能 profile 结果

## 我们做了什么优化

最初的 dynamic update 会在触发 rebuild 时把较多 CPU 侧工作压在 verify 热路径上，尤其是 delta flush、pair 过滤、排序、hash payload 构造和 table 替换。为了确认 dynamic 慢在哪里，我们先把 rebuild 路径拆开，并做了两类优化：

1. **异步 rebuild**：`maybe_start_async_rebuild()` 在 verify 后检查 delta buffer 是否达到阈值。如果达到阈值，只在主线程做 counter 读取、flush delta 和提交后台任务；真正的 CPU hash payload 构建放到单线程 `ThreadPoolExecutor` 后台执行。
2. **verify 边界 apply**：`maybe_apply_async_rebuild()` 在下一次 verify 前检查后台 future 是否完成。完成后主线程只做 `future.result()`、CUDA tensor materialize 和 table swap。后台线程不直接创建 CUDA tensor，避免跨线程操作 GPU 对象。
3. **减少无意义排序**：`filtered_keys()` 只有在设置 `top_keep` 时才需要排序并截断；没有 `top_keep` 时 hash table 只关心 membership，因此现在只过滤、不排序。`topkeep15000_*` 仍会排序，因为它需要保留排名靠前的 keys。
4. **增加 profiler ranges**：通过 `torch.profiler.record_function` 加了 `csd_worker:*`、`csd_verify:*`、`csd_rebuild_start:*`、`csd_rebuild_apply:*` 等 range。这里的 range 是我们手动插入的 CPU event，不依赖 Python stack；关闭 `with_stack` 不影响这些 event 的计时，但会看不到更细的 Python 调用栈。

当前 online update 的实际时序是：

```text
verify 前:
  maybe_apply_async_rebuild()
    future_result
    materialize_and_swap

target forward + verify:
  target_forward_verify
  verify_total
    target_probs_softmax_topk_topp
    tree_spec_sampling_kernel
    accept_index_predict_to_cpu_sync

verify 后:
  maybe_start_async_rebuild()
    counter_threshold_check
    flush_delta
    executor_submit(background CPU payload build)
```

## Profile 配置

- methods: `plain` / `dynamic_no_ratio` / `plain_entropy_p20` / `p20_no_ratio`
- TP=4，GPU 4/5/6/7
- `CSD_REBUILD_THRESHOLD=512`
- profile 请求数：4
- 主要 profile 轮次：
  - `max_new_tokens=4096`：看 dynamic / entropy 的整体 gap
  - `max_new_tokens=2048`：看 verify step 细分
  - entropy kernel microbench：单独估算 entropy 计算增量

## 端到端与 rebuild 触发

`max_new_tokens=4096`、4 requests 的 profile 里，端到端结果如下：

| method | elapsed s | output tokens | output tok/s |
| --- | ---: | ---: | ---: |
| `plain` | 127.495 | 9621 | 75.462 |
| `plain_entropy_p20` | 124.436 | 6365 | 51.151 |
| `dynamic_no_ratio` | 128.447 | 9621 | 74.902 |
| `p20_no_ratio` | 143.039 | 6959 | 48.651 |

这组短 profile 里 P20 方法生成 token 明显更少，因此 tok/s 不能直接当作公平速度结论。更可靠的是看 verify 次数、per-step event 和 rebuild range。

在强制 `CSD_REBUILD_THRESHOLD=512` 的 4096 profile 中，`dynamic_no_ratio` 确认触发了真实 online rebuild：12 次 start，12 次 apply。旧版聚合结果显示：

| method | rebuild started | rebuild applied |
| --- | ---: | ---: |
| `plain` | 0 | 0 |
| `dynamic_no_ratio` | 12 | 12 |

## Rebuild 路径开销

4 个 TP rank 求和的旧版 rebuild profile：

| event | plain | dynamic_no_ratio | delta |
| --- | ---: | ---: | ---: |
| `csd_worker:verify_total` | 38749.1 ms / 9416 | 37790.7 ms / 9176 | -958.4 ms |
| `csd_verify:accept_index_predict_to_cpu_sync` | 25582.3 ms / 9416 | 25067.1 ms / 9176 | -515.2 ms |
| `csd_verify:tree_spec_sampling_kernel` | 1147.5 ms / 9416 | 1033.9 ms / 9176 | -113.6 ms |
| `csd_worker:maybe_start_async_rebuild_total` | 82.3 ms / 4708 | 1836.7 ms / 9176 | +1754.4 ms |
| `csd_worker:async_rebuild_started` | 0 | 61.8 ms / 12 | +61.8 ms |
| `csd_worker:maybe_apply_async_rebuild` | 211.3 ms / 4708 | 324.4 ms / 4600 | +113.1 ms |
| `csd_worker:async_rebuild_applied` | 0 | 0.7 ms / 12 | +0.7 ms |

拆分后的 TP0 profile 进一步说明，overlap 后剩下的 rebuild 管理开销很小：

| event | plain | dynamic_no_ratio | p20_no_ratio |
| --- | ---: | ---: | ---: |
| `csd_worker:maybe_start_async_rebuild_total` | 6.9 ms / 508 / 0.014 ms | 46.6 ms / 511 / 0.091 ms | 43.1 ms / 565 / 0.076 ms |
| `csd_rebuild_apply:materialize_and_swap` | 0.0 ms / 0 | 9.2 ms / 2 / 4.584 ms | 9.2 ms / 2 / 4.593 ms |

这里的 `apply` 不是完整 rebuild，而是后台任务完成后的结果应用：`future.result()` 取回 CPU payload，然后在 verify 边界 materialize CUDA hash table 并 swap。真正完整的 CPU 构表已经在后台线程执行，所以 trace 中看到的 apply 很小。`maybe_start_async_rebuild_total` 里仍有一部分同步成本，主要是 counter threshold check、flush delta、executor submit；但这个量级不足以解释 LightEval 中 dynamic 的整体速度差异。

## Verify step 细分

`max_new_tokens=2048` 的 verify step profile：

| event | plain | dynamic_no_ratio | p20_no_ratio |
| --- | ---: | ---: | ---: |
| `csd_worker:target_forward_verify` | 8197.3 ms / 1524 / 5.379 ms | 8223.9 ms / 1533 / 5.365 ms | 9427.3 ms / 1695 / 5.562 ms |
| `csd_worker:verify_total` | 3865.3 ms / 1016 / 3.805 ms | 3738.9 ms / 1022 / 3.659 ms | 4491.0 ms / 1130 / 3.974 ms |
| `csd_verify:target_probs_softmax_topk_topp` | 316.7 ms / 1016 / 0.312 ms | 312.4 ms / 1022 / 0.306 ms | 339.6 ms / 1130 / 0.301 ms |
| `csd_verify:tree_spec_sampling_kernel` | 118.3 ms / 1016 / 0.116 ms | 109.5 ms / 1022 / 0.107 ms | 119.0 ms / 1130 / 0.105 ms |
| `csd_verify:accept_index_predict_to_cpu_sync` | 3025.6 ms / 1016 / 2.978 ms | 2907.5 ms / 1022 / 2.845 ms | 3596.0 ms / 1130 / 3.182 ms |
| `csd_worker:post_verify_mamba_update` | 371.9 ms / 1016 / 0.366 ms | 540.9 ms / 1022 / 0.529 ms | 546.0 ms / 1130 / 0.483 ms |
| `csd_worker:post_verify_gather_logits_hidden` | 62.8 ms / 1016 / 0.062 ms | 86.0 ms / 1022 / 0.084 ms | 99.7 ms / 1130 / 0.088 ms |
| `cudaMemcpyAsync` | 3406.2 ms / 20954 | 3242.7 ms / 22348 | 4064.7 ms / 24976 |
| `cudaGraphLaunch` | 1354.2 ms / 1523 | 1397.8 ms / 1532 | 1518.8 ms / 1694 |

`max_new_tokens=4096` 的 profile 也从 TP0 trace 直接聚合核对了一遍，用来确认同样的现象是否稳定：

| event | plain | plain_entropy_p20 | dynamic_no_ratio | p20_no_ratio |
| --- | ---: | ---: | ---: | ---: |
| `csd_worker:verify_total` | 8821.0 ms / 2152 / 4.099 ms | 8558.2 ms / 2138 / 4.003 ms | 8831.7 ms / 2152 / 4.104 ms | 9625.0 ms / 2366 / 4.068 ms |
| `csd_verify:accept_index_predict_to_cpu_sync` | 5809.0 ms / 2152 / 2.699 ms | 5674.5 ms / 2138 / 2.654 ms | 5835.5 ms / 2152 / 2.712 ms | 6421.0 ms / 2366 / 2.714 ms |
| `csd_verify:target_probs_softmax_topk_topp` | 648.2 ms / 2152 / 0.301 ms | 615.9 ms / 2138 / 0.288 ms | 653.9 ms / 2152 / 0.304 ms | 687.2 ms / 2366 / 0.290 ms |
| `csd_verify:tree_spec_sampling_kernel` | 262.1 ms / 2152 / 0.122 ms | 248.8 ms / 2138 / 0.116 ms | 242.0 ms / 2152 / 0.112 ms | 243.6 ms / 2366 / 0.103 ms |
| `cudaMemcpyAsync` | 6392.2 ms / 44380 | 6290.5 ms / 46375 | 6590.8 ms / 45465 | 7207.8 ms / 52834 |
| `cudaGraphLaunch` | 2936.0 ms / 3227 | 2893.6 ms / 3206 | 2914.8 ms / 3227 | 3231.4 ms / 3548 |

这里的 `csd_worker:verify_total` 不是单个 CUDA 算子，而是 Python 侧 `spec_info.verify()` 外层的 record range。它包住了 target logits 后处理、`tree_speculative_sampling_target_only` kernel、accept index/predict 读回 CPU、Python 更新 request 等操作。profile 中单独能看到的关键 GPU/runtime 项包括 `csd_verify:tree_spec_sampling_kernel`、`cudaGraphLaunch` 和 `cudaMemcpyAsync`；softmax/top-k/top-p 也在 verify 内，但会展开成多个底层 CUDA kernel，当前表里没有逐个列 kernel 名。

`accept_index_predict_to_cpu_sync` 也不是一个纯 CPU 算子。代码里它只包住 `accept_index.tolist()` 和 `predict.tolist()`，但这两步会把 GPU 上的 sampling 结果同步回 CPU，因此它经常吸收前面 GPU work 的等待时间。换句话说，`accept_index_predict_to_cpu_sync` 大，表示 verify 路径里有明显 CPU-visible synchronization point，不代表 Python list conversion 本身就消耗了 3 ms。

这个 profile 修正了一个早期判断：`dynamic_no_ratio` 并没有让每次 verify 变贵。它的 `verify_total` per-call 甚至低于 `plain`（3.659 ms vs 3.805 ms），`accept_index_predict_to_cpu_sync` 也低于 `plain`（2.845 ms vs 2.978 ms）。dynamic 的可见额外成本主要在 rebuild 管理和 post-verify shape-dependent work，例如 `post_verify_mamba_update`、`post_verify_gather_logits_hidden`。

`p20_no_ratio` 的异常点主要在 verify 形态：verify 次数从 1016 增加到 1130，`cudaGraphLaunch` 和 `cudaMemcpyAsync` 次数也同步增加；同时这个 2048 profile 里 per-call `accept_index_predict_to_cpu_sync` 从 2.978 ms 增加到 3.182 ms。4096 profile 里这个 per-call 增量较小（2.699 ms 到 2.714 ms），但方向仍然偏高。由于 accept sync 是同步点，这个 per-call 增量更可能反映 P20 改变 accept/reject 后导致上游 GPU work、数据读回和 batch shape 等待时间变重，而不是某个单独的 accept Python 操作突然变慢。

`p20_no_ratio` 的问题不同：它的 verify / target-forward 次数更多。和 `plain` 比，target verify forward 从 1524 次增到 1695 次，verify 从 1016 次增到 1130 次；因此 CPU sync、`cudaMemcpyAsync`、`cudaGraphLaunch` 总量都上升。也就是说 P20 的慢主要不是 entropy kernel 直接慢，而是 entropy gate 改变接受/拒绝路径之后，verify loop 形态变了。

## Entropy 计算开销

entropy gate 的实现不在 Python 侧单独执行，而是在 `tree_speculative_sampling_target_only` CUDA kernel 内部执行：

1. normal accept 失败后，先走 CSD table hit 和 logit gate；
2. 只有 `csd_force_accept == true` 且 `csd_force_accept_entropy_threshold >= 0` 时，才调用 `CsdComputeTargetEntropy`；
3. `CsdComputeTargetEntropy` 扫描当前 target probability vector，计算 `-sum(p * log(p))`；
4. entropy 小于等于阈值时保留 force accept，否则取消这次 CSD force accept。

因此普通 profiler 里不会出现单独的 `entropy_compute` event，只能看到整个 `csd_verify:tree_spec_sampling_kernel`。为了估算直接开销，我们做了 microbench：构造同一批输入，让每个 batch element 都满足 normal accept 失败、CSD table hit、logit gate 通过，从而每个 CUDA block 至少进入一次 entropy check；然后只切换 entropy gate 开/关。

这里的 `batch blocks` 表示一次 kernel launch 中的 batch elements / CUDA blocks 数。它不是串行循环次数；多个 block 并行执行，所以不能把总 delta 除以 block 数来解释“每个请求的开销”。下面的 `delta` 是该 batch size 下 **一次 tree sampling kernel launch 的总耗时增量**。

microbench 输入不是一段真实 prompt 序列，而是直接构造 `tree_speculative_sampling_target_only` kernel 需要的张量。每个 batch element 构造一个最小 speculative tree：

```text
candidates              = [root_token, draft_token=3, unused_sibling=4]
retrive_index           = [0, 1, 2]
retrive_next_token      = [1, -1, -1]
retrive_next_sibling    = [-1, -1, -1]
target_probs shape      = [batch_blocks, num_draft_tokens=3, vocab_size]
target_logits shape     = [batch_blocks, num_draft_tokens=3, vocab_size]
```

也就是说，每个 batch element 只有一个实际会被访问的 speculative candidate：`draft_token=3`。我们把 residual sampling 构造成选出 `resampled_token=2`，并把 `(3, 2)` 放进 CSD hash table，同时让 logit gate 通过。这样每个 batch element 的路径是：

```text
访问 draft_token=3
normal accept 失败
residual sampling 得到 token=2
CSD pair (3, 2) 命中
logit gate 通过
进入 CsdComputeTargetEntropy 一次
```

因此预计 entropy 计算次数是：

```text
entropy calls per kernel launch = batch_blocks
```

例如 `batch_blocks=48` 时，一次 kernel launch 中预计有 48 个 CUDA blocks 各算一次 entropy；不是 48 个 block 乘以多个 speculative token，也不是对真实生成序列里的每个 token 都算。真实 decode 中，一个 batch element 如果 tree 上多个 rejected candidate 都满足 CSD force-accept 条件，理论上可能多次进入 entropy；但这个 microbench 为了隔离单次 entropy 直接开销，刻意构造成每个 batch element 只触发一次。

| vocab d | batch blocks | no entropy | entropy on | kernel delta |
| ---: | ---: | ---: | ---: | ---: |
| 32768 | 1 | 40.04 us | 42.80 us | +2.76 us |
| 32768 | 4 | 40.05 us | 44.41 us | +4.37 us |
| 32768 | 16 | 38.62 us | 40.80 us | +2.18 us |
| 32768 | 48 | 39.95 us | 42.20 us | +2.25 us |
| 151936 | 1 | 142.41 us | 158.69 us | +16.28 us |
| 151936 | 4 | 142.78 us | 157.08 us | +14.30 us |
| 151936 | 16 | 206.69 us | 223.72 us | +17.03 us |
| 151936 | 48 | 236.83 us | 252.60 us | +15.77 us |

这可以看作一个偏压力测试的直接开销估计：真实运行中不是每个 rejected token 都会算 entropy，只有 CSD force-accept 候选才会算；但在 microbench 中，每个 batch block 都被构造成会触发 entropy check。在 Qwen3.5 真实 vocab 规模附近，单次 tree sampling kernel launch 的总增量大约是 14-17 us，约等于 tree sampling kernel 自身 5%-10% 的额外开销。因此 entropy 直接计算不是免费的。

但这个增量不能直接乘到完整 verify 上。完整 profile 里 `tree_spec_sampling_kernel` 每次约 0.10-0.12 ms，而 `verify_total` 每次约 4 ms；`verify_total` 包含 softmax/top-k/top-p、tree sampling、accept index 预测、CPU sync 和采样结果处理等一整段流程，其中 `accept_index_predict_to_cpu_sync` 本身就接近 3 ms/call。也就是说，entropy 的直接算子开销主要落在 tree sampling kernel 内，对完整 verify 的直接占比仍然很小；更大的端到端影响来自 entropy/P20 改变 accept/reject 行为后，让 verify loop 次数、target forward 次数和 post-verify/scheduler work 增加。

## Profile 结论

1. 异步 rebuild 和 filtered-key 优化已经把 rebuild 从主要热路径里移开。`apply` 只剩 materialize/swap，profile 中量级很小。
2. `dynamic_no_ratio` 的 CSD verify 本身没有变慢；短 profile 中它和 `plain` 的 verify 次数、verify_total、accept sync 都非常接近。
3. dynamic 在完整 LightEval 里慢或收益不明显，更可能来自生成分布和调度形态：长尾样本变多、请求结束时机改变、post-verify shape 变化、连续批处理有效 batch 形态变差。
4. `p20_no_ratio` 的慢主要来自 verify loop 变长，而不是 entropy kernel 直接慢。entropy 直接开销在 microbench 中只有十几微秒量级。
5. 因此后续优化重点不应该继续盯着 rebuild/filter 本身，而应该看 dynamic 如何影响接受长度分布、max_new hit、verify 次数、post-verify 更新和 batch scheduler 的尾部行为。
