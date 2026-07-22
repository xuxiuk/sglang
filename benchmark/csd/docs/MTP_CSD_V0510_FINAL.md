# SGLang v0.5.10：MTP + CSD 六数据集最终报告

## 1. 结论

本轮 30 项任务全部完成：LightEval 4 个数据集 × 5 种方法共 20 项，APPS/TACO 2 个数据集 × 5 种方法共 10 项。最终完成时间为 2026-07-22 01:20:34（Asia/Shanghai），结果中未发现 traceback 或任务失败。

主要结论：

1. 原生 MTP 相对 Auto 的平均吞吐加速为 **1.41×**（LightEval）和 **1.50×**（APPS/TACO）。
2. Static CSD 将平均接受长度由 MTP 的 **3.89** 提高到 **4.46**，LightEval 平均加速提高到 **1.66×**；在 APPS/TACO 上为 **1.84×**。
3. Dynamic CSD 的接受长度最高：LightEval 平均 **4.62**，APPS/TACO 平均 **4.49**；APPS、TACO 分别达到 **2.02×、2.00×**。
4. Entropy gate 降低了部分激进替换，精度比无 gate 的 Dynamic CSD 更接近 Auto，但吞吐略低。它不是本轮速度最优项。
5. 综合本轮数据，**Static CSD 是更稳健的默认配置，Dynamic CSD 是代码长输出场景的吞吐优先配置**；是否使用 entropy gate 应由精度约束决定。

## 2. 实验配置

| 项目 | 配置 |
| --- | --- |
| 仓库 | SGLang `v0.5.10-28-gfe46960cf` |
| 模型 | `/root/model/Qwen3.5-35B-A3B` |
| Python | `/root/miniconda3/envs/sglang/bin/python` |
| PyTorch | `2.9.1+cu128` |
| GPU | 4–7，TP=4 |
| 并发 | `max-running-requests=48`；APPS/TACO client parallel=24 |
| MTP | steps=5，top-k=1，draft tokens=5（记作 5/1/5） |
| 采样 | temperature=1.0，top_p=0.95，top_k=20，presence_penalty=1.5，seed=1234 |
| CSD | frequency threshold=6，prob ratio=0.3，rebuild threshold=4096 |
| Entropy gate | threshold=`1.5638477802276611` |
| 最大生成长度 | LCB/AIME25/Math500=81920；GSM8K=32768；APPS/TACO=40960 |
| 调度 | `mamba-scheduler-strategy=no_buffer`，确定性推理关闭 |

5/1/5 表示 5 层顺序 draft、每层 top-1、最多 5 个 draft token。一次 verify 还会产生一个 target reward token，因此“接受长度”的理论上限是 6；这不表示实际启动参数被误写成 5/1/6。

方法名称映射如下：

| 报告名称 | 原始结果名称 | 含义 |
| --- | --- | --- |
| Auto | `auto` | 不使用投机解码 |
| MTP | `eagle` | 原生 MTP/NEXTN 路径，不使用 CSD |
| MTP + CSD | `plain` | 加载静态 calibration table |
| MTP + Dynamic CSD | `dynamic` | 静态表 + 在线更新/rebuild |
| MTP + Dynamic CSD + Entropy Gate | `dynamic_entropy_p20_ignore_ratio` | Dynamic CSD + entropy threshold；报告不再使用含混的 “P20” 名称 |

## 3. LightEval：精度、速度与投机指标

加速比均以同一数据集的 Auto 吞吐为分母。

| 数据集 | 方法 | 精度 | 输出 tok/s | 加速比 | 平均接受长度 | 投机成功率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LCB | Auto | 0.7429 | 1676.6 | 1.00× | — | — |
| LCB | MTP | 0.7143 | 2491.6 | 1.49× | 3.4145 | 47.64% |
| LCB | MTP + CSD | 0.7371 | 3175.2 | 1.89× | 4.1040 | 62.92% |
| LCB | MTP + Dynamic CSD | 0.7143 | 3220.1 | 1.92× | 4.4718 | 72.59% |
| LCB | MTP + Dynamic CSD + Entropy Gate | 0.7429 | 3070.8 | 1.83× | 4.1982 | 64.55% |
| AIME25 | Auto | 0.8667 | 1275.9 | 1.00× | — | — |
| AIME25 | MTP | 0.9000 | 2129.7 | 1.67× | 3.7943 | 54.79% |
| AIME25 | MTP + CSD | 0.9333 | 2501.9 | 1.96× | 4.4870 | 69.98% |
| AIME25 | MTP + Dynamic CSD | 0.9000 | 2511.5 | 1.97× | 4.5696 | 72.16% |
| AIME25 | MTP + Dynamic CSD + Entropy Gate | 0.9000 | 2531.0 | 1.98× | 4.4871 | 70.92% |
| Math500 | Auto | 0.8580 | 3176.6 | 1.00× | — | — |
| Math500 | MTP | 0.8340 | 4140.1 | 1.30× | 4.0147 | 56.15% |
| Math500 | MTP + CSD | 0.8560 | 4838.7 | 1.52× | 4.6087 | 70.06% |
| Math500 | MTP + Dynamic CSD | 0.8340 | 4571.8 | 1.44× | 4.7343 | 74.94% |
| Math500 | MTP + Dynamic CSD + Entropy Gate | 0.8540 | 4680.0 | 1.47× | 4.6676 | 71.81% |
| GSM8K | Auto | 0.8893 | 3667.1 | 1.00× | — | — |
| GSM8K | MTP | 0.9067 | 4335.0 | 1.18× | 4.3170 | 58.78% |
| GSM8K | MTP + CSD | 0.9098 | 4663.8 | 1.27× | 4.6323 | 67.83% |
| GSM8K | MTP + Dynamic CSD | 0.9090 | 4497.5 | 1.23× | 4.7102 | 68.77% |
| GSM8K | MTP + Dynamic CSD + Entropy Gate | 0.9083 | 4569.6 | 1.25× | 4.7109 | 68.31% |

四数据集算术平均：

| 方法 | 平均精度 | 平均加速比 | 平均接受长度 |
| --- | ---: | ---: | ---: |
| Auto | 0.8392 | 1.00× | — |
| MTP | 0.8388 | 1.41× | 3.8851 |
| MTP + CSD | **0.8591** | **1.66×** | 4.4580 |
| MTP + Dynamic CSD | 0.8393 | 1.64× | **4.6215** |
| MTP + Dynamic CSD + Entropy Gate | 0.8513 | 1.63× | 4.5160 |

这里的平均精度只是四个任务分数的简单算术平均，任务样本量不同，因此仅用于快速总览，不应替代逐数据集比较。采样推理关闭了 deterministic inference；即使 seed 相同，不同解码路径也可能产生不同输出长度和精度，不能把单次采样的小幅差异解释为严格回归。

## 4. APPS/TACO：长代码生成结果

本轮按既定约定不报告 APPS/TACO 执行精度，只报告速度和接受长度。

| 数据集 | 方法 | 输出 tok/s | 相对 Auto | 相对 Static CSD | 平均接受长度 | 平均输出 token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| APPS | Auto | 1923.4 | 1.00× | 0.55× | — | 20583.7 |
| APPS | MTP | 2887.7 | 1.50× | 0.82× | 3.354 | 20353.3 |
| APPS | MTP + CSD | 3511.1 | 1.83× | 1.00× | 4.068 | 18528.8 |
| APPS | MTP + Dynamic CSD | **3884.2** | **2.02×** | **1.11×** | **4.562** | 18405.5 |
| APPS | MTP + Dynamic CSD + Entropy Gate | 3680.1 | 1.91× | 1.05× | 4.362 | 19375.5 |
| TACO | Auto | 1918.4 | 1.00× | 0.54× | — | 19665.5 |
| TACO | MTP | 2878.7 | 1.50× | 0.81× | 3.314 | 18977.0 |
| TACO | MTP + CSD | 3563.8 | 1.86× | 1.00× | 4.000 | 16146.1 |
| TACO | MTP + Dynamic CSD | **3842.6** | **2.00×** | **1.08×** | **4.417** | 15929.7 |
| TACO | MTP + Dynamic CSD + Entropy Gate | 3663.1 | 1.91× | 1.03× | 4.245 | 16611.6 |

APPS/TACO 的不同方法生成 token 总量并不完全一致，因此 tok/s 是各方法真实工作负载下的端到端吞吐，不是控制相同输出 token 后的纯 kernel microbenchmark。Dynamic CSD 在两个长输出任务上仍稳定超过 Static CSD 7.8%–10.6%，说明在线更新在长请求中有足够的积累窗口来摊薄 rebuild 成本。

## 5. 指标来源和聚合方法

最终运行目录：

```text
benchmark/csd/runs/20260721_mtp515_v0510_five_methods_mr48_v3/
```

原始汇总文件：

- LightEval：`lighteval/results/classic_tree_shape_sweep.jsonl`，20 行，一行对应一个“方法 × 数据集”。
- APPS：`code_ood/apps/results.jsonl`，5 行，一行对应一种方法；对应的 `summary.md`、answers 和 logs 在同一目录。源文件有 2000 条，本轮按 `offset=0, NUM_EXAMPLES=1000` 使用前 1000 条。
- TACO：`code_ood/taco/results.jsonl`，5 行，一行对应一种方法；对应的 `summary.md`、answers 和 logs 在同一目录。源文件和本轮评测均为 1000 条。
- 原始的 10 行合并汇总保留为 `code_ood/apps_taco_combined_results.jsonl`，用于校验整理前后内容一致。
- 完整配置：`version_compare_config.txt`。
- 各请求输出、逐请求 metrics、server/bench/decode logs 均保留在同一运行目录中。

本报告字段对应关系：

| 报告字段 | JSONL 字段/公式 |
| --- | --- |
| 精度 | LightEval 行的 `accuracy` |
| 输出 tok/s | `throughput` |
| 加速比 | 当前行 `throughput / 同数据集 auto.throughput` |
| 平均接受长度 | 行内 `accept_length`；LightEval 中等于各请求接受长度的算术平均 |
| 投机成功率 | LightEval 行内 `spec_success_rate = total_spec_accept_token_num / total_spec_draft_token_num` |
| 平均输出 token | `avg_completion_tokens` |

每轮 verify 除被接受的 draft token 外还会正常生成 1 个 target reward token。因此单请求接受长度为：

```text
1 + spec_accept_token_num / spec_verify_ct
```

LightEval 表中的 `accept_length` 是上述单请求值的非加权平均。若需要按 verify 次数加权，应使用：

```text
1 + total_spec_accept_token_num / total_spec_verify_ct
```

因此接受长度包含 reward token，而投机成功率只衡量被接受 draft token 占全部 draft token 的比例；二者含义不同。Auto 没有投机过程，报告统一记为“—”，不使用不同 runner 中的 0/1 占位值。

## 6. 复现入口与完整性

唯一正式入口：

```bash
conda activate sglang
bash benchmark/csd/eval/run_mtp515_v0510_six_datasets_mr48.sh
```

复现所需 calibration table SHA-256：

```text
763716f143d688d79eed06a7a2c52b2ec726d6ec87b6aed461dd3286521c3f61
```

评测入口会先校验 calibration table 和两个代码数据集存在，再记录 Git、Python、PyTorch、SGLang、kernel 路径与所有生成参数。`eval/libexec/` 仅承载该入口调用的内部矩阵执行逻辑。生成当前 RedPajama table 的代码保留在 `benchmark/csd/calibration/`，入口为 `run_redpajama_calibration_515.sh`。
