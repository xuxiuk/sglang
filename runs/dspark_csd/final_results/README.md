# DeepSeek-V4-Flash-DSpark 最终结果

本目录按公共矩阵脚本的原生输出布局组织：`<method>/<task>/`。统一入口为
`run_all.sh`，默认直接写入本目录；设置 `RUN_ROOT_OVERRIDE` 可写入新的空目录。

| 任务 | 重复次数 | 方法 | 当前结果来源 |
| --- | ---: | --- | --- |
| `lcb_avg4` | 4 | bare/plain/dynamic/entropy | `static_four_methods_four_tasks_20260808_021418/01_lcb` |
| `aime25_avg16` | 16 | bare/plain/dynamic/entropy | `aime_pass16_final_20260810` |
| `math500_avg4` | 4 | bare/plain/dynamic/entropy | `static_four_methods_four_tasks_20260808_021418/02_aime_math_gsm8k` |
| `gsm8k_avg4` | 4 | bare | 原 `02_aime_math_gsm8k` |
| `gsm8k_avg4` | 4 | plain/dynamic/entropy | 原 `03_gsm8k_remaining` 补跑 |

`run_all.sh` 使用同一套配置依次运行 LCB，以及 AIME/Math500/GSM8K。AIME 已固定为
`avg@16/pass@16`；其他三个任务为 `avg@4/pass@4`。旧的 AIME avg@4 和独立补跑目录不属于
最终结果。

每个 `<method>/<task>/eval/result.json` 保存精度结果，`metrics/task_metrics.json` 保存
吞吐、接受长度和 CSD counters，`eval/tracker/` 保存评测明细。

`source_configs/`保存被合并的原始轮次配置，仅用于来源审计。重新执行统一入口时会在根目录
生成`config_lcb.txt`和`config_aime16_math_gsm8k.txt`，分别对应脚本的两个执行阶段。
