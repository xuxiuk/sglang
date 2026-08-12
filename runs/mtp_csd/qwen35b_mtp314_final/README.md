# Qwen3.5-35B-A3B MTP 3-1-4 最终结果

主任务统一入口为 `run_all.sh`，输出固定为 `results/<task>/`，不再按 GPU 0-3/4-7
划分。GPU 分组只是并行调度手段，不是实验维度。

| 目录 | 内容 |
| --- | --- |
| `results/lcb_avg4` | Auto、bare MTP、plain、dynamic、dynamic + entropy gate |
| `results/aime25_avg16` | 同上，avg@16/pass@16 |
| `results/math500_avg4` | 同上，avg@4/pass@4；entropy方法使用p30阈值 |
| `results/gsm8k_avg4` | 同上，avg@4/pass@4 |
| `results/apps` | APPS n=4，四种投机方法 |
| `results/taco` | TACO n=4，四种投机方法；保留结果使用 40960-token cap |

主任务每个目录中的 `artifacts/`、`logs/` 和
`results/classic_tree_shape_sweep.jsonl` 与 `run_all.sh` 的输出位置一致。APPS/TACO 由
`run_apps_taco_n4.sh all` 生成，直接写入 `results/apps` 和 `results/taco`。

`calibration/` 保存正式评测加载的 MTP table；`vendor/` 是冻结的评测依赖。
