# Qwen3.5-397B-A17B-FP8 MTP 3-1-4 最终结果

统一入口为 `run_all.sh`，四个任务均输出到 `results/<task>/`。AIME、LCB、GSM8K使用
`CSD_PROB_RATIO=0.3`；Math500使用精度实验选定的 `CSD_PROB_RATIO=0.4`。

当前 Math500 最终口径为：

- bare MTP：沿用原四任务结果，ratio参数对bare无效；
- plain、dynamic、dynamic + entropy gate：使用ratio=0.4补跑结果。

`results/math500_avg4/results/source_summaries/` 保留两轮来源JSONL用于审计：旧ratio=0.3
文件只提供最终bare行，两个ratio=0.4文件提供三个最终CSD方法。正式精度artifact和日志已经
按上述口径放在 `results/math500_avg4/{artifacts,logs}`。

再次运行当前 `run_all.sh all` 会原生得到相同口径：脚本只在Math500分支将ratio切换为0.4，
其他任务仍为0.3。
