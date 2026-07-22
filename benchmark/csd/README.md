# MTP + CSD benchmark

- 最终技术总结：[docs/MTP_CSD_V0510_FINAL.md](docs/MTP_CSD_V0510_FINAL.md)
- 六数据集评测入口：[eval/run_mtp515_v0510_six_datasets_mr48.sh](eval/run_mtp515_v0510_six_datasets_mr48.sh)
- Calibration 实现：[calibration/run_redpajama_calibration_515.sh](calibration/run_redpajama_calibration_515.sh)；六数据集入口通过 `RUN_CALIBRATION=1` 调用它
- 最终原始结果：`runs/20260721_mtp515_v0510_five_methods_mr48_v3/`
- 校准表与固定数据：`assets/`
- 随实验提交的 LightEval 修改：`lighteval/`

## 运行

默认使用 GPU 4–7、TP=4、`max-running-requests=48`：

```bash
conda activate sglang
bash benchmark/csd/eval/run_mtp515_v0510_six_datasets_mr48.sh
```

常用覆盖项：

```bash
CUDA_DEVICES=0,1,2,3 \
RUN_STAMP=my_run \
bash benchmark/csd/eval/run_mtp515_v0510_six_datasets_mr48.sh
```

评测脚本是 **5 种方法 × 6 个数据集**：依次运行 Auto、MTP、MTP + static CSD、MTP + dynamic CSD、MTP + dynamic CSD + entropy gate，并覆盖 LCB、AIME25、Math500、GSM8K、APPS 和 TACO。`eval/libexec/` 中是评测入口使用的内部执行模块，不是独立实验入口。

每次正式运行会直接生成清晰的数据集目录：

```text
runs/<RUN_STAMP>/
├── lighteval/
└── code_ood/
    ├── apps/
    │   ├── results.jsonl
    │   ├── summary.md
    │   ├── answers/
    │   └── logs/
    └── taco/
        ├── results.jsonl
        ├── summary.md
        ├── answers/
        └── logs/
```

## Calibration 与 table 路径

单独生成 calibration table：

```bash
conda activate sglang

RUN_STAMP=$(date +%Y%m%d_%H%M%S)_mtp515_v0510
RUN_ROOT="$PWD/benchmark/csd/runs/${RUN_STAMP}"
CALIBRATION_TABLE="$RUN_ROOT/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"

OUT_DIR="$RUN_ROOT/calibration" \
CSD_TABLE_PATH="$CALIBRATION_TABLE" \
CUDA_DEVICES=4,5,6,7 \
TP_SIZE=4 \
MAX_RUNNING_REQUESTS=48 \
bash benchmark/csd/calibration/run_redpajama_calibration_515.sh
```

随后评测时，必须用 `TABLE` 显式覆盖测试 table 路径：

```bash
TABLE="$CALIBRATION_TABLE" \
RUN_STAMP="$RUN_STAMP" \
RUN_ROOT="$RUN_ROOT" \
bash benchmark/csd/eval/run_mtp515_v0510_six_datasets_mr48.sh
```

主评测入口会把 `TABLE` 继续传给 LightEval 和 APPS/TACO runner 的 `PLAIN_CSD_TABLE_PATH`，因此 static CSD、dynamic CSD 和 dynamic CSD + entropy gate 使用的是同一张新表。启动前会检查文件存在，并将路径及 SHA-256 写入 `version_compare_config.txt`。

也可以一条命令完成 calibration 并继续六数据集评测：

```bash
conda activate sglang

RUN_STAMP=$(date +%Y%m%d_%H%M%S)_mtp515_v0510
RUN_ROOT="$PWD/benchmark/csd/runs/${RUN_STAMP}"
CALIBRATION_TABLE="$RUN_ROOT/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"

RUN_CALIBRATION=1 \
RUN_STAMP="$RUN_STAMP" \
RUN_ROOT="$RUN_ROOT" \
CALIBRATION_TABLE_PATH="$CALIBRATION_TABLE" \
bash benchmark/csd/eval/run_mtp515_v0510_six_datasets_mr48.sh
```

当 `RUN_CALIBRATION=1` 时，入口脚本会在 calibration 完成后强制令 `TABLE="$CALIBRATION_TABLE_PATH"`，后续测试不会继续使用 `assets/calibration/` 中的旧表，也不会覆盖旧表。只生成新表而不评测时，再加 `CALIBRATION_ONLY=1`。

calibration 保留了生成当前 table 的原始配置：Qwen3.5-35B-A3B、GPU 4–7、TP4、RedPajama 六领域各 1000 条、MTP 5/1/5、temperature=1.0、top-p=1.0、max-new-tokens=1024、parallel=8、ungated record、frequency threshold=6、prob ratio=0.01、delta capacity=16777216。记录阶段使用 `--speculative-csd-force-accept-disabled`，因此只收集 pair，不用 CSD 改写输出。

## 新增的 ServerArgs

以下参数定义在 `python/sglang/srt/server_args.py`。除特别说明外，它们只在开启 `--speculative-csd` 后生效。

| 参数 | 默认值 | 作用与约束 |
| --- | --- | --- |
| `--speculative-csd` | 关闭 | CSD 总开关，在 speculative verify 阶段启用查表、记录和 force accept 逻辑。必须同时提供静态 table，或开启 dynamic update。 |
| `--speculative-csd-table-path PATH` | `None` | 加载静态 pair-frequency table。Static CSD 和在线 Dynamic CSD 都使用它；纯 calibration 可以不提供。 |
| `--speculative-csd-freq-threshold N` | `6` | table 加载或 rebuild 时的最低 pair 频次；必须至少为 1。`frequency` 策略直接用它过滤低频 pair。 |
| `--speculative-csd-key-selection-strategy STRATEGY` | `frequency` | hash key 的筛选/排序策略。支持 `frequency`、`count_squared_over_total` 和 `above_uniform_share`。 |
| `--speculative-csd-score-threshold X` | `0.0` | `count_squared_over_total` 策略的最低分数，分数为 `count(pair)^2 / count(draft, *)`；必须非负。 |
| `--speculative-csd-prob-ratio R` | `0.01` | force accept 的 target-logit gate，要求 `p(draft) / p(target) >= R`；范围为 `(0, 1]`。本轮评测显式使用 `0.3`，calibration metadata 记录为 `0.01`。 |
| `--speculative-csd-dynamic-update` | 关闭 | 在推理期间把合格的 rejected `(draft, target)` pair 写入 GPU delta buffer，并允许触发在线 rebuild。 |
| `--speculative-csd-dynamic-update-ignore-prob-ratio` | 关闭 | 收集 pair 时忽略 prob-ratio gate，得到 ungated 数据；force accept 仍然执行 prob-ratio gate。必须与 dynamic update 一起使用。 |
| `--speculative-csd-delta-save-path PATH` | `None` | 将动态收集的 delta pair 持久化到指定路径。不设置时 pair 仍可用于当前进程 rebuild，但不会自动保存。 |
| `--speculative-csd-delta-capacity N` | 按模式设置 | GPU delta buffer 的 pair 容量。纯 record 默认 `1<<20`；加载 table 的在线更新默认 `max(2 × rebuild_threshold, 1)`。每个 pair 是一个 int64，显存约为 `8N` 字节/TP rank；本轮 calibration 显式使用 `16777216`，约 128 MiB/rank。 |
| `--speculative-csd-rebuild-threshold N` | `4096` | 未处理 delta pair 达到该数量后启动后台 rebuild；`<=0` 关闭在线 rebuild。它决定触发频率，不是 table 容量。 |
| `--speculative-csd-rebuild-top-keep X` | `None` | 限制每次本地 rebuild 保留的高频项。`0` 表示不限制，`(0,1]` 表示保留比例，`>1` 表示绝对 top-K。旧别名是 `--speculative-csd-rebuild-top-freq-ratio`。 |
| `--speculative-csd-force-accept-disabled` | 关闭 | 禁止 CSD force accept，但继续记录 pair 和指标。calibration 必须开启它，以免采表过程改变模型输出。 |
| `--speculative-csd-force-accept-entropy-threshold H` | `-1` | entropy gate；target distribution entropy 高于 `H` 时禁止 CSD force accept。`-1` 表示关闭，其他值必须非负。本轮 entropy-gate 配置为 `1.5638477802276611`。 |

常用模式组合：

| 模式 | 必需参数 |
| --- | --- |
| Static CSD | `--speculative-csd --speculative-csd-table-path TABLE` |
| Calibration/record | `--speculative-csd --speculative-csd-dynamic-update --speculative-csd-force-accept-disabled`，本轮另加 `--speculative-csd-dynamic-update-ignore-prob-ratio` |
| Dynamic CSD | Static CSD 参数 + `--speculative-csd-dynamic-update --speculative-csd-rebuild-threshold 4096` |
| Dynamic + entropy gate | Dynamic CSD 参数 + `--speculative-csd-force-accept-entropy-threshold 1.5638477802276611` |

## 目录约束

- `runs/` 只放当前正式运行；新的正式复现完成后整体替换，不在这里累计历史目录。
- `assets/` 放复现必需且不随运行变化的输入。
- `docs/` 只保留最终总结。
- 临时日志、失败启动和探索性脚本不要提交到本目录。
