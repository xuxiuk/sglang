# MTP CSD 拒绝事件 Trace

当前实现只支持 MTP/EAGLE `topk=1`。Trace 读取 verifier 已产生的 draft、residual、
target distribution 和接受长度，不修改 CUDA operator，也不改变生成结果。

## 启动参数

```text
--speculative-csd-rejection-trace
--speculative-csd-rejection-trace-dir <directory>
--speculative-csd-rejection-trace-capacity 65536
```

开启 trace 时还必须开启：

```text
--speculative-csd
--speculative-csd-force-accept-disabled
--speculative-eagle-topk 1
```

`capacity`表示尚未写入磁盘的事件数。默认完整记录，不做抽样。缓冲区满时不覆盖旧
事件，而是在`csd_trace_dropped_ct`中计数。读取`/server_info`时会先等待后台 writer
写完，确保指标和 JSONL 一致。

## 输出

每个独立 attention lane 生成：

```text
metadata.rank<R>.json
requests.rank<R>.jsonl
events.rank<R>.jsonl
```

`requests`按`request_id`保存一次 prompt token、截至最后一个 trace 事件的生成 token
以及实际 sampling 参数。事件中的上下文可严格恢复为：

```python
context = prompt_token_ids + generated_token_ids[:generated_position]
```

`events`保存 draft/residual token、target probability、原始熵、词表归一化熵、
`exp(H)`有效支持规模、top-1/top-2 margin、table frequency/hit、ratio gate、entropy
gate 和关闭强制接受时的`would_force_accept`。

采样请求的熵基于 temperature、top-k、top-p 后的 target distribution；greedy 请求的
熵基于 penalty/grammar 处理后、过滤前的完整 target softmax。每条事件通过
`entropy_basis`明确标注口径。

## 构建

本功能只修改 Python 层，不改变 kernel schema，因此无需重新编译`sgl-kernel`；重启
服务即可生效。
