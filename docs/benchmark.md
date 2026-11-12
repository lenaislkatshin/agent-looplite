# 流式增量的 CPU 开销

用同一份模拟的流式响应，比较本库和 pi-python 处理增量的 CPU 开销。脚本是 `scripts/bench.py`。

## 方法

- 响应由 httpx 的 `MockTransport` 返回，不走网络。每个增量是 5 个汉字，最后一片带 `finish_reason`。
- 两边都走 Chat Completions 连接器加上层 `Agent`，事件交给一个只计数的监听函数。
- 每组先跑一次预热，再并发跑多路，用 `time.process_time()` 记 CPU 时间。
- pi-python 是交通大脑用的 fork，提交 `6d252c833a0e` 构建的 `pi_python_core-0.12.0`。

```bash
uv run python scripts/bench.py --deltas 4000 --runs 50
uv run --with <pi_python_core 的 wheel> python scripts/bench.py --impl pi-python --deltas 4000 --runs 50
```

## 结果

2026-11-12，Apple M4，Python 3.11.14。两边每路的事件数相同。

| 并发 × 每路增量 | 本库 CPU | 本库每个增量 | pi-python CPU | pi-python 每个增量 |
|---|---:|---:|---:|---:|
| 10 × 1000 | 0.02 秒 | 2.5 微秒 | 1.53 秒 | 153 微秒 |
| 10 × 4000 | 0.09 秒 | 2.4 微秒 | 11.55 秒 | 289 微秒 |
| 10 × 8000 | 0.19 秒 | 2.4 微秒 | 36.86 秒 | 461 微秒 |
| 50 × 4000 | 0.48 秒 | 2.4 微秒 | 59.65 秒 | 298 微秒 |

本库每个增量的开销不随回复长度变化。pi-python 每个增量都生成一份完整的部分消息，回复越长，每个增量越贵，总开销随长度平方增长。

这组数据只测流式增量这一条路径，不含真实网络、工具执行和模型本身的耗时。对照对象是 pi-python，不能据此推断和 Pi TS 原版谁快；也不能代替行为对照。
