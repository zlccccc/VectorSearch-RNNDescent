# Benchmark 性能基准

目标 Recall@10 ≥ 99%。`QPS/thread` 为整批查询 QPS 除以实际查询线程数。

| 版本 | 测试结果 |
|---|---|
| V1 | [跨设备与指令集](v1experiments/cross-platform.md) · [邻居缓存、去重和绑核实验](v1experiments/README.md) |
| V2 | [改动与收益来源](v2experiments/README.md) · [V1 与 V2 同图对照](v2experiments/fixed-graph-comparison.md) |

数据集准备见 [数据集说明](../benches/README.md)，构建和测试见 [开发与验证](../docs/development.md)。
