# Benchmark 性能基准

目标 Recall@10 ≥ 99%。`QPS/thread` 为查询 QPS 除以各实验注明的查询线程数。

| 版本 | 测试结果 |
|---|---|
| v1.0 | [跨设备与指令集](v1.0experiments/cross-platform.md) · [邻居缓存、去重和绑核实验](v1.0experiments/README.md) |
| v1.1 | [基线与改动](v1.1experiments/README.md) · [v1.0 与 v1.1 同图对照](v1.1experiments/README.md#同图性能对照) |
| v2.0 | [与 v1.1 的封板对照](v2.0experiments/README.md#封板性能对照) · [查询吞吐矩阵](v2.0experiments/batch-thread-dispatch.md#当前工作区复测) · [PCA 批内排序](v2.0experiments/query-batch-reorder.md) |
| v2.1 | [同图查询性能与小批准备门槛](v2.1experiments/README.md) |

数据集准备见 [数据集说明](../benches/README.md)，构建和测试见 [开发与验证](../docs/development.md)。
