# v1.1 Experiments

`v1.1` 对应 2026-11-24 11:05（北京时间）的 `ba23258`，位于 `00d9cbc` 引入小批分流和批内排序之前。当时多条查询可通过 OpenMP 并行，每条查询由一个线程完成。用这版作为后续批量调度和单查询内并行的吞吐基线。

同图对照覆盖 SIFT、GIST 的 100K 和 1M 数据，以及全缓存和零缓存。测量使用 `e78da58`，其求解器源码与 `v1.1` 一致。[v1.0 与 v1.1 同图对照](fixed-graph-comparison.md)

## 改动与收益来源

- **未缓存邻居按整张邻接表算距。** v1.0 每四个邻居调用一次距离计算器。v1.1 一次传入整张表的 ID，后端复用查询码地址和范数，循环仍以四路 SIMD 处理。64 个邻居的虚调用从 16 次减到 1 次。单项同图对照中，SIFT1M 和 GIST1M 的零缓存 QPS 分别提升 18.3% 和 9.8%。[批量算距实现](../../solution/rnndescent/discomputer/quantized/QuantizedDistanceComputer.h)
- **压缩邻接记录。** v1.0 每点的描述保存多个指针、码长和邻居数。v1.1 每点只存一个指针，指向池内的连续记录。缓存记录依次存度数与标志、范数、量化码和 ID，零缓存记录只存头和 ID。百万个节点的描述从每点 40 字节降到 8 字节，合计少 30.5 MiB。[邻接记录布局](../../solution/rnndescent/discomputer/quantized/NeighborsStorageBase.h)
- **零缓存邻接 ID 池使用大页。** 只存邻接 ID 时也对池内完整的 2 MiB 区间申请大页，未能提升的区间沿用普通页。SIFT1M 和 GIST1M 的零缓存吞吐分别提升 2.4% 和 1.7%，峰值 RSS 基本持平。[邻接池实现](../../solution/rnndescent/discomputer/quantized/NeighborsStorageBase.h#L94)
- **整轮筛选后读 ID。** v1.0 算完整轮距离后，把所有邻居 ID 复制到临时数组再筛选。v1.1 记录邻接段，先用 SIMD 生成整轮的 cutoff 掩码，再从原邻接表收集过线项的 ID，随后用 bitmap 去重、排序和归并。SIFT1M 固定紧凑布局的阶段对照中，相对逐邻接表筛选，整轮方案在全缓存和零缓存下分别提升 3.0% 和 6.8%。[查图筛选与合并](../../solution/rnndescent/search/GraphSearcher.cpp)
- **批量处理入口和 FP32 精排。** 入口和精排都把整批候选交给后端，复用查询数据和范数。FP32 Top40 测距批次在 128 维和 960 维分别省时 10.5% 和 2.5%，FP32 精排的单项端到端对照变化约在 1% 以内。[FP32 批量实现](../../solution/rnndescent/discomputer/flat/FlatDistanceComputer.h)
- **整理建图和测距接口。** `GraphBuilder` 自己管理节点锁，每个建图线程复用采样缓冲。精排的 ID 映射集中到 `rerank`，测距接口移除了旧的四路虚调用和重复的查询码访问方法。固定图对照衡量的是查询吞吐，建图整理发生在生成邻接表时。[建图实现](../../solution/rnndescent/build/GraphBuilder.cpp) · [精排实现](../../solution/rnndescent/search/CandidateReranker.cpp)

各项百分比分别以该阶段的参照版本为基准。
