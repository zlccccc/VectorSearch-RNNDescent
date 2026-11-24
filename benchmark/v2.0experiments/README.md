# v2.0 Experiments

比较基准为 [v1.1](../v1.1experiments/README.md)。从 `00d9cbc` 开始记录批量调度和查询并行的实验。

v2.0 的目标是让查询支持串行、单查询内多线程，以及批量查询之间的多线程分工。单条用 `SearchConfig::single_query_threads` 配置，批量用 `SearchConfig::num_threads` 配置。用同图、同查询和 Recall@10 ≥ 99% 的结果对照单条延迟、QPS 和 QPS/thread。[查询接口](../../docs/api.md)

- GIST1M 和 SIFT1M 的 PCA 批内排序，全缓存和零缓存对照。[PCA 批内排序](query-batch-reorder.md)
- GIST1M 和 SIFT1M 的 Batch 1、6、16 和整批，覆盖 1、6、16 线程的 QPS、QPS/thread、Recall@10 和搜索分支。[查询吞吐矩阵](batch-thread-dispatch.md#固定归档参数的线程对照)
- `L`、批大小和线程分工对吞吐的影响。[小批查询吞吐](small-batch-throughput.md)
- 按 v1.1 归档参数对跑四个数据集，并测搜索预算变化对整批吞吐的影响。[v1.1 与当前版同图复测](fixed-graph-comparison.md)
