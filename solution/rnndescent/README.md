# RNNDescent 索引模块

索引实现按职责拆成多个模块。`IndexRNNDescent` 管理 PCA、底库生命周期和查询批次，`RNNDescent` 保留兼容外观并组装图构建、图重排和搜索模块。

| 模块 | 文件 | 职责 |
|---|---|---|
| 公共基础设施 | `common/` | 断言、日志、矩阵视图、输入校验和量化格式 |
| 图类型与配置 | `build/GraphTypes.h`、`build/GraphConfig.h` | 邻居表示、图别名、建图和查询参数归一化 |
| 图构建 | `build/GraphBuilder.h/.cpp` | 初始随机图、邻居更新、反向边、裁剪和补齐前的邻接表 |
| 图重排 | `build/GraphReorder.h/.cpp` | BFS 编号、原始 ID 映射、量化底库和邻居缓存物化 |
| 图搜索初排 | `search/GraphSearcher.h/.cpp` | 遍历图、合并并去重候选，返回按查图分数排列的 BFS ID |
| 候选重排 | `search/CandidateReranker.h/.cpp` | 可选的原始维度量化粗排、原始 ID 映射与最终 FP32 精排 |

## 数据流

建图阶段由 `GraphBuilder` 生成排序、去重和裁剪后的邻接表。`GraphReorder` 把邻接 ID 展平、补齐到四的倍数，再按 BFS 顺序建立图 ID 到原始行号的映射。量化底库和可选邻居缓存直接按这个顺序写入索引。

查询阶段由 `GraphSearcher` 为每个 OpenMP 线程复用候选池、分数缓冲、ID 缓冲和 `CandidateBitmap`。邻居算距结果先写入连续分数缓冲，再按队尾 cutoff 筛选、去重、排序和合并。图候选缓冲按引用传给 `CandidateReranker`，它用独立配置的距离计算器做原始维度粗排。关闭粗排时，直接映射候选 ID 并计算 FP32 距离。编码选项见 [参数配置](../../docs/configuration.md)。

## 兼容边界

`RNNDescent::BuildConfig`、`SearchConfig`、`KNNGraph`、`searchSingle`、`final_graph_neighbors`、`graph_distance_computer` 和 `search_from_ids` 保持现有访问方式。模块之间通过引用传递图存储和距离计算器，搜索热循环不引入虚函数或额外拷贝。

量化和平台 SIMD 实现在 [`discomputer/README.md`](discomputer/README.md)。整体数据流和内存组成见 [`docs/architecture.md`](../../docs/architecture.md)，构建与测试命令见 [`docs/development.md`](../../docs/development.md)。
