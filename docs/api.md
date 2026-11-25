# C++ 接口

[Solution](../solution/solution.h) 提供批量建索引和查询。输入矩阵使用行优先的 `std::vector<float>`，向量连续拼接，结果 ID 对应底库原始行号，距离为原始 FP32 空间的平方 L2。

| 方法 | 输入和结果 |
|---|---|
| `build(d, base, warmup, build_config, search_config, pca_config)` | 根据底库创建索引，最后一项默认为关闭 PCA |
| `search(queries, ids, topk)` | 按查询条数选择执行方式，自动将 `ids` 调整为查询条数乘 Top K |
| `search(query_view, result_view)` | 直接写入调用方提供的 ID 和距离缓冲区，每个缓冲区须能容纳查询条数乘 Top K 项 |
| `search_batch(queries, ids, topk)` | 批量调用入口，使用与 `search` 相同的查询准备与调度 |
| `distance_buffer()` | 返回最近一次向量接口写入的距离，与 `ids` 按相同顺序排列 |
| `reset()` | 释放当前图、量化存储、邻居池和 PCA，后续可再次 `build` |

底库行数为 `base.size() / d`，查询条数为 `queries.size() / d`。第 `q` 条查询的第 `k` 个结果位于 `ids[q * topk + k]`，对应距离位于 `distance_buffer()[q * topk + k]`。输入要求维度为正、底库非空、元素数能整除维度，并提供有限数值。初始邻居数 `S` 应在底库条数范围内。Top K 需为正并受底库条数约束，入口需要满足四路批量算距的要求，示例使用 64 个入口。

**底库的生命周期覆盖索引的使用时间。**

默认 FP32 精排引用调用方的底库内存。保留 `base` 的内容与地址，直到索引重置或销毁，随后可以替换底库并重建。量化数据、图结构、PCA 和每线程搜索缓冲由索引管理。查询调用同步完成，返回后可以复用查询缓冲。距离结果在下一次查询时更新，对象销毁时失效。

`SearchConfig::search_L` 控制参与展开的候选排名，`SearchConfig::refine_max` 控制送入原始维度重排的候选数，两者可独立设置。`SearchConfig::graph_quantization` 选择图搜索编码，`SearchConfig::rerank_quantization` 选择原始维度粗排编码。两者均默认 `QuantizedFormat::Byte`，有对应 SIMD 内核时可设为 `QuantizedFormat::UInt4`。

量化粗排按分数保留 `fp32_refine_max` 项，默认 32，至少保留 Top K。设 `rerank_quantization = std::nullopt` 或 `fp32_refine_max = 0` 时，送入重排的候选直接做 FP32 精排。返回距离为原始 FP32 空间的平方 L2。

只传入一条时，可用 `SearchConfig::single_query_threads` 指定查图和重排算距共用的线程数，最多 8 个。传入多条时，每条查询由一个线程完成，OpenMP 线程组大小由 `SearchConfig::num_threads` 指定。至多 16 条且查询数超过线程数时，线程逐条动态领取查询。其余批次在每 4 条形成的块数不超过线程数时用静态分工，块数更多时按每块 4 条动态分工。符合 PCA 批内排序条件的查询默认在分流前排序，可将 `SearchConfig::reorder_query_batch` 设为 `false` 以按输入顺序执行。结果始终按输入行写回。

`SearchConfig::single_query_limits` 集中配置单查询并行的两个预算。`candidates_per_worker` 限制每线程进入 FP32 精排的候选数。默认 0 时，量化粗排后每个线程固定保留至多 16 项，Top K 大于 16 时上限用 Top K。正数覆盖默认上限，仍至少支持 Top K。直接 FP32 精排时，0 保留全部本地重排候选，正数按查图分数截断。`expansions_per_worker` 限制每线程实际展开的图节点数，默认 0 不限。索引建好后，可在两次搜索之间调用 `set_single_query_limits(candidates_per_worker, expansions_per_worker)`。

两条路径共用量化查询、PCA 查询缓冲和搜索工作区。同一 `Solution` 实例按顺序调用查询。不同实例分别管理自己的索引和邻居池。外层需要并发访问同一实例时，由调用方串行化这些入口。

`reset()` 之后，查询会报告索引尚未构建。它会保留部分查询工作缓冲的容量以便复用，`distance_buffer()` 可暂时保留上一批结果。对象销毁时释放这些缓冲。

**下面的完整示例生成 256 条 32 维向量并查询其中 8 条。**

~~~cpp
#include "solution/solution.h"

#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr int dimension = 32;
    constexpr int rows = 256;
    constexpr int query_count = 8;
    constexpr int topk = 5;

    std::mt19937 random(19);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> base(rows * dimension);
    for (float &coordinate : base)
        coordinate = value(random);
    std::vector<float> queries(base.begin(), base.begin() + query_count * dimension);

    Solution::WarmupConfig warmup;
    warmup.topk = 0;
    rnndescent::RNNDescent::BuildConfig build;
    build.S = 16;
    build.R = 32;
    build.K0 = 16;
    build.num_threads = 16;
    rnndescent::RNNDescent::SearchConfig search;
    search.num_initialize = 64;
    search.search_L = 48;
    search.refine_max = 48;
    search.fp32_refine_max = 32;
    search.beam_size = 4;
    search.num_threads = 16;
    rnndescent::IndexRNNDescent::PCAConfig pca{true, 16};

    Solution index;
    std::vector<int> ids;
    index.build(dimension, base, warmup, build, search, pca);
    index.search_batch(queries, ids, topk);
    const auto &distances = index.distance_buffer();
    for (int q = 0; q < query_count; ++q)
        std::cout << q << " " << ids[q * topk] << " " << distances[q * topk] << '\n';
    index.reset();
}
~~~

将示例保存为 `/tmp/rnndescent-example.cpp`，可从仓库根目录在支持 AVX2 和 FMA 的 x86-64 机器上编译运行。

~~~bash
c++ -O3 -std=c++17 -fopenmp -mavx2 -mfma -DRNNDESCENT_FORCE_AVX2 -I . \
  /tmp/rnndescent-example.cpp solution/solution.cpp solution/rnndescent/PCAProjection.cpp \
  solution/rnndescent/build/GraphBuilder.cpp solution/rnndescent/build/GraphReorder.cpp \
  solution/rnndescent/search/GraphSearcher.cpp solution/rnndescent/search/CandidateReranker.cpp \
  -lopenblas -o /tmp/rnndescent-example
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 /tmp/rnndescent-example
~~~

也可以在调用方 CMake 工程中通过 `add_subdirectory` 引入本项目，再让调用目标链接 `rnndescent_solution`。该 target 会传递 C++17、OpenMP、OpenBLAS 和所选 SIMD 编译选项。构建目标与选项见 [开发与验证](development.md)。

C++ 直接调用使用配置结构的初值，CLI 还会应用数据集预设。示例显式设置了入口、候选池和精排数量，字段含义见 [参数配置](configuration.md)。
