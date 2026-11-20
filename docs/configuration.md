# 参数配置

从仓库根目录运行程序。编译命令和指令集列表见 [README](../README.md)，C++ 直接调用的配置见 [接口说明](api.md)。命令行和预设定义在 [BenchmarkCLI.h](../benchmark/BenchmarkCLI.h) 与 [BenchmarkTypes.h](../benchmark/BenchmarkTypes.h)。

**输入模式由显式参数优先确定。**

~~~bash
./build/algorithm --help
./build/algorithm --mode dataset --dataset-dir benches/datasets/sift --topk 10 --repeat 5
./build/algorithm --mode dataset --dataset-name sift \
  --base benches/datasets/sift/base.fvecs \
  --query benches/datasets/sift/query.fvecs \
  --gt benches/datasets/sift/groundtruth.ivecs --topk 10 --repeat 5
~~~

传入 `--dataset-dir`、`--base`、`--query` 或 `--gt` 会选择 dataset 模式。使用独立文件路径时，用 `--dataset-name` 选择对应预设。目录输入省略该名称时，使用目录名。

直接启动可执行文件时，会尝试使用 `benches/datasets/` 下由 `DATASET` 指定的目录，默认名称为 `sift`。该目录存在时采用 dataset 模式，否则采用 random 模式。显式 `--mode random` 会直接选择随机数据。随机模式省略 `--dim` 时依次运行 256 和 512 维。

`run.sh` 会从仓库根目录配置和增量构建程序。无参数时，它显式选择 `DATASET` 对应的目录，默认 `sift`。可以用完整参数指定输入。

~~~bash
DATASET=sift100k OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 ./run.sh
BUILD_DIR=build/avx2 ./run.sh --mode random --data-size 10000 --dim 128 \
  --test-iter 100 --repeat 3 --exact-check-queries 20 --warmup-topk 0
~~~

**建图和搜索参数使用不同的配置。**

| CLI 参数 | 配置字段 | 含义 |
|---|---|---|
| `--S` | `BuildConfig::S` | 每个节点的初始随机邻居数量 |
| `--R` | `BuildConfig::R` | 反向边处理时保留的候选上限 |
| `--K0` | `BuildConfig::K0` | 最终邻接表的裁剪上限，存储时会补齐到四条一组 |
| `--local-three-hop-prune` | `BuildConfig::local_three_hop_pruning` | 启用局部三跳筛边，默认关闭 |
| `--num-initialize` | `SearchConfig::num_initialize` | 建图后选出的查询入口数，也影响 BFS 重排的起点 |
| `--search-L` | `SearchConfig::search_L` | 搜索中保持有序的候选池宽度 |
| `--beam-size` | `SearchConfig::beam_size` | 收集一批新候选前最多展开的节点数 |
| `--refine-max` | `SearchConfig::refine_max` | 查询结束时从候选池中取出、参与原始维度排序的数量 |
| `--fp32-refine-max` | `SearchConfig::fp32_refine_max` | 按原始维度 8 位距离选择后，做 FP32 精排的数量；默认 32，0 表示全部候选都做 FP32 精排 |
| `--neighbor-pool-gb` | `BuildConfig::neighbor_pool_size_limit_bytes` | 邻接 ID 与可选邻居向量缓存的池预算，CLI 单位为整数 GiB |
| `--pca-out` | `PCAConfig::out_dim` | 开启 PCA 并指定查图维度 |
| `--disable-pca` | `PCAConfig::enabled` | 使用原始维度建图和查图 |
| `--warmup-topk` | `WarmupConfig::topk` | 建图结束后的预热 Top K，0 跳过预热 |

搜索会将 `search_L` 和 `refine_max` 至少提高到 Top K，将入口数至少提高到 `search_L`，再根据底库条数和四路算距要求调整入口数。预设日志和报告记录输入配置，具体归一化规则见 [RNNDescent.h](../solution/rnndescent/RNNDescent.h) 中的 `sanitize_search_config`。实际使用时可以显式设置四的倍数入口，并取 `topk <= refine_max <= search_L <= num_initialize`，让预算含义直接对应候选池。

搜索会持续处理当前候选池中的未展开节点。扩大 `search_L` 会同时影响候选保留和搜索工作量。`beam_size` 会影响候选合并频率与搜索顺序，应结合 Recall@10 比较。`refine_max` 控制原始维度排序的候选量，`fp32_refine_max` 控制最后的 FP32 精排量。实际 FP32 数量至少为 Top K，并且不超过实际候选数；候选数不超过 FP32 数量时，直接做 FP32 精排。

局部三跳筛边在最终裁剪前替换有替代路径的边，保持每个节点的出度。启用后可用 ground truth 重新选择达到目标 Recall 的 `search_L`。

启用 PCA 时，先对候选计算原始维度的 INT8/UINT8 距离，再保留 FP32 精排候选。不开 PCA 时，查图已经计算相同原始维度的量化距离，直接复用这些分数。`--fp32-refine-max 0` 跳过量化筛选，恢复完整 FP32 精排。调整这个数量会影响 Recall，比较性能时需要分别调参到相同目标精度。

PCA 在 `enabled=true` 且 `0 < out_dim < 输入维度` 时启用，训练要求 `out_dim <= 底库条数`。`--pca-out 64` 同时设置开关与维度。`--enable-pca` 只设置开关，需要配合有效的输出维度。精排使用原始向量与原始维度。

邻居池为 0 GiB 时仍分配邻接 ID。正数预算用于邻接 ID 和缓存数据，邻接 ID 的必要空间优先保证，因此实际池大小可超过很小的预算。程序还会读取系统可用内存，并为系统预留 2 GiB 后限制可选缓存。日志里的缓存覆盖率和池大小反映实际分配情况。完整内存组成见 [算法与数据流](architecture.md)。

**Benchmark 先应用通用配置，再应用数据集预设，最后应用 CLI 覆盖值。**

| 数据集名称 | S | R | K0 | 入口数 | search_L | beam_size | refine_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| random / 自定义名称 | 196 | 2048 | 48 | 64 | 96 | 8 | 64 |
| sift / siftsmall | 196 | 2048 | 32 | 192 | 128 | 8 | 64 |
| sift100k | 64 | 96 | 24 | 80 | 72 | 4 | 24 |
| gist | 320 | 256 | 64 | 3072 | 1152 | 8 | 1024 |

表中的入口数为归一化前的配置值。所有预设的 `T1=4`、`T2=15`，FP32 精排数量默认 32，PCA 默认关闭，邻居池预算默认 0 GiB。名称按小写匹配，例如目录名为 `gist1m` 时可加 `--dataset-name gist`。预设值用于启动测量，目标 Recall 由实际数据与 ground truth 校验。

`BuildConfig` 的 `T1`、`T2`、`random_seed`、`save_neighbor`、`random_init` 可通过 C++ 设置。CLI 提供上表列出的建图和搜索覆盖项。

**输入与测量参数决定读取范围和报告内容。**

| CLI 参数 | 默认值 | 说明 |
|---|---|---|
| `--topk` | 10 | 每条查询返回的邻居数 |
| `--repeat` | 10 | 同一索引上的查询批次重复次数 |
| `--output-iter` | 5 | 打印结果详情的查询数，0 只保留汇总 |
| `--exact-check-queries` | 20 | 随机模式中做穷举校验的查询数 |
| `--data-size` | 100000 | 随机底库条数 |
| `--test-iter` | 10000 | 随机查询条数 |
| `--dim` | 分别跑 256、512 | 显式设置时只运行指定的随机维度 |
| `--dataset-dir` | 自动探测 | 包含底库、查询和可选 ground truth 的目录 |
| `--base`、`--query`、`--gt` | 由目录解析 | 指定独立的数据文件 |
| `--dataset-name` | 数据集目录名 | 选择预设与报告目录名称 |

dataset 模式读取文件中的全部底库和查询行，用对应 ground truth 评估。文件格式与命名规则见 [Benchmark 使用](../benches/README.md)。

**线程数由 Benchmark 和底层库分别控制。**

| 环境变量 | 作用 |
|---|---|
| `OMP_NUM_THREADS` | Benchmark 取 `min(16, omp_get_max_threads())` 设置建图和搜索线程数，同时影响未显式指定线程数的 OpenMP 循环 |
| `OPENBLAS_NUM_THREADS` | 控制 OpenBLAS 自身的线程数 |
| `OMP_DYNAMIC=FALSE` | 固定 OpenMP 请求的线程配置 |
| `OMP_PROC_BIND`、`OMP_PLACES` | 由 OpenMP 运行时管理线程亲和性 |
| `DATASET` | 默认数据集目录名 |
| `BUILD_DIR` | `run.sh` 使用的 CMake 构建目录 |
| `RESULT_DIR` | 本次报告的输出目录 |

一般查询测量可以使用 `OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1`。PCA 训练和矩阵投影使用 OpenBLAS，可以另行比较其线程数，并记录配置。C++ 调用时直接设置 `BuildConfig::num_threads` 与 `SearchConfig::num_threads`，Benchmark 的 16 线程限制只作用于其参数预设过程。
