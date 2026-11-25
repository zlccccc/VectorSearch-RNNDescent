# 参数配置

从仓库根目录运行程序。编译命令和指令集列表见 [README](../README.md)，C++ 直接调用的配置见 [接口说明](api.md)。命令行和预设定义在 [BenchmarkCLI.h](../benchmark/BenchmarkCLI.h) 与 [BenchmarkTypes.h](../benchmark/BenchmarkTypes.h)。

**输入模式由显式参数优先确定。**

~~~bash
./build/algorithm --help
./build/algorithm --dataset-dir benches/datasets/sift --repeat 5
./build/algorithm --dataset-name sift \
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

## 常用搜索参数

用数据集预设起跑。调 Recall 时先看 `--search-L`，再按原始维度粗排和 FP32 精排的开销调整后续候选数。

| CLI 参数 | C++ 字段 | 控制哪一步 |
|---|---|---|
| `--search-L` | `SearchConfig::search_L` | 查图时，只展开当前排名前 L 的候选。候选池更新后，总展开数可以超过 L |
| `--rerank-candidates` | `SearchConfig::refine_max` | 查图结束后，送入原始维度重排的候选上限 |
| `--fp32-candidates` | `SearchConfig::fp32_refine_max` | 量化粗排后，送入 FP32 精排的候选上限，默认 32 |
| `--topk` | 搜索调用的 `topk` | 最终返回的邻居数，Benchmark 默认 10 |

查图维护 `max(search_L, refine_max)` 个有序候选。搜索宽度和各阶段候选上限至少按 Top K 配置，实际候选数取决于上一步产出的数量。候选已经在 FP32 预算内时直接精排。用 `--rerank-quantization fp32` 可跳过量化粗排，对全部重排候选算原始 FP32 距离。`--fp32-candidates 0` 也可以表达全量 FP32 精排。

## 索引和查图进阶参数

| CLI 参数 | 配置字段 | 含义 |
|---|---|---|
| `--S` | `BuildConfig::S` | 每个节点的初始随机邻居数量 |
| `--R` | `BuildConfig::R` | 反向边处理时保留的候选上限 |
| `--K0` | `BuildConfig::K0` | 最终邻接表的裁剪上限，存储时会补齐到四条一组 |
| `--local-three-hop-prune` | `BuildConfig::local_three_hop_pruning` | 启用局部三跳筛边，默认关闭 |
| `--num-initialize` | `SearchConfig::num_initialize` | 查询入口数，也决定 BFS 重排从哪些点开始 |
| `--beam-size` | `SearchConfig::beam_size` | 每次更新候选池前最多展开的节点数 |
| `--graph-quantization` | `SearchConfig::graph_quantization` | 查图底库及邻居缓存的编码，支持 `8bit`（默认）和 `uint4` |
| `--rerank-quantization` | `SearchConfig::rerank_quantization` | 原始维度粗排的编码，支持 `8bit`（默认）、`uint4` 和 `fp32`（跳过量化粗排） |
| `--neighbor-pool-gb` | `BuildConfig::neighbor_pool_size_limit_bytes` | 邻接 ID 与可选邻居向量缓存的池预算，CLI 单位为整数 GiB |
| `--pca-out` | `PCAConfig::out_dim` | 建图和查图的 PCA 维度，默认 0 使用原始维度 |

入口数至少覆盖 `search_L` 和重排候选上限，并按四路算距对齐，受底库条数限制。日志和报告记录输入值，归一化规则见 [GraphConfig.h](../solution/rnndescent/build/GraphConfig.h)。`beam-size` 会改变合并频率和搜索顺序，调参时同时校验 Recall。

局部三跳筛边在最终裁剪前替换有替代路径的边，保持每个节点的出度。启用后可用 ground truth 重新选择达到目标 Recall 的 `search_L`。

`--pca-out 64` 会在输入维度大于 64 时启用 PCA。PCA 输出维度须不超过底库条数，原始维度粗排和 FP32 精排仍使用原始向量。

查图和粗排独立选编码。`8bit` 使用平台的 INT8 或 UINT8 内核。查图已在原始维度且两阶段均选 `8bit` 时，粗排可复用查图分数。`uint4` 需要 AVX-VNNI、AVX-512 VNNI 或支持点积的 NEON，查询用有符号字节参与点积。RNN-Descent 建边始终使用平台的 8 位距离计算器。

邻居池为 0 GiB 时仍分配邻接 ID。正数预算覆盖邻接 ID 和缓存向量，邻接 ID 优先分配。可选缓存还受可用内存限制，程序为系统预留 2 GiB。实际分配和覆盖率见日志，内存组成见 [算法与数据流](architecture.md)。

## 查询调度

| CLI 参数 | 默认值 | 含义 |
|---|---:|---|
| `--query-batch-size` | 0 | 每次请求最多提交多少条查询，0 提交整批 |
| `--single-query-threads` | 1 | 同一条查询的线程数，最多 8 个 |
| `--worker-candidates` | 0 | 单查询并行时每线程交给合并的候选上限，至少为 Top K，0 继承全局重排候选上限 |
| `--worker-expansions` | 0 | 单查询并行时每线程实际展开的图节点数上限，0 不限 |
| `--no-reorder-query-batch` | 关闭 | 开启该选项时，批量查询按输入顺序执行 |

多条查询按 `SearchConfig::num_threads` 在查询之间分工。至多 16 条且查询数超过线程数时，逐条动态分工。其余批次在每 4 条形成的块数不超过线程数时采用静态分工，块数更多时按每块 4 条动态分工。默认在 PCA 至少输出 4 维且本次至少传入 2 条查询时，索引在分流前按前四个主成分生成排序键。可用 `--no-reorder-query-batch` 按输入顺序执行查询，结果均写回原输入行。

单查询并行按四路算距的粒度给各线程分配互不重叠的入口区间，并分摊 `search_L`。线程共用节点访问标记，合并、去重后由同一线程组分担粗排和 FP32 算距。可用入口不足时会减少实际线程数。并行和单线程的候选池不同，比较延迟时请用 ground truth 重新校验达到目标 Recall 的 `search_L`。

例如，用 `--query-batch-size 1 --single-query-threads 8` 测单查询延迟。每线程预算只在有多个线程共同处理一条查询时生效。展开上限按实际读到的邻接表计数，访问标记拦下的节点不计入。候选上限作用于合并输入，本地候选池至少保有 `search_L` 分摊后的宽度。

## 数据集预设

Benchmark 先应用通用配置，再应用数据集预设，最后应用命令行覆盖值。报告沿用原有字段名。

| 数据集名称 | S | R | K0 | 入口数 | 查图宽度 | Beam | 重排候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random / 自定义名称 | 196 | 2048 | 48 | 64 | 96 | 8 | 64 |
| sift / siftsmall | 196 | 2048 | 32 | 192 | 128 | 8 | 64 |
| sift100k | 64 | 96 | 24 | 80 | 72 | 4 | 24 |
| gist | 320 | 256 | 64 | 3072 | 1152 | 8 | 1024 |

表中的入口数为归一化前的配置值。所有预设的 `T1=4`、`T2=15`，查图和粗排默认用 8 位，FP32 精排数量默认 32，PCA 默认关闭，邻居池预算默认 0 GiB。名称按小写匹配，例如目录名为 `gist1m` 时可加 `--dataset-name gist`。预设值用于启动测量，目标 Recall 由实际数据与 ground truth 校验。

`BuildConfig` 的 `T1`、`T2`、`random_seed`、`save_neighbor`、`random_init` 可通过 C++ 设置。旧命令中的 `--refine-max`、`--fp32-refine-max`、`--single-query-refine-max` 和 `--single-query-max-expansions` 仍可使用。`--disable-pca` 等价于 `--pca-out 0`。旧开关 `--enable-pca` 仍需有有效的输出维度才会启用 PCA。

## 输入与测量

| CLI 参数 | 默认值 | 说明 |
|---|---|---|
| `--repeat` | 10 | 同一索引上的查询批次重复次数 |
| `--warmup-topk` | 10 | 建图后用于预热的 Top K，0 跳过预热 |
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
| `OMP_WAIT_POLICY` | 控制空闲 OpenMP 线程的等待策略，`run.sh` 默认设为 `ACTIVE` |
| `GOMP_SPINCOUNT` | GCC libgomp 线程休眠前的自旋次数，`run.sh` 在 `ACTIVE` 下默认设为 `16000` |
| `DATASET` | 默认数据集目录名 |
| `BUILD_DIR` | `run.sh` 使用的 CMake 构建目录 |
| `RESULT_DIR` | 本次报告的输出目录 |

PCA 的并行查询投影使用 `SearchConfig::num_threads` 作为线程预算。OpenBLAS 线程数仍影响 PCA 训练、底库大批投影和单线程下的整批查询投影。C++ 调用时直接设置 `BuildConfig::num_threads` 与 `SearchConfig::num_threads`，Benchmark 的 16 线程限制只作用于其参数预设过程。[PCA 投影规则](architecture.md)

`run.sh` 默认使用 `ACTIVE + 16000`，启动前设置的环境变量优先。可用 `OMP_WAIT_POLICY=PASSIVE ./run.sh` 选择被动等待。直接启动二进制或在宿主程序中使用索引时，由启动进程的环境决定运行时配置。 [原默认与各自旋设置的实测](../benchmark/v2.0experiments/single-query-wait.md)
