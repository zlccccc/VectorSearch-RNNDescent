# RNNDescent 向量检索算法

本项目实现了基于 RNNDescent 的近似最近邻检索，使用平方 L2 距离。构建阶段迭代生成近邻图并按 BFS 重排节点。查询依次经过图搜索初排、可选的原始维度量化粗排和原始 FP32 精排，返回 Top K。查图和粗排可分别选择 8 位或 4 位编码，默认均为 8 位。启用量化粗排时默认保留 32 个候选做 FP32 精排。

后续优化优先减少数据读取和搬运，向量及邻居向量缓存尽量使用 INT8/UINT8 存储。计算按速度选择精度，在目标 Recall 下以端到端性能决定取舍。具体约定见 [开发与验证](docs/development.md)。

| 内容 | 文档 |
|---|---|
| 索引模块、建图、重排和搜索职责 | [RNNDescent 模块](solution/rnndescent/README.md) |
| 运行模式、参数默认值、数据集预设和线程配置 | [参数配置](docs/configuration.md) |
| C++ 调用示例、输入输出、内存所有权和重建 | [C++ 接口](docs/api.md) |
| 建图、PCA、搜索、候选合并和内存组成 | [算法与数据流](docs/architecture.md) |
| 距离公式、SIMD 内核和邻居缓存扩展 | [测距实现](solution/rnndescent/discomputer/README.md) |
| 本地测试、GitHub Actions、构建问题定位 | [开发与验证](docs/development.md) |
| 当前 QPS、Recall 与复测方法 | [Benchmark 性能基准](benchmark/README.md) |
| 数据格式、下载、SIFT 与 GIST 测量方法 | [Benchmark 使用](benches/README.md) |

## 许可声明

本项目（外部调试版本）的知识产权归华为公司所有，未经授权不得用于商业目的。请参阅文末的[商用许可](#商用许可)。

## 构建

在 Linux 或 WSL 2 中使用支持 C++17 和 OpenMP 的编译器、CMake 3.23 或更新版本，以及包含 LAPACK 的 OpenBLAS。以下是 Ubuntu 24.04 的构建示例。程序位于 `build/algorithm`。

```bash
sudo apt install build-essential cmake libopenblas-dev libgtest-dev libgmock-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure --no-tests=error
```

下面的随机数据示例可直接验证构建和查询入口。运行命令均从仓库根目录执行。

~~~bash
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  ./build/algorithm --mode random --data-size 10000 --dim 128 \
  --test-iter 100 --topk 10 --repeat 3 --exact-check-queries 20 \
  --S 32 --R 64 --K0 16 --num-initialize 64 --search-L 48 --refine-max 48 \
  --warmup-topk 0 --output-iter 0
~~~

用 `-DRNNDESCENT_SIMD=<值>` 选择编译目标。切换指令集时建议用不同的构建目录，以便分别保留可执行文件和测试结果。

| 值 | 平台与指令集 |
|---|---|
| `native`（默认） | x86-64 按构建机选择 AVX2 或 AVX-512 VNNI，AArch64 使用 NEON Dot Product |
| `avx2` | x86-64，要求 AVX2 和 FMA |
| `avx2-uint8` | x86-64，要求 AVX2 和 FMA，使用 0～126 的量化码和字节乘加 |
| `avx-vnni` | x86-64，要求 AVX2、FMA 和 AVX-VNNI，使用相同量化码和 256 位 VNNI 点积 |
| `avx512` | x86-64，要求 AVX-512F、BW、VL 和 VNNI |
| `neon` | AArch64，要求 NEON Dot Product（Linux 的 `asimddp`） |

指令集在编译时选定。运行测试的机器也需要支持所选指令集。

AVX-VNNI、AVX-512 VNNI 和支持点积的 NEON 构建可用 `--graph-quantization uint4` 选择 4 位查图，用 `--rerank-quantization uint4` 选择 4 位粗排。用 `--rerank-quantization fp32` 可以直接对图搜索送出的候选做 FP32 精排。详见 [参数配置](docs/configuration.md)。

### 量化算距

[QuantizedDistanceComputer.h](solution/rnndescent/discomputer/quantized/QuantizedDistanceComputer.h) 中的 `QuantizedDistanceComputer<Kernel, Encoding>` 为 8 位和 4 位量化共用。编码策略负责参数训练、码布局和分数换算，SIMD 内核负责点积和邻居缓存块。DistanceComputer 按 BFS 顺序编码底库，管理查询缓冲并提供统一算距入口。未缓存的邻居通过四路算距接口读取底库码。

| 构建目标 | 量化码 | 每块读取字节数 | 点积内核 |
|---|---|---:|---|
| `avx2` | INT8 `[-127, 127]` | 16 | 符号扩位后 `VPMADDWD` |
| `avx2-uint8` | UINT8 `[0, 126]` | 32 | `VPMADDUBSW` 和 `VPMADDWD` |
| `avx-vnni` | 同一套 UINT8 量化 | 32 | 256 位 `VPDPBUSD` |
| `avx512` | 同一套 UINT8 量化 | 64 | 512 位 `VPDPBUSD` |
| `neon` | INT8 `[-127, 127]` | 16 | `SDOT` |

`Kernel` 提供 `Quantization`、`block_bytes`、`dot`、`dot4` 和 `cached_distances`。建图、查图和邻居缓存用 32 位整数计算量化码的平方 L2，再转成 `float` 用于排序。底库和查询在编码时各计算一次平方范数。测试可将量化分数乘 `scale²`，换回量化输入的距离单位并校验公式。启用 PCA 时，这是投影空间的距离。最终返回原始向量的 FP32 精排距离。

量化索引支持最多 33,286 维，由 INT8 每维最大平方差 `254²` 和有符号 32 位整数上限算出。超过上限会在初始化时直接报错。

`dimension()` 返回逻辑维度，`code_stride()` 返回补零后的行宽。邻居缓存把四条向量的第一个 SIMD 块放在一起，再放四条向量的下一块。新增内核时，让连续读和缓存读都使用 `block_bytes`，然后在平台选择中给算距和邻居缓存绑定同一个 `Kernel`。模板在编译期绑定内核，查询入口仍使用 `MyDistanceComputer` 的接口。

AVX2、AVX512、NEON、CBLAS 的 FP32 算距，以及 NEON 的 FP16 和 INT16，都由 [FlatDistanceComputer.h](solution/rnndescent/discomputer/flat/FlatDistanceComputer.h) 管理查询、平方范数和平方 L2 计算。FP32 直接读取原始行，作为默认精排。FP16 存半精度并用 FP32 累加。INT16 用 64 位整数累加，再乘量化步长的平方换回输入单位。三种 NEON 实现共用向量点积循环和逐元素尾部处理。邻居缓存由前面的 8 位量化实现管理。

## 测试

测试使用 [GoogleTest](https://google.github.io/googletest/) 编写，CTest 自动发现每个测试用例。默认构建测试，不需要下载数据集。CMake 优先复用本机安装的 GoogleTest 和 GoogleMock（1.14 或更新版本），例如 Ubuntu 的 `libgtest-dev` 和 `libgmock-dev`。找不到 GoogleTest 时，CMake 会下载并校验固定版本 1.17.0。若本机只装了 GoogleTest，可以补装 GoogleMock，或用 `-DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON` 使用同一份下载。缓存用例要求系统可用内存超过程序预留的 2 GiB。

~~~bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -R RecallTest --output-on-failure
~~~

| 测试套件 | 覆盖内容 |
|---|---|
| `SimdDistanceTest`、`SimdQuantizationTest` | 对照标量结果校验 FP32、INT8/UINT8 距离，覆盖单条、四条一组、建图距离、非对齐维度和常量向量 |
| `NeighborCacheLayoutTest`、`NeighborCacheTest` | 校验四邻居 SIMD 数据布局、ID 重映射、预算用尽时回退以及缓存开关下的距离一致性 |
| `PCAProjectionTest` | 校验协方差和 Gram 矩阵两条训练路径的主方向、中心化及采样跨批次行为 |
| `BenchmarkCliTest`、`BenchmarkDataTest` | 校验 CLI、显式随机模式、邻居池预算，以及 fvecs/bvecs/ivecs 的读取和坏文件处理 |
| `SolutionTest` | 小数据集建图、查询、重建和重置，用穷举校验 Top K ID 和精排距离，覆盖 PCA、缓存开关和两个索引交替查询 |
| `RecallTest` | 固定随机种子，限制搜索预算，校验 Recall@10、缓存开关下的结果一致性和精排距离 |

只构建 benchmark 程序时，可在配置阶段传入 `-DBUILD_TESTING=OFF` 跳过测试依赖。

[GitHub Actions](.github/workflows/unit-tests.yml) 在推送到 `main`、面向 `main` 的 PR 和手动触发时，在 Ubuntu 24.04 上构建 AVX2 版本并运行 `unit` 标签的测试。本地完整测试还包含 `integration` 标签下的索引生命周期和召回率验证。

## Benchmark

SIFT100K 和 GIST100K 用固定图对照 AVX-VNNI、AVX-512 VNNI 和 NEON Dot Product 的 8 位与 UINT4 查图性能。各平台在 Recall@10 ≥ 99% 时的最佳 QPS、本机百万级结果及复测命令见 [Benchmark 性能基准](benchmark/README.md)。

### 随机向量

下面在 10 万条归一化随机向量上，分别对 256、512、1024 维跑 1 万条查询。固定建图和搜索参数，方便对比维度或 SIMD 路径。将 `OMP_NUM_THREADS` 设为测量使用的线程数，benchmark 的建图和查图线程数最多为 16。

```bash
for dim in 256 512 1024; do
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
    ./build/algorithm --mode random --data-size 100000 --dim "$dim" \
    --test-iter 10000 --topk 10 --repeat 10 --exact-check-queries 100 --output-iter 0 \
    --K0 32 --num-initialize 160 --search-L 160 --refine-max 128 --beam-size 8 \
    --disable-pca --neighbor-pool-gb 4
done
```

用 `--neighbor-pool-gb 0` 测只存邻接 ID 的路径。正整数给出池的 GiB 上限，池包含邻接 ID、缓存向量和对齐开销。实际缓存容量还受系统可用内存限制，可从日志查看分配大小和覆盖比例。对比吞吐量时同时记录 Recall@K 和缓存覆盖，并交替测试各版本，避免把单轮 CPU 争用当成性能差异。

### 真实数据集

```bash
sh benches/download_dataset.sh sift
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 ./build/algorithm \
  --mode dataset --dataset-dir benches/datasets/sift --topk 10 --repeat 5
```

下载脚本还支持 `siftsmall` 和 `gist`。数据目录可使用 [benches/README.md](benches/README.md) 中的文件别名，也可用 `--base`、`--query` 和可选的 `--gt` 指定文件。底库和查询支持 `fvecs`、`bvecs`，预计算近邻使用 `ivecs`。传入 `--pca-out <维度>` 可开启建图与查询的 PCA 投影，精排仍使用原始向量。

自定义数据集和随机向量采用通用参数。搜索预算可通过 `--search-L`、`--num-initialize` 和 `--refine-max` 等参数显式指定。SIFT、SIFT100K、SIFT Small 和 GIST 在 benchmark 中有数据集预设，命令行参数优先。可用 `./run.sh` 增量构建并运行 benchmark，默认使用 `benches/datasets/sift`。也可以在脚本后传入完整的 benchmark 参数，并通过 `BUILD_DIR` 使用已有的 CMake 构建目录。

将 SIFT100K 的底库、查询及对应的 ground truth 放入 `benches/datasets/sift100k`，用 `DATASET=sift100k ./run.sh` 运行以 99% Recall@10 为目标的预设。

用 `--S`、`--R` 和 `--K0` 调整建图。它们分别控制每个节点的初始随机邻居数、反向边处理时的候选上限和最终邻接表的边数上限。

要固定近邻结构测试向量变宽的影响，或把 Recall@10 拉离饱和区再比较 QPS，可运行 [SIFT benchmark 脚本](benches/README.md#可复现的-sift-测试)。脚本支持 128、256、512、1024 维对比和低搜索预算扫描，每档汇总多次独立建图的结果。

### 读取结果

程序默认写到 `benches/results/<数据集名>/`，随机测试写到 `benches/results/random-<维度>d/`。每次生成带时间戳的报告，同时更新 `latest.txt`。可用 `RESULT_DIR` 为每个版本或每轮测试指定单独的目录。

`latency_ms[i]` 是第 `i` 轮每条查询的平均毫秒数，对应 `QPS = 1000 / latency_ms[i]`。建图时间包含配置的预热。随机测试用 `--exact-check-queries` 指定穷举校验的查询条数，数据集模式提供 ground truth 后可用其计算 Recall@K。

图搜索的细粒度计时和计数默认关闭。定位热点时，可在 CMake 配置阶段传入 `-DRNNDESCENT_PROFILE_SEARCH=ON`。benchmark 始终记录整批查询耗时。

## 代码入口

- `benchmark/` 提供命令行、数据读取、评估和报告。
- `solution/solution.h` 提供 `Solution::build`、`search` 和 `reset`。底库向量供 FP32 精排引用，需在索引使用期间保持有效。结果按查询行连续存储，距离可通过 `distance_buffer()` 读取。更新底库需要重新构建。
- `solution/rnndescent/` 包含索引编排、图构建、BFS 重排、图搜索和候选重排。模块边界见 [RNNDescent 模块](solution/rnndescent/README.md)。
- `solution/rnndescent/discomputer/` 包含共用测距接口和平台入口。`flat/` 下是 FP32、FP16 和 INT16 的测距实现，`quantized/` 下是 INT8、UINT8 测距和邻居池。`tests/` 包含回归测试。

## 商用许可

本项目（外部调试版本）遵循以下商用许可条款：

1. **非商业用途许可**：未经华为公司书面授权，不得将本项目用于任何商业目的。

2. **内部调试使用**：本代码仅用于算法调试和性能测试，与华为公司内部商用版本存在较大差异。

3. **知识产权声明**：本项目包含的所有代码、算法和优化方法的知识产权归华为公司所有。

4. **修改限制**：未经授权，不得修改、分发、逆向工程或以任何形式传播本项目代码。

5. **免责声明**：本调试版本不保证性能、稳定性或正确性，仅供参考和学习使用。

如需商业授权或了解更多商用版本信息，请联系华为公司相关部门。

---

*本项目为华为算法大赛参赛作品(外部调试版本)，实现了高效的大规模向量检索算法。与内部商用有较大差别*
