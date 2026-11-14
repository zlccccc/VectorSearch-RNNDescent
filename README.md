# RNNDescent 向量检索算法

本项目实现了基于 RNNDescent 的近似最近邻检索，使用平方 L2 距离。构建阶段迭代生成近邻图并重排节点。查询阶段用 8 位量化向量在图中筛选候选，再用原始 FP32 向量重新排序。

## 许可声明

本项目（外部调试版本）的知识产权归华为公司所有，未经授权不得用于商业目的。请参阅文末的[商用许可](#商用许可)。

## 构建

在 Linux 或 WSL 2 中使用支持 C++17 和 OpenMP 的编译器、CMake 3.23 或更新版本、Faiss 和 OpenBLAS。以下是 Ubuntu 24.04 的构建示例。已有 Faiss 安装时，把 `CMAKE_PREFIX_PATH` 指向其安装前缀即可。

```bash
sudo apt install build-essential cmake git libopenblas-dev
git clone --depth 1 --branch v1.8.0 https://github.com/facebookresearch/faiss.git /tmp/rnndescent-faiss
cmake -S /tmp/rnndescent-faiss -B /tmp/rnndescent-faiss/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local/faiss" \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON -DBLA_VENDOR=OpenBLAS
cmake --build /tmp/rnndescent-faiss/build --parallel 4
cmake --install /tmp/rnndescent-faiss/build
```

回到本项目根目录构建，程序位于 `build/algorithm`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/.local/faiss"
cmake --build build --parallel 4
```

用 `-DRNNDESCENT_SIMD=<值>` 选择编译目标。切换指令集时建议用不同的构建目录，以便分别保留可执行文件和测试结果。

| 值 | 平台与指令集 |
|---|---|
| `native`（默认） | x86-64 按构建机选择 AVX2 或 AVX-512 VNNI，AArch64 使用 NEON Dot Product |
| `avx2` | x86-64，要求 AVX2 和 FMA |
| `avx512` | x86-64，要求 AVX-512F、BW、VL 和 VNNI |
| `neon` | AArch64，要求 NEON Dot Product（Linux 的 `asimddp`） |

指令集在编译时选定。运行测试的机器也需要支持所选指令集。

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
| `BenchmarkCliTest`、`BenchmarkDataTest` | 校验 CLI、显式随机模式、邻居池预算，以及 fvecs/bvecs/ivecs 的读取和坏文件处理 |
| `SolutionTest` | 小数据集建图、查询、重建和重置，用穷举校验 Top K ID 和精排距离，覆盖 PCA、缓存开关和两个索引交替查询 |
| `RecallTest` | 固定随机种子，限制搜索预算，校验 Recall`10、缓存开关下的结果一致性和精排距离 |

只构建 benchmark 程序时，可在配置阶段传入 `-DBUILD_TESTING=OFF` 跳过测试依赖。

## Benchmark

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

自定义数据集和随机向量采用通用参数。搜索预算可通过 `--search-L`、`--num-initialize` 和 `--refine-max` 等参数显式指定。SIFT、SIFT Small 和 GIST 在 benchmark 中有各自的数据集预设，命令行参数优先。可用 `./run.sh` 增量构建并运行 benchmark，默认使用 `benches/datasets/sift`。也可以在脚本后传入完整的 benchmark 参数，并通过 `BUILD_DIR` 使用已有的 CMake 构建目录。

要固定近邻结构测试向量变宽的影响，或把 Recall@10 拉离饱和区再比较 QPS，可运行 [SIFT benchmark 脚本](benches/README.md#可复现的-sift-测试)。脚本支持 128、256、512、1024 维对比和低搜索预算扫描，每档汇总多次独立建图的结果。

### 读取结果

程序默认写到 `benches/results/<数据集名>/`，随机测试写到 `benches/results/random-<维度>d/`。每次生成带时间戳的报告，同时更新 `latest.txt`。可用 `RESULT_DIR` 为每个版本或每轮测试指定单独的目录。

`latency_ms[i]` 是第 `i` 轮每条查询的平均毫秒数，对应 `QPS = 1000 / latency_ms[i]`。建图时间包含配置的预热。随机测试用 `--exact-check-queries` 指定穷举校验的查询条数，数据集模式提供 ground truth 后可用其计算 Recall@K。

## 代码入口

- `benchmark/` 提供命令行、数据读取、评估和报告。
- `solution/solution.h` 提供 `Solution::build`、`search` 和 `reset`。底库向量供 FP32 精排引用，需在索引使用期间保持有效。结果按查询行连续存储，距离可通过 `distance_buffer()` 读取。更新底库需要重新构建。
- `solution/rnndescent/discomputer/` 包含平台距离内核和邻居池，`tests/` 包含回归测试。

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
