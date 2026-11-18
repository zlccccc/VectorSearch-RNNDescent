# 开发与验证

从仓库根目录执行以下命令。程序和测试使用 C++17、OpenMP 和包含 LAPACK 的 OpenBLAS。Ubuntu 的 `libopenblas-dev` 提供 CBLAS 头文件和所需的 LAPACK 符号。测试优先使用本机 GoogleTest 和 GoogleMock 1.14 及以上版本，缺少 GoogleTest 时由 CMake 下载校验后的 1.17.0 源码。

**优化优先减少数据搬运，计算按速度选择精度。**

向量数据和邻居向量缓存尽量使用 8 位量化存储，按目标平台选择 INT8 或 UINT8。设计数据布局时优先考虑读取字节数、缓存局部性和复用，减少重复保存与拷贝。存储位宽和计算累加位宽分别选择，例如 8 位向量用 32 位累加。

计算优先选择吞吐量更高的 SIMD、整数乘加和近似方案。在任务指定的 Recall 与结果语义下，优先提升查询速度，数值精度按实际需求确定。取舍使用目标机器、多线程端到端 QPS、Recall 和内存占用验证，距离公式与结果 ID 按当前表示校验。

**用独立构建目录保留不同 SIMD 配置。**

~~~bash
cmake -S . -B build/avx2 -DCMAKE_BUILD_TYPE=Release \
  -DRNNDESCENT_SIMD=avx2 -DBUILD_TESTING=ON
cmake --build build/avx2 --parallel 4
ctest --test-dir build/avx2 --output-on-failure --no-tests=error
~~~

支持该指令集的 AArch64 机器可以选择 `-DRNNDESCENT_SIMD=neon`。x86-64 还支持 `native`、`avx2-uint8`、`avx-vnni` 和 `avx512`，对应要求见 [README](../README.md)。这些选项在编译时绑定内核，执行环境需要支持相应指令。可执行文件和库应与调用方使用一致的编译配置。

| CMake target | 内容 |
|---|---|
| `rnndescent_solution` | `Solution`、PCA 实现及传递给调用方的构建依赖 |
| `vector_search_app` | Benchmark 程序，输出名为 `algorithm` |
| `rnndescent_unit_tests` | 算距、缓存、CLI、数据读取和 PCA 测试 |
| `rnndescent_integration_tests` | 索引生命周期、多实例和召回率测试 |

只构建程序可以在配置时传入 `-DBUILD_TESTING=OFF`。安装于自定义前缀的 OpenBLAS 或 GoogleTest 可以通过 `CMAKE_PREFIX_PATH` 提供给 CMake。所有构建选项见 [CMakeLists.txt](../CMakeLists.txt)。

**CTest 根据标签运行相应范围。**

~~~bash
ctest --test-dir build/avx2 -N
ctest --test-dir build/avx2 -L '^unit$' --output-on-failure --no-tests=error
ctest --test-dir build/avx2 -L '^integration$' --output-on-failure --no-tests=error
ctest --test-dir build/avx2 -R 'PCAProjectionTest|RecallTest' --output-on-failure
~~~

测试自动发现单个 GoogleTest 用例。环境文件将测试进程设置为 `OMP_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=1` 和 `OMP_DYNAMIC=FALSE`。完整测试使用小规模内存数据即可运行，缓存测试需要系统可用内存覆盖程序的 2 GiB 预留。

| 改动范围 | 对应校验 |
|---|---|
| SIMD 点积、补齐、量化分数 | `SimdDistanceTest` 与 `SimdQuantizationTest` 的标量参考、非对齐维度和极端量化码 |
| 缓存布局、BFS 顺序、池分配 | `NeighborCacheLayoutTest`、`NeighborCacheTest` 和 `RowOrderMatchesQuantizingACopy` |
| PCA | `PCAProjectionTest`、启用 PCA 的 `SolutionTest` 与 `RecallTest` |
| 候选合并和查询流程 | `SolutionTest`、`RecallTest`，目标数据上的等 Recall 对照 |
| CLI、数据集或预设 | `BenchmarkCliTest` 与 `BenchmarkDataTest` |

单测中的有限数据验证算距公式与接口行为。性能结论还需要在目标数据、相同 Recall 和相同线程配置下测量。测量方法与日志字段见 [Benchmark 使用](../benches/README.md)。

**GitHub Actions 在远端执行 AVX2 单测。**

[工作流](../.github/workflows/unit-tests.yml) 在推送到 `main`、面向 `main` 的 PR 和 `workflow_dispatch` 时触发。Ubuntu 24.04 runner 安装 OpenBLAS、GoogleTest、GoogleMock 后，构建 `rnndescent_unit_tests` 并运行 `unit` 标签。可在 [Actions 页面](https://github.com/zlccccc/VectorSearch-RNNDescent/actions/workflows/unit-tests.yml) 选择 Run workflow 手动启动。

远端成功结果代表该次 AVX2 单测执行通过。其余 SIMD 配置和 `integration` 用例可用上面的本地命令在对应机器验证，报告时记录平台与测试范围。

**细粒度性能计时按需开启。**

~~~bash
cmake -S . -B build/profile -DCMAKE_BUILD_TYPE=Release \
  -DRNNDESCENT_SIMD=avx2 -DRNNDESCENT_PROFILE_SEARCH=ON -DBUILD_TESTING=OFF
cmake --build build/profile --target vector_search_app --parallel 4
~~~

该选项启用 `INTERNAL_CLOCK_TEST`，记录入口算距、邻居读取、候选更新和精排等线程累计耗时与计数。计时和聚合会增加运行开销，端到端 QPS 对照使用关闭该选项的构建。

**构建错误可从依赖和实际目标配置定位。**

| 现象 | 检查方式 |
|---|---|
| 找不到 `cblas.h` 或 `openblas` | 安装 `libopenblas-dev`，自定义安装时设置 CMake 搜索前缀 |
| 链接找不到 `dsyev_` | 检查 OpenBLAS 安装是否包含 LAPACK，PCA 训练会调用该符号 |
| 缺少 `GTest::gmock` | 安装 `libgmock-dev`，或配置 `-DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON` 使用源码依赖 |
| 运行时报告非法指令 | 对照 CPU 实际支持的指令集与 `RNNDESCENT_SIMD` 构建值 |
| 找不到数据文件 | 检查工作目录、`--dataset-dir` 和文件别名，或使用独立文件路径 |
| 切换依赖或指令集后沿用旧配置 | 从相应构建目录的 `CMakeCache.txt` 核对编译选项与依赖路径，使用新的构建子目录重新配置 |
