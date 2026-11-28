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

## 查询性能剖析

~~~bash
cmake -S . -B build/profile -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DRNNDESCENT_SIMD=avx2 -DRNNDESCENT_PROFILE_SEARCH=ON -DBUILD_TESTING=OFF
cmake --build build/profile --target vector_search_app --parallel 4
~~~

索引实现按图构建、BFS 重排和图搜索拆分。新增模块的实现文件由 `rnndescent_solution` 目标统一编译，CMake 中按 `Index Sources` 和 `Index Headers` 分组。模块职责见 [RNNDescent 模块说明](../solution/rnndescent/README.md)。

计时和统计由 `SearchProfile` 模块负责，查询流程在各阶段边界记录采样。普通构建使用内联空操作，普通和计时构建共用批量搜索的 OpenMP 循环。`RNNDESCENT_PROFILE_SEARCH` 按测量区间累计输出以下三类指标。预热单独输出，正式结果在最后一轮查询后输出。[计时实现](../solution/rnndescent/search/SearchProfile.cpp)

| 日志 | 计量方式 | 指标 |
|---|---|---|
| `query_profile` | 调用方的墙钟毫秒 | 粗排量化及 FP32 范数准备、PCA 投影、查图量化、排序、查图及重排。各项除以 `total_ms` 得到端到端占比 |
| `graph_profile` | 各工作线程的墙钟毫秒之和 | 入口算距和筛选、邻接读取、邻居算距、候选池更新、线程内重排。`single_setup_ms` 和 `single_merge_ms` 单独记录单条并行的串行准备及最终合并 |
| `team_profile` | 并行区域与工作线程分别累计 | `wall_ms` 是进入并行区到返回，拆为首次进入 `entry_ms`、首次进入到最晚完成 `work_window_ms`、最晚完成到调用方返回 `join_ms`。`unused_slots` 记录未分到任务的线程席位 |

`tail_worker_ms` 累计各线程完成自己任务后到最后一个线程完成之间的时间，反映潜在的尾部等待。`worker_cpu_ms` 是线程从进入到完成任务时的 CPU 时间，`worker_off_cpu_ms` 是同一区间墙钟时间减 CPU 时间的估计，包含抢占和阻塞。`cpu_slots` 是可用的 CPU 时钟样本数，CPU 时间超过同线程墙钟区间的样本计入 `cpu_overrun_slots`，从这两项 CPU 时间统计中排除。线程累计时间可以超过端到端墙钟时间。正式 QPS 对照使用关闭 `RNNDESCENT_PROFILE_SEARCH` 的构建。

### CPU 火焰图

使用 `perf` 和 [FlameGraph 脚本](https://github.com/brendangregg/FlameGraph)。下面用和普通基准相同的参数跑查询，`--repeat` 用于累积样本。

~~~bash
OMP_NUM_THREADS=16 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
  python3 benches/profile_search.py --output benches/results/profile-sift \
  --flamegraph-dir /path/to/FlameGraph -- build/profile/algorithm \
  --mode dataset --dataset-dir benches/datasets/sift --query-batch-size 6 \
  --repeat 30 --output-iter 0
~~~

脚本默认采样 `cpu-clock:u`，用单调时钟标记筛出正式查询区间。`perf.data` 保存原始采样，`perf.script`、`perf-report.txt` 和 `cpu.svg` 只含正式查询。可用 `--perf` 指定工具路径，用 `--event cycles:u` 在支持硬件 PMU 的机器上按 CPU 周期采样。火焰图横向宽度按 CPU 样本数计算。等待可从 `team_profile` 查看，能访问 `sched:sched_switch` 的 Linux 环境还可用 `perf sched` 核对内核调度事件。[Linux perf record](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-record.txt) · [Linux perf script](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-script.txt)

**构建错误可从依赖和实际目标配置定位。**

| 现象 | 检查方式 |
|---|---|
| 找不到 `cblas.h` 或 `openblas` | 安装 `libopenblas-dev`，自定义安装时设置 CMake 搜索前缀 |
| 链接找不到 `dsyev_` | 检查 OpenBLAS 安装是否包含 LAPACK，PCA 训练会调用该符号 |
| 缺少 `GTest::gmock` | 安装 `libgmock-dev`，或配置 `-DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON` 使用源码依赖 |
| 运行时报告非法指令 | 对照 CPU 实际支持的指令集与 `RNNDESCENT_SIMD` 构建值 |
| 找不到数据文件 | 检查工作目录、`--dataset-dir` 和文件别名，或使用独立文件路径 |
| 切换依赖或指令集后沿用旧配置 | 从相应构建目录的 `CMakeCache.txt` 核对编译选项与依赖路径，使用新的构建子目录重新配置 |
