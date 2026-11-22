# 测距实现

共用接口由 [MyDistanceComputer.h](MyDistanceComputer.h) 定义。[PlatformSimdDistanceComputer.h](PlatformSimdDistanceComputer.h) 在编译期选择可用的 SIMD 内核，在创建索引时按阶段配置创建对应的 DistanceComputer。上层索引流程见 [算法与数据流](../../../docs/architecture.md)。

| 位置 | 职责 |
|---|---|
| [flat/FlatDistanceComputer.h](flat/FlatDistanceComputer.h) | 管理原始行或编码行、底库与查询平方范数，并组装平方距离 |
| `flat/` 下的平台内核 | FP32 点积、四路点积、NEON FP16 与 INT16 运算 |
| [flat/NeonDotProduct.h](flat/NeonDotProduct.h) | NEON FP32、FP16 和 INT16 共用的分块与尾部循环 |
| [quantized/QuantizedDistanceComputer.h](quantized/QuantizedDistanceComputer.h) | 共用的底库和查询存储、单条与批量算距、邻居缓存接口，以及 8 位编码策略 |
| [quantized/UInt4Quantization.h](quantized/UInt4Quantization.h) | 4 位查图和粗排的量化、半字节布局和分数换算 |
| [quantized/QuantizedKernelBase.h](quantized/QuantizedKernelBase.h) | 共用的分块打包、缓存遍历和平方 L2 写回 |
| `quantized/` 下的平台内核 | 连续读和缓存读的点积 |
| [quantized/NeonInt8Kernel.h](quantized/NeonInt8Kernel.h) | NEON INT8 点积与四路邻居缓存算距 |
| [quantized/Avx2UInt8Kernel.h](quantized/Avx2UInt8Kernel.h) | AVX2 UINT8 内核，以及与 AVX-VNNI 共用的 AVX2 运算和批量循环 |
| [quantized/AvxVnniUInt8Kernel.h](quantized/AvxVnniUInt8Kernel.h) | AVX-VNNI 内核和字节乘加实现 |
| [quantized/Avx2SimdDistanceComputerInt8.h](quantized/Avx2SimdDistanceComputerInt8.h) | AVX2 INT8 四行内核 |
| [quantized/Avx512SimdDistanceComputerUInt8.h](quantized/Avx512SimdDistanceComputerUInt8.h) | AVX-512 UINT8 的缓存打包、GEMV 与四行算距 |
| [quantized/NeighborsStorageBase.h](quantized/NeighborsStorageBase.h) | 邻居池、邻接 ID 和范数的存储与分配 |
| [utils.h](utils.h) | 对齐分配器、访问表和候选表示 |

**FlatDistanceComputer 直接引用 FP32 行，并为低精度变体保存编码行。**

默认精排使用当前平台的 FP32 内核，引用调用方的原始底库。底库范数用 `Kernel::dot(row, row, d)` 计算，查询范数由 `Kernel::query_norm` 计算。距离按照 `norm(query) + norm(base) - 2 * dot(query, base)` 组装。

默认对原始维度 8 位粗排留下的 Top32 做 FP32 算距。粗排底库按 BFS 顺序编码，保留的候选在 FP32 精排前映射回原始行号。查图在原始维度且两阶段均用 8 位时可直接复用查图分数。`--rerank-quantization fp32` 或 `--fp32-refine-max 0` 将全部图候选直接交给 FP32。

`--rerank-quantization uint4` 显式启用 4 位粗排。编码先减去底库逐维均值，再为每行计算偏移和步长。每行保存按 SIMD 块补齐的半字节码以及 12 字节参数及范数。查询按向量量化到有符号 8 位，四路内核在寄存器里解包底库码并执行字节点积。粗排按重建的平方 L2 排序，省略对同一查询恒定的查询平方范数。

`--graph-quantization uint4` 显式启用 4 位查图。该编码用采样误差选择底库共享步长。每行保存按 SIMD 块补齐的量化码和 4 字节平方范数。查询以更细的步长量化到有符号字节。查图和粗排共用 `PackedUInt4Layout`。同一个字节的低、高半字节分别存储两个连续特征块的同一位置，由内核指定特征块宽度。逐行编码和分数组装在公共层，内核负责点积和邻居缓存分块。AVX-VNNI 和 AVX-512 按邻居数划分 16 路主块和四路尾块，各块按实际码长遍历。AVX-512 的四路内核每次将一块 32 字节码展开到 64 字节执行点积。NEON 使用四路成组加载和 `SDOT`。未缓存的邻居按 ID 从 4 位底库读码。[AVX-VNNI](quantized/AvxVnniUInt4Kernel.h) · [AVX-512](quantized/Avx512UInt4Kernel.h) · [NEON](quantized/NeonUInt4Kernel.h)

UINT4 的 DistanceComputer 在初始化时确定统一的补齐码长。AVX-VNNI 和 AVX-512 每个特征块覆盖 64 维，NEON 每块覆盖 32 维。底库及查询都以零码补齐，邻居缓存复制完整码块。点积循环按整块推进，x86 的 16 路缓存主块也按固定步长顺序读取查询和缓存码。

NEON FP16 把行转换为半精度，并用 FP32 累加。NEON INT16 保存编码行，用 64 位整数累加并乘量化步长的平方。CBLAS FP32 变体使用 `cblas_sdot`，测试中用于单条与批量距离校验。默认精排由平台工厂选择 SIMD FP32。

**8 位量化使用逐维中心与索引级统一步长。**

对每一维 `j`，设训练底库的最小值和最大值为 `min_j`、`max_j`。

~~~text
center_j = (max_j + min_j) / 2
range = max_j((max_j - min_j) / 2)

signed_scale = range / 127
signed_code_j = clamp(round((x_j - center_j) / signed_scale), -127, 127)

unsigned_scale = range / 63
unsigned_code_j = clamp(round((x_j - center_j) / unsigned_scale), -63, 63) + 63
~~~

`range` 为 0 时步长取 1。量化码分别落在 INT8 的 `[-127, 127]` 和 UINT8 的 `[0, 126]`。相同索引中的查询复用底库训练的中心和步长，超出范围的分量会截断。UINT8 的平移在平方差中相消，而且所有码都能作为有符号字节的非负值参与字节乘加。

~~~text
score(q, x) = sum_j(code(q)_j - code(x)_j)^2
            = squared_norm(q) + squared_norm(x) - 2 * dot(q, x)

输入单位下的近似平方 L2 = score * scale * scale
~~~

建图、查图和邻居缓存的返回分数使用码空间单位。同一个索引的统一 `scale²` 保持排序，最终精排返回原始 FP32 单位下的平方 L2。实现对所有 8 位内核统一限制最多 33,286 维，控制码空间平方差和中间值的 32 位累加范围。

**补齐后的行宽同时决定连续读与缓存读。**

| 构建目标 | 量化格式 | `block_bytes` | 点积操作 |
|---|---|---:|---|
| `avx2` | INT8 | 16 | 扩位到 INT16 后用 `VPMADDWD` |
| `avx2-uint8` | UINT8 | 32 | `VPMADDUBSW` 和 `VPMADDWD` |
| `avx-vnni` | UINT8 | 32 | 256 位 `VPDPBUSD` |
| `avx512` | UINT8 | 64 | 512 位 `VPDPBUSD` |
| `neon` | INT8 | 16 | `SDOT` |

`dimension()` 返回逻辑维度。`code_stride()` 把逻辑维度向上补齐到 `block_bytes` 的倍数。底库与查询只编码逻辑维度，补齐字节均为 0。点积循环处理完整 SIMD 块，范数与填充共同保证补齐部分的平方差为 0。

BFS 重排通过 `row_order` 指定每个图节点对应的原始行，量化器直接按该顺序编码。`copy_index` 默认复制逻辑码，邻居缓存通过 `include_padding=true` 复制完整行宽。

默认缓存中四条邻居的第一个 SIMD 块连续存储，再放四条的第二个块，依次排列。AVX2 UINT8 和 AVX-VNNI 用 `Tile16` 把完整的 16 行按每行 4 字节交错打包。查询广播对应的 4 字节，每条 SIMD lane 累加一个邻居的点积。剩余 4、8、12 行以 `Tile4` 为单位，按默认四路布局打包和算距。两种布局都按邻接表原顺序写回平方 L2。打包只要求邻居数为 4 的倍数。

`QuantizedKernelBase<Format, BlockBytes, Rows, InterleaveBytes>` 由后端指定码类型、编码补齐宽度、每块邻居数和每次交错的字节数。基类处理已分好的完整块，混合块由外层切分。GEMV 主块采用 16 行、4 字节交错，各后端的四行块按各自的 SIMD 块宽交错。基类打包只需暂存一行编码，各后端提供自己的 SIMD 点积和归约。

AVX2 UINT8 和 AVX-VNNI 各有一个 kernel 类型，`Tile4` 和 `Tile16` 指定打包形状。算距入口直接遍历 16 行主块和四行尾块，整个邻接表共用一次广播的查询平方范数。`compute4` 合并归约四个点积，再用 SIMD 同时写出四个分数。后端在编译时选择自己的字节乘加。

这两个后端把底库和查询的编码长度向上补零到 32 字节的倍数。32 和 64 字节步长走固定长度循环。128 字节步长在 AVX2 UINT8 展开四行尾块，在 AVX-VNNI 使用独立的函数展开 16 行主块和四行尾块。通用的 16 行主块每次处理 16 字节，用四组独立的累加器缩短点积的指令依赖链。查询循环沿补齐后的跨度读取，维度尾部无需单独处理。

AVX-512 UINT8 在 64 字节步长展开 16 行 GEMV，四行块用 AVX-512 lane 重排把四个点积一起归约，再向量化合成平方 L2。

NEON INT8 的邻居缓存按四行交错。每步用 `vld1q_s8_x4` 成组加载四条邻居的 16 字节块，用 `SDOT` 累加点积，再用三次成对相加合并归约，并一次写出四个平方 L2。

池预算耗尽时，邻接表保留 ID，算距通过底库码的四路接口完成。缓存内核优化的整体收益取决于查询花在缓存内算距的时间占比。

**扩展量化内核时沿用同一套量化与缓存协议。**

| 内核成员 | 约定 |
|---|---|
| `block_bytes` | 单条码的 SIMD 块宽，用于连续读、补齐和默认缓存布局 |
| `dot(q, x, stride)` | 按传入的码长返回整数点积 |
| `dot4(q, x0, x1, x2, x3, stride)` | 按传入的码长返回同一查询与四条码的整数点积 |
| `pack_neighbor_cache(...)` | 可选的建图打包入口，提供时由内核决定缓存布局 |
| `cached_distances(...)` | 按对应的缓存布局写入每个邻居的码空间平方 L2 |

用内核和编码策略实例化 `QuantizedDistanceComputer<Kernel, Encoding>`。策略提供底库和查询的行宽、行参数、训练和编码方法，以及整数点积到分数的换算。8 位默认使用 `ByteQuantization`，4 位查图使用共享步长，4 位粗排使用逐行仿射参数。邻居容器通过 DistanceComputer 调用缓存打包、缓存算距和四路回退。Flat 内核提供 `Code`、`Dot`、`dot`、`dot4`，FP32 内核还提供 `query_norm`，编码行内核提供其转换和缩放字段。

现有 [SIMD 测试](../../../tests/simd_test.cpp) 对照标量结果，覆盖单条、四路、非对齐维度、极端码、BFS 行重排和缓存回退。修改计算或布局后，在对应指令集机器上运行这些用例，并通过相同 Recall 的 Benchmark 检查查询速度。构建和测试命令见 [开发与验证](../../../docs/development.md)。
