# 测距实现

共用接口由 [MyDistanceComputer.h](MyDistanceComputer.h) 定义，[PlatformSimdDistanceComputer.h](PlatformSimdDistanceComputer.h) 在编译期选择建图、查图、邻居容器与精排内核。上层索引流程见 [算法与数据流](../../../docs/architecture.md)。

| 位置 | 职责 |
|---|---|
| [flat/FlatDistanceComputer.h](flat/FlatDistanceComputer.h) | 管理原始行或编码行、底库与查询平方范数，并组装平方距离 |
| `flat/` 下的平台内核 | FP32 点积、四路点积、NEON FP16 与 INT16 运算 |
| [flat/NeonDotProduct.h](flat/NeonDotProduct.h) | NEON FP32、FP16 和 INT16 共用的分块与尾部循环 |
| [quantized/QuantizedDistanceComputer.h](quantized/QuantizedDistanceComputer.h) | 训练 8 位量化参数、编码底库与查询，并计算码空间平方距离 |
| `quantized/` 下的平台内核 | INT8 或 UINT8 连续读和缓存读的点积 |
| [quantized/NeighborsStorageBase.h](quantized/NeighborsStorageBase.h) | 每个索引的邻居池、邻接 ID、范数和四路缓存布局 |
| [utils.h](utils.h) | 对齐分配器、访问表和候选表示 |

**FlatDistanceComputer 直接引用 FP32 行，并为低精度变体保存编码行。**

默认精排使用当前平台的 FP32 内核，引用调用方的原始底库。底库范数用 `Kernel::dot(row, row, d)` 计算，查询范数由 `Kernel::query_norm` 计算。距离按照 `norm(query) + norm(base) - 2 * dot(query, base)` 组装。

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

缓存中四条邻居的第一个 SIMD 块连续存储，再放四条的第二个块，依次排列。查询一个邻接表时，`cached_distances` 根据相同块宽读码，结合预存范数生成距离。池预算耗尽时，邻接表保留 ID，算距通过底库码的四路接口完成。

**扩展量化内核时沿用同一套量化与缓存协议。**

| 内核成员 | 约定 |
|---|---|
| `Quantization` | 提供 `Code`、`max_scale` 和 `encode` |
| `block_bytes` | 单条码的 SIMD 块宽，连续读、补齐和缓存布局共用 |
| `dot(a, b, stride)` | 返回两条码的整数点积 |
| `dot4(q, x0, x1, x2, x3, stride)` | 返回同一查询与四条码的点积 |
| `cached_distances(...)` | 读取四路交错缓存并写入每个邻居的码空间平方 L2 |

用内核实例化 `QuantizedDistanceComputer<Kernel>` 和 `QuantizedNeighbors<Kernel>`，在平台入口同时绑定距离计算器与邻居容器。Flat 内核提供 `Code`、`Dot`、`dot`、`dot4`，FP32 内核还提供 `query_norm`；编码行内核提供其转换和缩放字段。

现有 [SIMD 测试](../../../tests/simd_test.cpp) 对照标量结果，覆盖单条、四路、非对齐维度、极端码、BFS 行重排和缓存回退。修改计算或布局后，在对应指令集机器上运行这些用例，并通过相同 Recall 的 Benchmark 检查查询速度。构建和测试命令见 [开发与验证](../../../docs/development.md)。
