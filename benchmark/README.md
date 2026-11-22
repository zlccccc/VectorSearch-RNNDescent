# Benchmark 性能基准

2026-11-22，Release 构建，目标 Recall@10 ≥ 99%。`QPS/thread` 是整批查询 QPS 除以实际查询线程数，保留一位小数。

## 数据集参数

同一数据集复用同一张固定图。每组跨平台对照使用相同的查图维度、搜索预算和重排设置。建图统一 `T1=4`、`T2=15`。

| 数据集 | 建图维度 | 查图维度 | S / R / K0 | 入口 / beam | `search_L` / `refine_max` | 原始维度重排 |
|---|---|---|---:|---:|---:|---|
| SIFT100K | 原始 128 维 | PCA64 | 64 / 96 / 24 | 128 / 4 | 84 / 80 | FP32 直排 |
| GIST100K | PCA64 | PCA128 | 96 / 128 / 48 | 1024 / 4 | 176 / 176 | 8 位粗排 → FP32 Top40 |
| SIFT1M（16 线程） | 原始 128 维 | PCA64 | 80 / 112 / 32 | 256 / 8 | 121 / 104 | UINT4 粗排 → FP32 Top40 |
| SIFT1M（1 / 2 线程） | 原始 128 维 | PCA64 | 80 / 112 / 32 | 256 / 8 | 128 / 120 | UINT4 粗排 → FP32 Top40 |
| GIST1M（本机） | PCA64 | PCA128 | 320 / 256 / 64 | 3072 / 6 | 377 / 377 | UINT4 粗排 → FP32 Top40 |

入口数用于查询。GIST1M 的固定图在建图时启用了局部三跳筛边。

## 10 万条固定预算对照

每格为 **1 线程 / 2 线程的 QPS/thread**。交替运行各档，每档取 24 个查询批次的中位数。SIFT100K 的 Recall@10 为 99.126%–99.248%，GIST100K 为 99.140%–99.250%。各档邻居缓存覆盖率均为 100%。

| CPU | 指令集 | SIFT100K 8 位 | SIFT100K UINT4 | GIST100K 8 位 | GIST100K UINT4 |
|---|---|---:|---:|---:|---:|
| Core i7-12700KF | AVX2（INT8） | 19,894.1 / 19,987.0 | — | 5,701.9 / 5,383.9 | — |
| Core i7-12700KF | AVX2（UINT8） | 23,850.7 / 24,017.2 | — | 6,265.8 / 6,025.2 | — |
| Core i7-12700KF | AVX-VNNI | 24,783.1 / 25,025.8 | 25,196.9 / 25,517.5 | 6,773.8 / 6,537.9 | 7,629.3 / 7,481.1 |
| Intel 云机（family 6，model 106） | AVX2（INT8） | 17,381.5 / 17,424.5 | — | 4,576.9 / 4,447.5 | — |
| Intel 云机（family 6，model 106） | AVX2（UINT8） | 21,515.7 / 21,244.9 | — | 5,276.5 / 5,125.4 | — |
| Intel 云机（family 6，model 106） | AVX-512 VNNI | 22,321.0 / 21,970.5 | 22,032.4 / 21,771.1 | 5,629.5 / 5,518.9 | 6,316.3 / 6,205.1 |
| Huawei AArch64（0x48 / 0xd02） | NEON Dot Product | 18,953.9 / 18,965.3 | 20,514.1 / 20,168.4 | 5,706.2 / 5,225.6 | 5,816.0 / 5,385.2 |

AVX-VNNI 和 AVX-512 VNNI 的 8 位路径使用 UINT8，NEON 使用 INT8。本机双线程绑定到两个不同的核，ARM 双索引测试的系统内存预留为 1.75 GiB。

SIFT100K 使用 10,000 条查询。GIST100K 取 GIST1M 的前 10 万条底库，沿用 1,000 条查询并重算 ground truth。

## 百万级结果

各档用 UINT4 查图和粗排，再做 FP32 Top40。SIFT1M 的 1 / 2 线程档在本机和云机上共用底库、查询、固定图和搜索参数。1 / 2 线程交替测，每档取 8 批的中位数。

全缓存用 `--neighbor-pool-gb 5`，关闭邻居向量缓存用 `--neighbor-pool-gb 0`。两台云机按默认的 2 GiB 系统内存预留测试时，SIFT1M 实际缓存覆盖率为 0%。表中的云机两档使用独立测试构建，将系统内存预留设为 1 GiB 后同轮复测。索引就绪 RSS 在开始查询前读取。峰值 RSS 包含底库加载、PCA、固定图装配和查询。

| 数据集 | CPU / 指令集 | 查询线程 | 邻居缓存覆盖率 | Recall@10 | QPS/thread | 索引就绪 RSS | 峰值 RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| SIFT1M | Core i7-12700KF / AVX-VNNI | 16 | 100% | 99.006% | 8,326.3 | — | — |
| SIFT1M | Core i7-12700KF / AVX-VNNI | 1 / 2 | 100% | 99.122% | 10,955.3 / 11,403.2 | 1,565 MiB | 1,825 MiB |
| SIFT1M | Core i7-12700KF / AVX-VNNI | 1 / 2 | 0% | 99.122% | 7,544.1 / 7,775.1 | 771 MiB | 1,031 MiB |
| SIFT1M | Intel 云机 / AVX-512 VNNI | 1 / 2 | 100% | 99.116% | 9,599.1 / 9,442.2 | 1,565 MiB | 1,825 MiB |
| SIFT1M | Intel 云机 / AVX-512 VNNI | 1 / 2 | 0% | 99.116% | 6,841.3 / 7,005.0 | 771 MiB | 1,032 MiB |
| SIFT1M | ARM 云机 / NEON Dot Product | 1 / 2 | 100% | 99.117% | 9,099.8 / 8,867.7 | 1,562 MiB | 1,822 MiB |
| SIFT1M | ARM 云机 / NEON Dot Product | 1 / 2 | 0% | 99.117% | 7,288.3 / 7,131.0 | 770 MiB | 1,030 MiB |
| GIST1M | Core i7-12700KF / AVX-VNNI | 16 | 100% | 99.01% | 1,140.6 | — | — |
| GIST1M | Core i7-12700KF / AVX-VNNI | 1 / 2 | 100% | 99.01% | 2,676.4 / 2,619.3 | 6,486 MiB | 6,636 MiB |
| GIST1M | Core i7-12700KF / AVX-VNNI | 1 / 2 | 0% | 99.01% | 1,578.3 / 1,657.4 | 4,399 MiB | 4,549 MiB |

## 复测

按 [构建说明](../README.md#构建) 选择指令集。设 `OMP_NUM_THREADS=1` 或 `2`，然后在仓库根目录运行以下 GIST100K 示例。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  ./build/algorithm --dataset-dir benches/datasets/gist100k --dataset-name gist \
  --pca-out 128 --S 96 --R 128 --K0 48 --num-initialize 1024 --beam-size 4 \
  --search-L 176 --refine-max 176 --graph-quantization uint4 \
  --rerank-quantization 8bit --fp32-refine-max 40 --neighbor-pool-gb 1 \
  --topk 10 --repeat 28 --warmup-topk 0 --output-iter 0
```

将 `--graph-quantization` 改为 `8bit` 可测 8 位查图。示例会重新建图，记录实测 Recall@10，去掉前 4 批，取 QPS 中位数并除以查询线程数。数据格式见 [数据集说明](../benches/README.md)。
