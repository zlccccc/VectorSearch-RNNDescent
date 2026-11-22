# Benchmark 性能基准

2026-11-22，Release 构建，各档 Recall@10 均超过 99%。QPS 取 24 个查询批次的中位数。

## 跨平台对照

每格为 **1 线程 / 2 线程 QPS**。同一数据集在三台机器上共用底库、查询、ground truth 和图。同组的两种线程数交替测，使用同一组搜索参数，邻居缓存覆盖率均为 100%。

| CPU | 指令集 | SIFT100K 8 位 | SIFT100K UINT4 | GIST100K 8 位 | GIST100K UINT4 |
|---|---|---:|---:|---:|---:|
| Core i7-12700KF | AVX2（INT8） | 21,019 / 42,867 | — | 5,674 / 10,851 | — |
| Core i7-12700KF | AVX2（UINT8） | 26,642 / 55,258 | — | 6,599 / 12,575 | — |
| Core i7-12700KF | AVX-VNNI | 26,739 / 54,455 | 24,420 / 49,488 | 7,022 / 13,524 | 7,548 / 14,828 |
| Intel 云机（family 6，model 106） | AVX2（INT8） | 20,058 / 39,723 | — | 5,155 / 10,131 | — |
| Intel 云机（family 6，model 106） | AVX2（UINT8） | 24,364 / 48,299 | — | 6,003 / 11,554 | — |
| Intel 云机（family 6，model 106） | AVX-512 VNNI | 24,433 / 48,069 | 22,747 / 44,870 | 6,283 / 12,171 | 6,608 / 12,664 |
| Huawei AArch64（0x48 / 0xd02） | NEON Dot Product | 21,745 / 43,608 | 21,607 / 42,362 | 6,251 / 11,306 | 6,079 / 11,301 |

AVX-VNNI 和 AVX-512 VNNI 的 8 位列使用 UINT8，NEON 使用 INT8。本机把 OpenMP 线程绑定到两个不同的核。ARM 的双索引测试进程使用 1.75 GiB 系统内存预留。

SIFT100K 用 10,000 条查询、PCA64 查图和原始 128 维 FP32 直排，NEON 的 UINT4 档使用原始 128 维查图。GIST100K 取 GIST1M 的前 10 万条底库，沿用 1,000 条查询并重算 ground truth。各档使用 PCA128 查图及原始 960 维粗排后 FP32 Top40，AVX2 用 8 位粗排，其余用 UINT4 粗排。

## 本机百万级数据

Core i7-12700KF，AVX-VNNI，16 个查询线程，邻居缓存覆盖率 100%。

| 数据集 | 查图 | 原始维度重排 | Recall@10 | QPS |
|---|---|---|---:|---:|
| SIFT1M | PCA64，UINT4 | UINT4 → FP32 Top40 | 99.006% | 133,221 |
| GIST1M | PCA128，UINT4 | UINT4 → FP32 Top40 | 99.01% | 18,249 |

## 复测

按 [构建说明](../README.md#构建) 选择指令集。设 `OMP_NUM_THREADS=1` 或 `2`，然后在仓库根目录运行以下 GIST100K 示例。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  ./build/algorithm --dataset-dir benches/datasets/gist100k --dataset-name gist \
  --pca-out 128 --S 96 --R 128 --K0 48 --num-initialize 1024 --beam-size 4 \
  --search-L 167 --refine-max 167 --graph-quantization uint4 \
  --rerank-quantization uint4 --fp32-refine-max 40 --neighbor-pool-gb 1 \
  --topk 10 --repeat 28 --warmup-topk 0 --output-iter 0
```

将 `--graph-quantization` 改为 `8bit` 可测 8 位查图。AVX2 同时使用 `--rerank-quantization 8bit`。示例会重新建图，按实测 Recall@10 调整搜索预算，去掉前 4 批再取中位数。数据格式见 [数据集说明](../benches/README.md)。
