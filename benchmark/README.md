# Benchmark 性能基准

2026-11-22，Release 构建，目标 Recall@10 ≥ 99%。

## 数据集参数

同一数据集复用同一张固定图。跨平台主对照的各平台和编码使用下表中的统一参数。建图统一 `T1=4`、`T2=15`。

| 数据集 | 建图维度 | 查图维度 | S / R / K0 | 入口 / beam | `search_L` / `refine_max` | 原始维度重排 |
|---|---|---|---:|---:|---:|---|
| SIFT100K | 原始 128 维 | PCA64 | 64 / 96 / 24 | 128 / 4 | 84 / 80 | FP32 直排 |
| GIST100K | PCA64 | PCA128 | 96 / 128 / 48 | 1024 / 4 | 176 / 176 | 8 位粗排 → FP32 Top40 |
| SIFT1M（本机） | 原始 128 维 | PCA64 | 80 / 112 / 32 | 256 / 8 | 121 / 104 | UINT4 粗排 → FP32 Top40 |
| GIST1M（本机） | PCA64 | PCA128 | 320 / 256 / 64 | 3072 / 6 | 377 / 377 | UINT4 粗排 → FP32 Top40 |

入口数用于查询。GIST1M 的固定图在建图时启用了局部三跳筛边。

## 跨平台对照

每格为 **1 线程 / 2 线程 QPS**，取 24 个交替批次的中位数。SIFT100K 的 Recall@10 为 99.126%–99.248%，GIST100K 为 99.140%–99.250%。各档邻居缓存覆盖率均为 100%。

| CPU | 指令集 | SIFT100K 8 位 | SIFT100K UINT4 | GIST100K 8 位 | GIST100K UINT4 |
|---|---|---:|---:|---:|---:|
| Core i7-12700KF | AVX2（INT8） | 19,894 / 39,974 | — | 5,702 / 10,768 | — |
| Core i7-12700KF | AVX2（UINT8） | 23,851 / 48,034 | — | 6,266 / 12,050 | — |
| Core i7-12700KF | AVX-VNNI | 24,783 / 50,052 | 25,197 / 51,035 | 6,774 / 13,076 | 7,629 / 14,962 |
| Intel 云机（family 6，model 106） | AVX2（INT8） | 17,381 / 34,849 | — | 4,577 / 8,895 | — |
| Intel 云机（family 6，model 106） | AVX2（UINT8） | 21,516 / 42,490 | — | 5,276 / 10,251 | — |
| Intel 云机（family 6，model 106） | AVX-512 VNNI | 22,321 / 43,941 | 22,032 / 43,542 | 5,629 / 11,038 | 6,316 / 12,410 |
| Huawei AArch64（0x48 / 0xd02） | NEON Dot Product | 18,954 / 37,931 | 20,514 / 40,337 | 5,706 / 10,451 | 5,816 / 10,770 |

AVX-VNNI 和 AVX-512 VNNI 的 8 位路径使用 UINT8，NEON 使用 INT8。本机双线程绑定到两个不同的核，ARM 双索引测试的系统内存预留为 1.75 GiB。

SIFT100K 使用 10,000 条查询。GIST100K 取 GIST1M 的前 10 万条底库，沿用 1,000 条查询并重算 ground truth。

| 补充配置 | 查图维度 | `search_L` / `refine_max` | Recall@10 | 1 线程 / 2 线程 QPS |
|---|---|---:|---:|---:|
| SIFT100K · ARM NEON INT8 | 原始 128 维 | 68 / 16 | 99.014% | 22,857 / 43,709 |

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
  --search-L 176 --refine-max 176 --graph-quantization uint4 \
  --rerank-quantization 8bit --fp32-refine-max 40 --neighbor-pool-gb 1 \
  --topk 10 --repeat 28 --warmup-topk 0 --output-iter 0
```

将 `--graph-quantization` 改为 `8bit` 可测 8 位查图。示例会重新建图，记录实测 Recall@10，去掉前 4 批再取 QPS 中位数。数据格式见 [数据集说明](../benches/README.md)。
