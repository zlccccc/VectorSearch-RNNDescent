# Benchmark 性能基准

2026-11-22，Release 构建，目标 Recall@10 ≥ 99%。表中是每个平台和编码测得的最佳配置，QPS 取批次中位数。

## 跨平台对照

每台机器用 2 个查询线程。同一数据集的底库、查询、ground truth 和图在三台机器上相同。所有结果的邻居缓存覆盖率为 100%。x86 的 8 位列使用 UINT8，NEON 使用 INT8。

| 数据集 | CPU | 指令集 | 8 位查图 QPS / Recall / 维度 | UINT4 查图 QPS / Recall / 维度 |
|---|---|---|---|---|
| SIFT100K | Core i7-12700KF | AVX-VNNI | 55,065 / 99.030% / PCA64 | 52,667 / 99.020% / PCA64 |
| SIFT100K | Intel 云机（family 6，model 106） | AVX-512 VNNI | 48,784 / 99.010% / PCA64 | 47,428 / 99.055% / PCA64 |
| SIFT100K | Huawei AArch64（0x48 / 0xd02） | NEON Dot Product | 43,541 / 99.001% / PCA64 | 42,055 / 99.005% / 原始 128 维 |
| GIST100K | Core i7-12700KF | AVX-VNNI | 14,057 / 99.02% / PCA128 | 15,329 / 99.02% / PCA128 |
| GIST100K | Intel 云机（family 6，model 106） | AVX-512 VNNI | 12,184 / 99.02% / PCA128 | 12,641 / 99.05% / PCA128 |
| GIST100K | Huawei AArch64（0x48 / 0xd02） | NEON Dot Product | 11,396 / 99.02% / PCA128 | 11,285 / 99.01% / PCA128 |

SIFT100K 使用 10,000 条查询和原始 128 维 FP32 直排。GIST100K 取 GIST1M 的前 10 万条底库，沿用 1,000 条查询并重新计算 ground truth，统一用原始 960 维 UINT4 粗排和 FP32 Top40。每组交替测 24 轮。ARM 的 GIST100K PCA128 复测在双索引测试进程中使用 1.75 GiB 系统内存预留。

## 本机百万级数据

Core i7-12700KF，AVX-VNNI，16 个查询线程，邻居缓存覆盖率 100%。

| 数据集 | 查图 | 原始维度重排 | Recall@10 | QPS |
|---|---|---|---:|---:|
| SIFT1M | PCA64，UINT4 | UINT4 → FP32 Top40 | 99.006% | 133,221 |
| GIST1M | PCA128，UINT4 | UINT4 → FP32 Top40 | 99.01% | 18,249 |

## 复测

按 [构建说明](../README.md#构建) 选择 `avx-vnni`、`avx512` 或 `neon`。准备数据后，在仓库根目录运行以下 GIST100K 示例。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  ./build/algorithm --dataset-dir benches/datasets/gist100k --dataset-name gist \
  --pca-out 128 --S 96 --R 128 --K0 48 --num-initialize 1024 --beam-size 4 \
  --search-L 167 --refine-max 167 --graph-quantization uint4 \
  --rerank-quantization uint4 --fp32-refine-max 40 --neighbor-pool-gb 1 \
  --topk 10 --repeat 28 --warmup-topk 0 --output-iter 0
```

将 `--graph-quantization` 改为 `8bit` 可测 8 位查图。示例会重新建图，按实测 Recall@10 调整搜索预算，比较时去掉前 4 批。数据格式见 [数据集说明](../benches/README.md)。
