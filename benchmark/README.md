# Benchmark 性能基准

2026-11-22，i7-12700KF、AVX-VNNI Release、16 个查询线程，邻居缓存覆盖率 100%。

| 数据集 | 查图 | `search_L` | Recall@10 | QPS 中位数 |
|---|---|---:|---:|---:|
| SIFT100K | 原始 128 维，8 位 | 72 | 99.11%–99.14% | 194,667 |
| GIST1M | PCA64，8 位 | 538 | 99.00% | 14,231 |
| GIST1M | PCA64，UINT4 | 682 | 99.00% | 16,115 |

GIST1M 两档共用一张图，均用 UINT4 粗排和 FP32 Top40。

## 复测

按 [构建说明](../README.md#构建) 选择 `-DRNNDESCENT_SIMD=avx-vnni`，在仓库根目录运行。数据准备见 [数据集说明](../benches/README.md)。

```bash
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  ./build/algorithm --dataset-dir benches/datasets/gist --pca-out 64 \
  --S 320 --R 256 --K0 64 --num-initialize 3072 --beam-size 6 \
  --search-L 682 --refine-max 682 --graph-quantization uint4 \
  --rerank-quantization uint4 --fp32-refine-max 40 --neighbor-pool-gb 5 \
  --topk 10 --repeat 28 --warmup-topk 0 --output-iter 0
```

去掉前 4 批后取 QPS 中位数。命令会重新建图，比较时先把 Recall@10 调到 99%。
