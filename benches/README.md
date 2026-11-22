## Bench Datasets

参考上游 `rnn-descent/benches`，数据集统一放在 `benches/datasets/<dataset>`。

构建命令见 [项目 README](../README.md)，参数定义和数据集预设见 [参数配置](../docs/configuration.md)。本页命令均从仓库根目录执行。

近期 QPS、Recall 和性能复测命令见 [Benchmark 性能基准](../benchmark/README.md)。

| 数据集 | 底库规模 | 原始维度 | 准备方式 |
|---|---:|---:|---|
| SIFT Small | 10,000 | 128 | 下载 `siftsmall` |
| SIFT1M | 1,000,000 | 128 | 下载 `sift` |
| GIST1M | 1,000,000 | 960 | 下载 `gist` |
| SIFT100K | 100,000 | 128 | 自行准备底库子集和相匹配的 ground truth |

下载示例：

```bash
sh benches/download_dataset.sh siftsmall
sh benches/download_dataset.sh sift
sh benches/download_dataset.sh gist
```

下载完成后，目录里会同时保留 TexMex 原始文件名，并创建当前工程默认识别的别名：

- `base.fvecs`
- `query.fvecs`
- `groundtruth.ivecs`
- `learn.fvecs`

然后可以直接运行：

```bash
./build/algorithm --mode dataset --dataset-dir benches/datasets/siftsmall --topk 10 --repeat 5
```

**文件按行存储，ID 使用从 0 开始的底库行号。**

| 格式 | 每行记录 | 用途 |
|---|---|---|
| `.fvecs` | 一个 32 位整数维度，后跟该维度数量的 FP32 分量 | 底库与查询 |
| `.bvecs` | 一个 32 位整数维度，后跟该维度数量的无符号字节分量 | 底库与查询，读取时转为 FP32 |
| `.ivecs` | 一个 32 位整数邻居数，后跟对应数量的 32 位整数 ID | 每条查询的有序 ground truth |

这些文件按 TexMex 数据的小端布局准备，程序在每条记录前读取维度并校验记录长度。底库与查询维度需一致，ground truth 的行数需与查询条数一致。

目录输入优先识别 `base.fvecs`、`base.bvecs`，其次是 `learn.fvecs`、`learn.bvecs`。查询可用 `query.fvecs`、`query.bvecs`、`queries.fvecs`、`queries.bvecs`，ground truth 可用 `groundtruth.ivecs`、`gt.ivecs`、`gnd.ivecs`。还可通过 `--base`、`--query`、`--gt` 指定独立文件。

底库做过截取或抽样时，重新计算相对该底库的 ground truth，并按当前行号保存 ID。例如 SIFT1M 的近邻列表可能包含 SIFT100K 子集以外的 ID，子集评估需使用自己的近邻列表。SIFT100K 目录准备好后，可运行 `DATASET=sift100k OPENBLAS_NUM_THREADS=1 ./run.sh`。

**GIST1M 可从显式的 PCA64 配置开始测量。**

~~~bash
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  RESULT_DIR=benches/results/gist-pca64 \
  ./build/algorithm --mode dataset --dataset-name gist \
  --dataset-dir benches/datasets/gist --topk 10 --repeat 7 \
  --S 320 --R 256 --K0 64 --num-initialize 3072 \
  --search-L 768 --beam-size 8 --refine-max 768 --fp32-refine-max 32 \
  --pca-out 64 --neighbor-pool-gb 4 --warmup-topk 0 --output-iter 0
~~~

这组参数默认先用原始 960 维 INT8/UINT8 粗排，再对 Top32 做 FP32 精排。可单独加 `--graph-quantization uint4` 或 `--rerank-quantization uint4` 对照 4 位查图或粗排。加 `--rerank-quantization fp32` 则对图搜索送出的所有候选做 FP32 精排。根据当前数据的 ground truth 调整 `search_L`、`refine_max` 和 `fp32_refine_max`，达到目标 Recall@10 后比较 QPS。多线程建图的调度会影响图结构，记录相同种子之外，还需保存每轮的 Recall 与缓存覆盖。

**报告同时记录配置、吞吐量和精度。**

程序读取整份 dataset 文件，`--test-iter` 和 `--exact-check-queries` 用于 random 模式。提供 ground truth 时，dataset 模式评估全部查询，Recall@K 的 K 为 `min(topk, ground_truth 每行邻居数)`。保持 ground truth 每行至少 Top K 条，便于直接比较目标 Recall。

| 输出 | 解释 |
|---|---|
| 控制台 `average test time`、`QPS` | 整批查询耗时除以查询条数，再取倒数得到吞吐量 |
| `latency_ms[i]` | 第 i 批总查询耗时除以查询条数，单位为毫秒 |
| `build_seconds` | 本次 `Solution::build` 的总耗时，包含配置的预热，数据文件读取另行完成 |
| `recall@10` | 例如 0.99 表示命中 ground truth 前 10 个 ID 的比例为 99% |
| `avg_distance_error`、`max_distance_error` | 返回 FP32 距离与逐维平方差参考结果之间的绝对误差 |
| `neighbors can be cached` | 实际邻居缓存覆盖率，输出在控制台日志中 |
| `latest.txt` 与时间戳报告 | 默认写入 `benches/results/<数据集名>/`，可用 `RESULT_DIR` 指定具体目录 |

多线程情况下，`latency_ms[i]` 表示摊到每条查询的批次时间，单条请求的端到端延迟需另行测量。每次启动程序都会新建索引。一次运行内的 `--repeat` 复用同一张图。比较两个版本时可以交替运行，多次独立建图后汇总；要求严格同图时，需要在单进程对照程序中复用图对象。

比较时记录提交、二进制、CPU、指令集、OpenMP 与 OpenBLAS 线程数、数据哈希、查询条数、PCA 维度、缓存预算和 Recall。先对齐 Recall，再汇总相邻 A/B 的 QPS 比值，能减少机器负载随时间变化的影响。

### 可复现的 SIFT 测试

先构建 benchmark 程序。输入目录需要包含 128 维的 `base.fvecs`、`query.fvecs`，以及对应的 `groundtruth.ivecs`。可以传入 SIFT1M 或已计算过 ground truth 的子集。脚本使用 Python 3.8 或更新版本的标准库。

~~~bash
python3 benches/run_sift.py dimensions benches/datasets/sift benches/results/sift-dimensions
python3 benches/run_sift.py recall benches/datasets/sift benches/results/sift-recall
~~~

| 模式 | 测试内容 |
|---|---|
| `dimensions` | 对比 128、256、512、1024 维，固定 32 个初始候选、`search_L=16`、精排上限 16 |
| `recall` | 固定 128 维，扫描 `search_L=16,32,64,128`。初始候选数取预算和 32 的较大值，精排上限等于预算 |

两种模式固定 `K0=32`、`beam-size=8`、Top 10，并关闭 PCA。默认使用 16 线程和 4 GiB 邻居池预算。每个配置独立建图 3 次，每次查询 7 遍，相邻轮倒序运行。每轮先取 7 个 QPS 的中位数，再汇报 3 轮中位数的中位值和范围。结果目录保存 `summary.csv`、可执行文件和原始底库、查询、ground truth 的 SHA-256、逐轮报告和日志。更换机器或缓存预算时，可以从报告核对实际缓存比例。

结果目录应为新目录。用 `--binary` 指定其他构建目录的可执行文件。`--threads`、`--pool-gib`、`--builds` 和 `--repeat` 可覆盖对应默认值。

### 从 SIFT128 生成高维测试数据

`dimensions` 模式自动生成或复用输入目录旁的 `<输入目录名>-expanded`。也可以用 `--expanded` 指定保留扩维文件的目录，或单独运行生成脚本。

~~~bash
python3 benches/expand_sift.py benches/datasets/sift benches/datasets/sift-expanded
~~~

脚本把每条 128 维向量分别复制 2、4、8 份并拼接。平方 L2 距离同步乘以 2、4、8，精确近邻的 ID 和顺序保持一致，所以每档通过符号链接共享源目录的 ground truth。底库如果做过抽样，需使用该子集对应的 ground truth。

这种数据适合固定近邻结构，查验 SIMD、邻居缓存和图搜索是否随维度变化而出现回归。复制坐标保留了原始数据的 128 个自由度。SIFT1M 的三档扩维文件合计约占 7.3 GB。

### 参考结果

以下是 `e5cd14e` 在 i7-12700KF 的 WSL 2 环境下，用 AVX2 跑同一份 SIFT 子集的结果。底库 10 万条、查询 1 万条，16 线程，各档邻居缓存覆盖率为 100%。QPS 随机器负载变化，表中数值用于对照。

| 模式 | 维度 | `search_L` | Recall@10 | QPS 中位数 |
|---|---:|---:|---:|---:|
| dimensions | 128 | 16 | 92.76% | 280,472 |
| dimensions | 256 | 16 | 92.82% | 172,202 |
| dimensions | 512 | 16 | 92.67% | 88,380 |
| dimensions | 1024 | 16 | 92.67% | 51,558 |
| recall | 128 | 16 | 92.74% | 284,389 |
| recall | 128 | 32 | 96.95% | 175,721 |
| recall | 128 | 64 | 99.25% | 119,327 |
| recall | 128 | 128 | 99.89% | 63,125 |
