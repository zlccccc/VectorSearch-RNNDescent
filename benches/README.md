## Bench Datasets

参考上游 `rnn-descent/benches`，数据集统一放在 `benches/datasets/<dataset>`。

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

| 模式 | 维度 | `search_L` | Recall`10 | QPS 中位数 |
|---|---:|---:|---:|---:|
| dimensions | 128 | 16 | 92.76% | 280,472 |
| dimensions | 256 | 16 | 92.82% | 172,202 |
| dimensions | 512 | 16 | 92.67% | 88,380 |
| dimensions | 1024 | 16 | 92.67% | 51,558 |
| recall | 128 | 16 | 92.74% | 284,389 |
| recall | 128 | 32 | 96.95% | 175,721 |
| recall | 128 | 64 | 99.25% | 119,327 |
| recall | 128 | 128 | 99.89% | 63,125 |
