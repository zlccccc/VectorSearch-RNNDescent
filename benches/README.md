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

### 从 SIFT128 生成高维测试数据

复用同一份底库、查询和 ground truth，在 WSL 或 Linux 中运行下面的命令。输出目录需为新目录。生成脚本使用 Python 3.8 或更新版本的标准库。

~~~bash
python3 benches/expand_sift.py benches/datasets/sift benches/datasets/sift-expanded

for dim in 256 512 1024; do
  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
    RESULT_DIR="benches/results/sift-expanded-$dim" ./build/algorithm \
    --mode dataset --dataset-name sift --dataset-dir "benches/datasets/sift-expanded/$dim" \
    --K0 32 --num-initialize 32 --search-L 16 --refine-max 16 --beam-size 8 \
    --neighbor-pool-gb 4 --disable-pca --topk 10 --repeat 7 --output-iter 0
done
~~~

脚本把每条 128 维向量分别复制 2、4、8 份并拼接。平方 L2 距离同步乘以 2、4、8，精确近邻的 ID 和顺序保持一致，所以每档通过符号链接共享源目录的 ground truth。底库如果做过抽样，需使用这份子集对应的 ground truth。

这种数据适合固定近邻结构，查验 SIMD、邻居缓存和图搜索是否随维度变化而出现回归。复制坐标保留了原始数据的 128 个自由度，高维数据上的召回率仍需用真实高维数据集验证。SIFT1M 的三档扩维文件合计约占 7.3 GB。
