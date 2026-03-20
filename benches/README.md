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
