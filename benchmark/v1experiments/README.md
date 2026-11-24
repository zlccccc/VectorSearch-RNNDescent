# V1 实验

这里归档 `v1.0`（`1413447`）的性能基准和同数据集方案对照。

本机对照在 Core i7-12700KF、WSL 2 上用 Release 和 AVX-VNNI 运行。百万级邻居缓存对照也覆盖 Intel AVX-512 VNNI 和鲲鹏 NEON Dot Product 云机。查图统一用 UINT4，各组固定底库、查询、图、搜索预算及重排，对照组的 Top10 ID 和距离逐项一致。[数据集参数](cross-platform.md#数据集参数)

表中吞吐为 QPS/thread。配对变化取每轮吞吐比值的中位数，与两列中位数直接相除可能略有差别。

## 实验目录

- 10 万和百万级 SIFT、GIST 的跨设备、指令集和线程数结果。[跨设备与指令集](cross-platform.md)
- SIFT100K、GIST100K 的缓存开关和去重方式，以及本机与云机的百万级 0 缓存成绩。[邻居缓存与去重](neighbor-cache.md)
- SIFT1M、GIST1M 采用整表清零、按字清零和线性探测哈希的对照。[百万级 bitmap 清零](bitmap-reset.md)
- GIST100K 在固定线程位置和自由调度下的吞吐。[查询线程绑核](thread-affinity.md)

## 复测说明

上面的数据来自读取固定图的实验构建。仓库复测命令会建图，适合复测同参数下的趋势。哈希和按字清零由实验构建切换。
