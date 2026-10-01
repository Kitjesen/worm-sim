# 整机带状 Schur CPU/CUDA 对照

测试主机：RTX 5090 32607 MiB、Xeon Platinum 8470Q、CUDA 12.8、PyTorch 2.8.0。每个算例为 8 片独立钢带、`dt=2 ms`、1 个时间步、0 mm 收绳；CPU 与 CUDA 使用同一份输入。CPU 将每片局部 Hessian 按节点重排为半带宽 10 的带状块，再解 12×12 Schur 补；CUDA 当前批量解局部 dense FP64 块和 12×12 Schur 补。两条路径的几何、Sano 材料、接触、装配和线搜索都在 CPU。

| 每片节点 | Newton 维度 | CPU 带状 (s) | CUDA Schur (s) | CPU/CUDA | 最大 `q` 差 (m) | 最大端板 COM 差 (m) | CUDA 求解器累计 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 180 | 3.4508 | 3.8212 | 0.903 | 5.39e-14 | 3.24e-16 | 0.4806 |
| 17 | 436 | 7.3640 | 7.8915 | 0.933 | 4.71e-14 | 9.96e-16 | 0.5036 |
| 33 | 948 | 31.5546 | 31.1078 | 1.014 | 4.36e-13 | 3.24e-16 | 0.5511 |

`CPU/CUDA` 大于 1 表示 CUDA 较快。三组结果的位移和端板质心差异均远小于 Newton 残差（约 `1e-6`），说明带状 CPU 路径和 CUDA Schur 路径保持了同一接触、绳索和惯性结果。当前单次计时中 N9/N17 的 GPU 传输和同步开销超过线性代数收益，N33 约快 1.4%；重复运行时应报告中位数而不是单次 wall time。

每组的 `summary.json` 与 `trajectory.npz` 保存在本目录的 `banded_n*_cpu` / `banded_n*_cuda` 子目录。此前 `full_robot_n*_cpu_*` / `full_robot_n*_cuda_probe` 是旧的 dense-local Schur 追踪输出，保留用于审计，不混入本表。

GPU 仍不是完整设备驻留的物理核心：材料、接触历史、几何 Jacobian、Newton 线搜索和状态提交都在 CPU。下一阶段应先把八片材料与接触点按固定节点数 batch 化，再迁移接触历史和残差范数判断，最后才评估多环境 GPU 并行。
