# 整机带状 Schur CPU/CUDA 对照

测试主机：RTX 5090 32607 MiB、Xeon Platinum 8470Q、CUDA 12.8、PyTorch 2.8.0。每个算例为 8 片独立钢带、`dt=2 ms`、1 个时间步、0 mm 收绳；CPU 与 CUDA 使用同一份输入。CPU 将每片局部 Hessian 按节点重排为半带宽 10 的带状块，再解 12×12 Schur 补；Sano 批量梯度/Hessian 使用闭式 NumPy 公式；CUDA 当前批量解局部 dense FP64 块和 12×12 Schur 补。两条路径的几何、接触、装配和线搜索都在 CPU。

| 每片节点 | Newton 维度 | CPU 带状 (s) | CUDA Schur (s) | CPU/CUDA | 最大 `q` 差 (m) | 最大端板 COM 差 (m) | CUDA 求解器累计 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 180 | 2.3568 | 2.8445 | 0.829 | 6.04e-14 | 2.98e-16 | 0.4838 |
| 17 | 436 | 3.5547 | 3.8820 | 0.916 | 4.36e-14 | 2.41e-16 | 0.5021 |
| 33 | 948 | 7.1821 | 7.5086 | 0.957 | 7.49e-13 | 4.47e-15 | 0.5782 |

`CPU/CUDA` 大于 1 表示 CUDA 较快。三组结果的位移和端板质心差异均远小于 Newton 残差（约 `1e-6`），说明闭式 Sano、接触、绳索和惯性结果保持一致。优化后 N33 的 CUDA 时间约为 CPU 的 1.05 倍；单环境仍被 CPU 几何、接触和线搜索主导。重复运行时应报告中位数而不是单次 wall time。

每组的 `summary.json` 与 `trajectory.npz` 保存在本目录的 `fast_n*_cpu` / `fast_n*_cuda` 子目录。此前 `banded_n*_cpu` / `banded_n*_cuda` 是仅带状、仍使用 autograd 的追踪输出，保留用于审计，不混入本表。

GPU 仍不是完整设备驻留的物理核心：材料、接触历史、几何 Jacobian、Newton 线搜索和状态提交都在 CPU。下一阶段应先把八片材料与接触点按固定节点数 batch 化，再迁移接触历史和残差范数判断，最后才评估多环境 GPU 并行。
