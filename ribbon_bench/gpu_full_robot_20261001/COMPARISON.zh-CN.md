# 整机带状 Schur CPU/CUDA 对照

测试主机：RTX 5090 32607 MiB、Xeon Platinum 8470Q、CUDA 12.8、PyTorch 2.8.0。每个算例为 8 片独立钢带、`dt=2 ms`、1 个时间步、0 mm 收绳；CPU 与 CUDA 使用同一份输入。CPU 将每片局部 Hessian 按节点重排为半带宽 10 的带状块，再解 12×12 Schur 补；Sano 批量梯度/Hessian 使用闭式 NumPy 公式；CUDA 复用设备缓冲区，把 13 个局部右端项合并后用 `solve_ex` 解 FP64 块和 12×12 Schur 补。两条路径的几何、接触、装配和线搜索都在 CPU。

| 每片节点 | Newton 维度 | CPU 带状 (s) | CUDA Schur (s) | CPU/CUDA | 最大 `q` 差 (m) | 最大端板 COM 差 (m) | CUDA 求解器累计 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 180 | 1.2296 | 1.6442 | 0.748 | 6.04e-14 | 2.98e-16 | 0.4415* |
| 17 | 436 | 1.9789 | 2.4108 | 0.821 | 6.39e-14 | 5.52e-16 | 0.4604* |
| 33 | 948 | 3.5889 | 4.0131 | 0.894 | 1.70e-13 | 2.32e-15 | 0.4872* |

`CPU/CUDA` 大于 1 表示 CUDA 较快。三组结果的位移和端板质心差异均远小于 Newton 残差（约 `1e-6`），说明闭式 Sano、接触、绳索和惯性结果保持一致。优化后 N33 的 CUDA 时间约为 CPU 的 1.12 倍；单环境仍被 CPU 钢带链式装配、几何、接触和线搜索主导。表中为独立进程冷启动 wall time，`*` 包含首次 CUDA kernel/context 开销；同进程 warm-up 后 N33 局部 solve 约 16.6 ms。重复运行时应报告中位数并单独记录 warm-up。

每组的 `summary.json` 与 `trajectory.npz` 保存在本目录的 `optimized_n*_cpu` / `optimized_n*_cuda` 子目录。此前 `fast_n*` 是删除无用能量和 CUDA 缓冲区复用前的对照；`banded_n*` 是仅带状、仍使用 autograd 的追踪输出，均保留用于审计，不混入本表。

GPU 仍不是完整设备驻留的物理核心：材料、接触历史、几何 Jacobian、Newton 线搜索和状态提交都在 CPU。下一阶段应先把八片材料与接触点按固定节点数 batch 化，再迁移接触历史和残差范数判断，最后才评估多环境 GPU 并行。
