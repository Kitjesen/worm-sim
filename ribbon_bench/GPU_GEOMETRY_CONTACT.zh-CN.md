# 钢带几何、材料装配与接触的 CUDA 迁移记录

日期：2026-10-01。目标是把 8 片钢带在每次 Newton 评估中的高频计算从逐片 NumPy 循环改成一个 FP64 Torch CUDA batch，同时保持 CPU 参考路径和结果可对照。

## 迁移内容

每片钢带的表面不是中心线，而是每条边的 12 个材料点。CUDA kernel 对 8 片钢带同时完成：

1. 根据节点位置和边扭转角，做最小旋转传输、材料方向投影和矩形表面点计算；
2. 计算每个表面点对两个端点平移和边扭转角的局部 Jacobian；
3. 计算地面单边罚接触、切向黏着、库仑滑动、离地以及接触历史更新；
4. 用局部 Jacobian 把接触力和接触切线直接收缩到钢带自由度与端板 12 个自由度；
5. 只把每片的小型残差/Hessian 块回传给 CPU Schur 接口。接受时间步时才回传完整接触历史。

Sano 材料路径使用 `cuda_sano.py`：应变值、vendor 兼容的应变梯度/Hessian、闭式材料导数、链式法则 `einsum` 和全局 `scatter_add` 在 GPU。端板的采样点、刚体 Jacobian 和接触本构也在同一 CUDA batch 中计算；CPU 只接收紧凑的局部块，供现有 Schur 接口求解并在接受步后提交历史。

## 数值验收

`gpu_geometry_contact.py` 在 CPU Torch 上与原 NumPy 实现逐项比较：

| 项目 | 最大误差 |
| --- | ---: |
| 表面点 | `0`（原始状态）；扰动状态不超过 `6.94e-18 m` |
| 局部几何 Jacobian | `2.22e-16` |
| 接触力 | `0` |
| 接触 Jacobian | `0` |

`cuda_sano.py` 的 N9 单片链式装配自检通过，力最大绝对误差 `2.17e-11 N`，Hessian 最大绝对误差 `2.79e-9`。随机扰动的 N9/N17/N33 应变、梯度和 Hessian 最大绝对误差分别为 `4.44e-16`、`2.84e-13`、`4.58e-10`。项目原有 `full_robot.py --self-check` 仍通过，残差 `3.636e-7 N`，最大穿透 `1.402e-7 m`。

## RTX 5090 实测

配置为 8 片钢带、`dt=2 ms`、单时间步、零驱动变化；CPU 与 CUDA 使用同一远程主机和同一输入。以下为一次冷启动运行，适合看端到端路径，不等同于 kernel 微基准：

| 节点数 | CPU 整机 | CUDA 整机 | CUDA/CPU |
| ---: | ---: | ---: | ---: |
| N9 | `1.27 s` | `2.03 s` | `1.60×` |
| N17 | `2.27 s` | `1.99 s` | `0.88×` |
| N33 | `4.92 s` | `2.11 s` | `0.43×` |

N33 的整机收益约 `2.3×`；单独测量 8 片几何+接触 batch，N9/N17/N33 分别约为 `2.23×/2.57×/3.15×`。CPU 与 CUDA 的 Newton 残差、接触穿透、绳索长度和端板轨迹保持在数值误差范围内。

## 运行入口

```powershell
$env:PYTHONPATH='ribbon_bench/vendor/discrete-elastic-ribbon/src'
& 'D:\doso\robot\ribbon_bench\.venv\Scripts\python.exe' ribbon_bench/cuda_sano.py
& 'D:\doso\robot\ribbon_bench\.venv\Scripts\python.exe' ribbon_bench/gpu_geometry_contact.py
& 'D:\doso\robot\ribbon_bench\.venv\Scripts\python.exe' ribbon_bench/full_robot.py --self-check
```

远程 RTX 5090 路径：

```bash
export PYTHONPATH=vendor/discrete-elastic-ribbon/src
python full_robot.py --nodes 33 --steps 1 --duration .002 --command-mm 0 --solve-backend cuda --output /tmp/gpu_n33
```

带较大收绳时，N17 已通过 38 次 Newton 迭代。加入 `--max-command-step-mm 0.1` 后，0.5 mm 收绳被拆成 6 个隐式子步，N33 也能稳定通过。迁移端板几何后的 RTX 5090 对照耗时为 CPU/CUDA `9.39/7.99 s`，最后子步 Newton 迭代 `7/7`，最大残差 `4.219e-6/4.218e-6`，最大穿透 `4.715e-6/4.715e-6 m`。轨迹最大差异为节点 `1.64e-14`、端板 COM `1.25e-16`、端板旋转矩阵 `3.11e-15`。结果保存在 `gpu_full_robot_20261001/adaptive_n33_cpu_v2` 和 `adaptive_n33_cuda_v3`；这仍是数值稳定性验证，不是爬行性能结论。

用于汇报的 10 ms、6 帧 CUDA 收绳动画保存在 `gpu_full_robot_20261001/adaptive_n33_cuda_motion/robot_contraction.gif`，同时提供 PNG/SVG 静态图和指标图。
