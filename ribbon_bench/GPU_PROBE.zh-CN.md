# Sano 材料 GPU 实测与完整路径对照

日期：2026-10-01。材料微基准与一对完整路径对照均已完成。完整路径的逐帧物理对照通过，但本次 CUDA 材料版本用时 **36.325 s**，CPU 版本 **31.462 s**，约慢 **15.46%**；此次接入没有取得整段加速。

本实验回答两个不同问题：Sano 材料能量、梯度和 Hessian 能否在 GPU 上正确计算，以及把这部分接入现有求解器后，整段准静态路径是否更快。**材料微基准的加速不能直接解释为整个机器人仿真加速。** 当前实现没有训练或调用神经网络。

## 实测设备与隔离范围

| 实验 | 环境 | 本轮范围 |
| --- | --- | --- |
| 本机 DirectML | Windows 11；AMD Radeon 890M；Python 3.12.14；PyTorch 2.4.1；torch-directml 0.2.5.dev240914；NumPy 1.26.4 | 独立环境中的 FP32 / FP64 材料探针及 FP32 输出装配复核 |
| CUDA eager / compiled | Linux；NVIDIA GeForce RTX 5090；Python 3.12.3；PyTorch 2.8.0+cu128；NumPy 2.3.2 | FP32 / FP64 材料微基准；FP64 材料接入的完整路径对照 |

保留原 CPU 参考环境；GPU 依赖与实验在隔离环境中准备，没有更新本机显卡驱动。CUDA 主机为 Xeon Platinum 8470Q，GPU 显存总量记录为 32607 MiB，驱动 595.58.03，详见 [硬件与依赖记录](gpu_probe_20261001/remote_environment.json)。微基准显式把 PyTorch CPU 计算线程与 interop 线程设为 1；两次完整路径均设置 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`。这不是主机全部 208 个逻辑核的吞吐测试。不能把两台机器的绝对耗时差解释为 GPU 的独立收益，应比较每组实验同机的 CPU 与 GPU 路径。

## 输入与计算边界

[export_gpu_inputs.py](export_gpu_inputs.py)从已通过 CPU64 验证的八片、33 节点绳驱路径中导出初态、峰值、偏航卸载、压缩卸载和回到初态五个关键帧。每片每帧有 31 个材料求值点，共 **1240 行真实应变**；保留逐行材料参数、自然应变扣除后的归一化应变及 CPU64 能量、梯度和 Hessian。

输入来源与 SHA-256 见 [inputs.json](gpu_probe_20261001/inputs.json)，数组见 [inputs.npz](gpu_probe_20261001/inputs.npz)。三种材料微基准读取同一份数组，输入 SHA-256 为 `20863574b7ce7b23dd8f1f8c8d40d92048415bba70ffc98028bb1b93889ee332`。各轮保存所执行脚本的 SHA，不能把后来新增 CUDA 支持的脚本哈希误写为较早 DirectML 实验的哈希。

[gpu_probe.py](gpu_probe.py)以 Torch 张量运算实现 [fast_sano.py](fast_sano.py)同一闭式材料公式，每行输出 1 个能量、4 个一阶导数和 16 个 Hessian 分量。CUDA compiled 仅通过 `torch.compile(fullgraph=True, dynamic=False)`融合材料计算，不编译整个求解器。

这一级测试**不计算 GPU 几何导数、全局装配、Newton、接触或时间积分**。数组中的材料能量及导数采用上游归一化定义，不能将其数值直接标成 J、N 或 N·m；物理钢带能量需乘保存的逐行能量尺度后求和。

## 微基准方法与结果入口

每项先预热 3 次，再计时 **9 次并取中位数**。排除设备初始化、编译和常量系数首次上传。每个样本都把完整能量、梯度和 Hessian 读回 CPU，并转换为 NumPy float64 数组后结束计时，因此包含等待 GPU 完成，绝非仅测命令提交。

- **端到端材料调用：** 包含 CPU64 应变转换到目标精度、上传、GPU 计算以及完整结果读回。
- **输入驻留设备：** 省去每次应变上传，但仍包含完整结果读回；它不是纯 GPU 内核时间。
- CPU 对照同时保留 NumPy FP64 闭式实现与同精度 Torch CPU 实现。FP32 对 CPU64 的速度比同时改变了精度，不能当成同精度比较。

材料行数为 31、248、7936、63488。前两项分别对应单片和八片在一帧中的材料求值量；后两项重复平铺真实峰值应变，分别为 248 行的 32 倍与 256 倍。**这种 tile 只测试材料吞吐，不等于 32 / 256 个独立机器人环境；没有独立状态、约束、接触、求解器或控制循环。**

下表为九次测量的中位数，单位均为 **ms**。CUDA 表选用 FP64，与 NumPy64 参考保持同精度；CUDA FP32 的全部原始数据仍保留在摘要中。

| 材料后端 / 精度 | 行数 | NumPy CPU64 | Torch CPU 同精度 | GPU 端到端 | GPU 输入驻留，含读回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DirectML FP32 | 31 | 0.0433 | 0.1282 | 0.9571 | 0.7946 |
| DirectML FP32 | 248 | 0.0567 | 0.1333 | 0.8844 | 0.7888 |
| DirectML FP32 | 7936 | 0.6765 | 0.6343 | 1.2821 | 1.3527 |
| DirectML FP32 | 63488 | 8.9817 | 4.7590 | 6.5791 | 6.6717 |
| CUDA eager FP64 | 31 | 0.0519 | 0.1780 | 0.5126 | 0.4890 |
| CUDA eager FP64 | 248 | 0.0948 | 0.1945 | 0.5260 | 0.5080 |
| CUDA eager FP64 | 7936 | 1.8862 | 0.7760 | 0.8293 | 0.7202 |
| CUDA eager FP64 | 63488 | 24.6519 | 13.9324 | 2.8697 | 2.5577 |
| CUDA compiled FP64 | 31 | 0.0492 | 0.1693 | 0.1920 | 0.1725 |
| CUDA compiled FP64 | 248 | 0.0939 | 0.1948 | 0.2093 | 0.1879 |
| CUDA compiled FP64 | 7936 | 1.9002 | 0.8446 | 0.9282 | 0.4882 |
| CUDA compiled FP64 | 63488 | 25.1243 | 15.7489 | 2.8526 | 2.4892 |

来源：[DirectML 九次采样](gpu_probe_20261001/directml/summary.json)、[CUDA eager 九次采样](gpu_probe_20261001/cuda_eager/summary.json)、[CUDA compiled 九次采样](gpu_probe_20261001/cuda_compiled/summary.json)。表中不同轮次各有自己的 CPU 对照，不混用基准分母。

当前单片的 31 行求值中，即使编译后的 CUDA FP64 仍慢于 NumPy CPU64；63488 行批量下，CUDA compiled 的材料端到端耗时约为其同轮 NumPy64 对照的 **1/8.81**。后者只支持“大批材料计算具有吞吐收益”这一结论。7936 行下 CUDA 端到端仍慢于同精度 Torch CPU；DirectML 最大批量也仍慢于其 Torch CPU FP32 对照，不能只挑选较慢的 CPU 实现宣称普遍 GPU 加速。个别驻留输入样本比端到端样本稍慢，说明微秒至毫秒级测量存在波动；不把单组中位数差解释为严格成本分解。

## 精度验收与峰值装配检查

材料误差按分量的应变尺度和自然刚度尺度归一化，避免把不同物理量混在同一个绝对误差中。微基准的 FP32 尺度化误差限为 $2\times10^{-5}$，FP64 为 $3\times10^{-10}$；FP64 另检查 $1+2^{-40}$ 的增量是否保留。测试捕获后端警告，计时期间将警告视为失败，以免把有 CPU fallback 提示的调用当成合格 GPU 性能。

DirectML FP32 材料求值通过该级检查；本机完整 FP64 材料调用报错。后者只说明此次软硬件组合与材料算子路径未成功，不外推为所有 DirectML 算子或所有 AMD GPU 都不支持 FP64。

[check_gpu_outputs.py](check_gpu_outputs.py)进一步在原 CPU64 峰值几何上，把**实际记录的 GPU FP32 材料输出**接回 CPU64 的几何链式求导、绳载荷与全局装配。它同时执行 CPU64 记录回放作为控制，核对输入、材料参数、自然应变及源码哈希，没有重新运行 Newton。

该检查记录在 [assembled_gpu_check.json](gpu_probe_20261001/assembled_gpu_check.json)：GPU 材料输出接入后的自由节点力残差为 **$1.5171\times10^{-6}$ N**，超过原有 **$1.0\times10^{-6}$ N** 门槛，故未通过既有平衡判据。把输出存成 float64 不会恢复 FP32 运算已丢失的精度。这一结论属于峰值处的单次装配检查，不能直接宣称 FP32 重新求解必然不收敛，但足以说明材料误差通过不等于物理残差通过；本轮没有放宽原阈值来接纳它。

CUDA 两种精度均通过材料级检查，FP64 保留了精度探针增量。进一步使用 CUDA compiled 输出复核峰值装配：[FP32 记录](gpu_probe_20261001/assembled_cuda_float32.json)同样得到 $1.5171\times10^{-6}$ N，自由节点力残差未通过；[FP64 记录](gpu_probe_20261001/assembled_cuda_float64.json)为 $5.9765\times10^{-9}$ N，通过既有全部平衡门槛。因此完整路径采用 compiled FP64，并在材料级、峰值装配级之后，再由完整路径逐帧对照验收。

发布目录也已通过同一 [FP64 装配复核](gpu_probe_20261001/packaged_cuda_float64_check.json)。报告记录了发布包中CAD与默认参数路径适配的精确源码哈希配对；参数、CAD及物理执行代码仍按原记录核验。

上游独立 `forward()` 用于线搜索能量，材料接口返回的能量在装配接口中并未直接作为系统能量采用。因此报告中的“实际 GPU 钢带能量”由记录的 GPU 能量逐行乘物理尺度独立求和，不将 CPU `forward()` 的结果标成 GPU 能量。

## 完整路径对照：材料在 GPU，求解器仍在 CPU

[cuda_solver_probe.py](cuda_solver_probe.py)采用局部材料适配器，把真实 Newton 迭代的应变上传至 CUDA FP64，计算材料能量及导数后完整读回 CPU。参数不变时复用 GPU 常量；参数变化时重新上传。既有自然应变、几何导数、稠密装配、带状块求解、Schur 消元、Newton 正则化和线搜索继续在 CPU 执行；独立 `forward()` 能量也继续在 CPU。

对照在同一机器各运行 **一次** CPU 与 CUDA compiled 材料版本，均为八片、33 节点、25 状态完整绳驱路径。四根理想绳的收放长度由 40 mm / 30° 目标几何生成；实际后板姿态由平衡求解，不能称为规定后板达到 30°。

完整 `results.json`、实际 CUDA 调用计数及双向结果对照均已通过。比较沿用原自由节点力 / 力矩和板广义力 / 力矩残差门槛，同时比较节点、材料方向、反力、张力与能量；没有只看终帧或只看材料函数误差。

| 项目 | CPU 路径 | CUDA 材料 + CPU 求解器 |
| --- | --- | --- |
| 完整 25 状态结果 | [full_cpu/results.json](gpu_probe_20261001/full_cpu/results.json) | [full_cuda/results.json](gpu_probe_20261001/full_cuda/results.json) |
| 路径耗时，单次 | 31.462 s | 36.325 s |
| 路径内 CUDA 材料调用 | 0 | 6512 次，每次 31 行，共 201872 行 |
| 路径内 CUDA 适配器累计时间 | 不适用 | 4.185 s，含上传、材料计算和读回 |
| 路径外材料预热 | 不适用 | 3.189 s，8 次调用、248 行 |
| 路径内自适应二分次数 | 0 | 0 |

[逐帧比较记录](gpu_probe_20261001/compare_full.json)为 `passed`：节点分量最大差 $6.1943\times10^{-11}$ m，材料宽度方向分量最大差 $2.3998\times10^{-10}$，后端反力分量最大差 $3.3554\times10^{-7}$ N，绳张力最大差 $2.9639\times10^{-8}$ N。两份结果均满足原残差要求。

GPU 路径计时前已完成 8 次常量上传，路径内没有重复上传常量；路径内应变上传累计 6459904 字节、材料结果读回 33914496 字节。设备设置另记录 1.293 s，均不能误算为路径内纯内核耗时。详见 [CPU 运行元数据](gpu_probe_20261001/full_cpu/probe_metadata.json)与 [CUDA 运行元数据](gpu_probe_20261001/full_cuda/probe_metadata.json)。

全路径计时使用 `actuate` 的路径计时器，排除导入、初始化、CUDA 预热和绘图；材料输入输出传输发生在求解过程中，计入路径耗时。单次全路径对照与微基准的九次取中位数是两种统计口径，不能混称为重复验证的总体加速率。

此次 CPU / CUDA 耗时比为 **0.866**，CUDA 路径约慢 **15.46%**。浮点运算顺序变化也改变了部分 Newton 迭代步数，因此不能将这 15.46% 全部归因于传输或 GPU 启动开销，也不能将 4.185 s 适配器累计时间直接当作相对 CPU 的净增量。本轮结果支持继续以 CPU 精确求解器作为当前小系统基准；未来需要更大批量且设备驻留的几何、装配与求解流程，才能重新评估整体收益。

## 纯 RTX 5090 的时间估算与加速路线

这里的“纯 GPU”定义为：钢带几何、Sano 材料、绳索、接触历史、全局装配、Newton 迭代和线性求解都在 CUDA 上，时间步之间不把状态往返 CPU；只在保存轨迹时读回结果。当前已完成的是 CUDA Schur 混合版本：8 个局部块和 12×12 Schur 在线性代数中使用 RTX 5090，几何、Sano 材料、接触与线搜索仍在 CPU。因此下面同时报告实测混合时间和纯 GPU 的后续目标，不能把混合时间写成纯 GPU 结果。

首版共享钢带增量的整机 CPU 数字（N9/N17/N33 为 4.70/7.40/28.12 s）已作废，不能代表 8 片独立钢带。当前独立带状 Schur 版本在同一 RTX 5090 主机上的 CPU/CUDA 单步分别为：N9 **3.45/3.82 s**，N17 **7.36/7.89 s**，N33 **31.55/31.11 s**。这组 CUDA 只迁移局部线性代数，GPU 相对 CPU 的端到端收益为 N9 0.90×、N17 0.93×、N33 1.01×；它证明了接口与数值一致性，但仍不是纯 GPU 物理核心。旧的 N33 十步结果保留作 Schur 长时程追踪，不与当前带状单步计时混写。

每一步的真实成本应按下面的分解测量：

\[
T_\mathrm{run}=N_\mathrm{step}N_\mathrm{Newton}\left(T_\mathrm{geom}+T_\mathrm{material}+T_\mathrm{contact}+T_\mathrm{assemble}+T_\mathrm{solve}+T_\mathrm{line\ search}\right)+T_\mathrm{I/O}.
\]

当前每一步通常需要 10–20 次 Newton/线搜索评估；N33 的正确模型每次是 948 维系统。直接把矩阵搬到 GPU 仍会重复构造并产生同步，实测 N33 单步只得到约 7% 混合收益。纯 GPU 版本需要将材料、几何 Jacobian、接触历史、装配和线搜索状态一起设备驻留；在实现前只能把 N33 约 **数十秒/10 步**、N65 约 **1–3 分钟/10 步**、N129 约 **数分钟/10 步**作为工程目标，不能写成实测论文结果。

同一 RTX 5090 主机上的 N33 十步完整对照已经完成：CPU Schur **418.29 s**，CUDA Schur **393.86 s**，加速比 **1.062×**；节点轨迹最大差 `2.77e-11 m`，端板 COM 最大差 `1.51e-13 m`。CUDA 线性求解累计只有 1.40 s，端到端瓶颈仍在 CPU 几何、材料、接触和 Newton 评估。

提升顺序如下：

1. 已完成每片局部 Hessian 的节点重排和半带宽 10 带状存储；CPU 自检与 dense Schur 增量相对误差小于 `1e-8`。
2. 把八条钢带、表面接触点和材料行按 batch 向量化，固定节点数和形状，使用 CUDA compiled/fused kernel。
3. 将接触历史、stick/slip mask、绳索导数和 Newton 线搜索状态留在设备端；每次 Newton 只做设备内残差范数和收敛判断，避免同步。
4. N65 以上再做 16–64 个机器人环境的 batch；多环境才足以摊薄 RTX 5090 的启动成本。单个 N33 环境即使纯 GPU 仍可能不如 CPU 双精度。
5. 先以 FP64 作为可信基准。已有 CUDA FP32 峰值装配残差为 `1.517e-6 N`，超过项目 `1e-6 N` 门槛；可以用 FP32 几何加 FP64 累加/迭代修正做实验，但不能直接替代 FP64 物理结果。

## 复现入口与来源

以下从 `ribbon_bench` 目录执行，使用已准备好的对应虚拟环境，输出选择新路径；命令不包含远程地址或账号。

```bash
# 在对应 DirectML / CUDA 环境分别运行；输入保持同一文件
python gpu_probe.py --backend directml --inputs gpu_probe_20261001/inputs.npz --output gpu_reproduce/directml
python gpu_probe.py --backend cuda --inputs gpu_probe_20261001/inputs.npz --output gpu_reproduce/cuda_eager
python gpu_probe.py --backend cuda --compile --inputs gpu_probe_20261001/inputs.npz --output gpu_reproduce/cuda_compiled

# 在同一 CUDA 主机、同一 CPU 线程设置下比较完整路径
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python cuda_solver_probe.py --backend cpu --nodes 33 --steps 6 --compression-mm 40 --yaw-deg 30 --output gpu_reproduce/full_cpu
python cuda_solver_probe.py --backend cuda --compile --nodes 33 --steps 6 --compression-mm 40 --yaw-deg 30 --output gpu_reproduce/full_cuda
python cuda_solver_probe.py --compare gpu_reproduce/full_cpu/results.json gpu_reproduce/full_cuda/results.json --output gpu_reproduce/compare_full.json
```

后端适用范围参考 [微软 DirectML 安装说明](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)与 [torch-directml 包元数据](https://pypi.org/project/torch-directml/)；本实验实际依赖版本以各份摘要为准。材料来源固定为 [Discrete Elastic Ribbons 提交 c9d3411](https://github.com/StructuresComp/discrete-elastic-ribbon/tree/c9d341164e2927fc24b2c43dff97fcfb492cf700)，具体闭式实现和验证入口见 [CPU 物理核心说明](PHYSICS_CORE.zh-CN.md)。

本轮不测自由爬行、惯性振动、摩擦、真实绳孔接触或神经网络代理。无论大批材料计算取得何种速度比，都仍需把几何、装配、线性求解和独立环境状态一起纳入设计与计时，才能评估未来多环境 GPU 仿真。
