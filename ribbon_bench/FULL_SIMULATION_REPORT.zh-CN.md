# 八片钢带绳驱蠕虫机器人：整机仿真报告

## 1. 目标与验收口径

本轮目标是把“钢片变形—端板运动—绳索受力—地面接触—摩擦历史”放进同一个可重复调用的物理计算核心。验收标准是：每一个时间步都能保存候选状态，Newton 残差有限且下降，接触历史只在接受步后更新，节点数和时间步改变时不会立刻崩溃。这里的“完成”表示数值闭环跑通，不表示已经完成实物标定，也不表示机器人已经证明能向前爬行。

## 2. 模型

每片钢带使用项目现有的 Discrete Elastic Ribbons/Sano 能量。钢带有 (n) 个节点和 (n-1) 个边角变量；首尾两节点及首尾边扭转角连接到刚性端板，8 片钢带的内部自由度彼此独立，再与两个端板的 12 个位姿自由度共同求解。每片每步的钢带自由度为 `4n-15`，全局 Newton 维度为 `8(4n-15)+12`：N9/N17/N33 分别为 180/436/948。钢带表面不是中心线：每条边在材料坐标的 (s=0,0.5,1) 处取矩形四角，宽度为 16 mm、厚度为 0.15 mm，并使用材料方向的解析 Jacobian。

端板质量和惯量来自 `longworm2.SLDASM.urdf`。前板合并 `front2_Link,w2-1_Link,w2-2_Link`，后板合并 `back2_Link,w2-3_Link,w2-4_Link`，锁定轮子随端板刚性运动。质量为 0.0831119 kg 和 0.5603828 kg，未人为对称化。

四根绳索是独立长度、无质量、仅受拉的直线弹簧，刚度取参数快照中的 20,000 N/m。每步由给定的 rest length 计算张力；本轮尚未加入真实双电机绕线、孔摩擦、绳索下垂和绳轮转动惯量。

地面接触采用世界坐标 (z=0) 的单边罚法向力

\[
N=k_n\max(-g,0),\qquad g=z-z_\mathrm{ground},
\]

切向使用带历史的弹性黏着和库仑滑动返回映射。滑动时切向力大小为 \(\mu N\)，离地时清空切向历史。该接触是柔顺正则化，会有微小穿透；滑动切线一般不对称，所以 Newton 用残差范数线搜索，不能套用全局保守势能的 Armijo 证明。

时间离散采用隐式 Euler。钢带内部残差可写成

\[
R_q=\nabla E_\mathrm{Sano}+M(q-q^n-\Delta t\,u^n)/\Delta t^2
       -J_\mathrm{contact}^{T}F_\mathrm{contact}.
\]

端板残差再加刚体平动/转动惯性、重力、绳索梯度和端板采样接触力。端板接触几何采用端板圆周两侧的离散点，是可复现实验包络，不是 CAD 连续碰撞检测。

## 3. 运行条件与结果

| 输出目录 | 节点/片 | \(\Delta t\) | 步数 | 收绳幅值 | 耗时 | 最大残差 | 最大穿透 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_robot_n9_one_independent` | 9 | 2 ms | 1 | 0.5 mm | 22.83 s | 9.99e-6 | 1.52 µm |
| `full_robot_n9_independent_probe` | 9 | 2 ms | 1 | 0 mm | 15.80 s | 1.66e-6 | 0.16 µm |
| `full_robot_n17_zero_independent` | 17 | 2 ms | 1 | 0 mm | 109.42 s | 4.93e-6 | 0.32 µm |

此前 `full_robot_n9_cmd5`、`full_robot_n17_cmd5_full` 和 `full_robot_n33_cmd5_full` 使用了共享钢带增量，属于首版降维调试结果，不能作为 8 片独立钢带的物理结果；其目录保留作追溯，但本报告的性能和论文数值以带有 `independent` 的新结果为准。N17 的 0.5 mm 收绳试算在 30 次 Newton 内未收敛，说明当前直接稠密耦合求解器还不能承担正式长时程驱动计算。

N9 的 20 ms 轨迹、PNG/SVG 论文图和 GIF 已生成。端板接触法向力约 6 N，绳张力随相位在约 0–100 N 之间变化。当前对称四绳命令主要激发绳张力和微小端板扰动，钢带弹性能仍近似为零，净平移是微米量级。因此本轮不能宣称已经得到有效爬行步态；这个负结果本身说明驱动时序、接触不对称性和真实绳路必须进入后续模型。
同一 20 ms 时间窗下，N9 的 1 ms 与 2 ms 端板 COM 最大差约 8.21 µm；这是时间步对照，不等于完整的时间收敛证明。

## 4. 已验证内容

- 接触本构的 stick/slip/liftoff、历史不可变、离地清空和 Jacobian 有独立自检，中心差分误差约 `1.32e-10`（按刚度缩放）。
- 宽厚表面几何 Jacobian 通过 N9/N17 有限差分，最大绝对误差约 `2.23e-10`，刚体旋转客观性误差约 `2e-17` m。
- CAD 刚体质量合并、COM 平移、惯量平行轴和绳索梯度/Hessian 通过有限差分；绳索总力矩平衡误差约 `1.2e-15 N·m`。
- 独立钢带版本的 N9 单步和 N17 单步均保存了 Newton 迭代、残差、接触穿透、法向力、绳张力和端板位姿；每片内部自由度互不共享。

## 5. 首轮问题与论文价值

1. CAD 预弯钢带有部分表面低于端板底部约 4 cm。若直接用端板底部放地面，第一步会出现非物理大冲击。因此初始化高度必须按实际宽厚表面最低点确定，并作为实验条件记录。
2. 罚接触穿透随节点数、时间步和接触刚度变化，本轮约 14–56 µm。论文应报告接触正则化误差和 (k_n\)、采样密度、时间步敏感性。
3. 对称四绳和均匀摩擦下，5 mm 周期收绳没有产生可观净爬行。可将“中心线/宽厚接触、对称/非对称驱动、是否存在净位移”作为可检验对照，而不是预设正结果。
4. 后端板质量约为前端板的 6.7 倍，不能把机器人简化成两端对称质量；该不对称性会改变法向载荷、摩擦阈值和步态相位。
5. 当前耦合 Newton 是 CPU 双精度稠密版本。独立 N9/N17 单步已分别耗时约 15.8/109.4 s；RTX 5090 上材料批量计算约有 8.8× 微基准加速，但整条路径不会按这个倍数缩短。下一步必须恢复局部带状结构并采用 Schur 消元，再测 GPU 端到端收益。

## 6. 可复现实验命令

在 `D:/doso/robot` 下：

```powershell
$env:PYTHONPATH='ribbon_bench/vendor/discrete-elastic-ribbon/src'
.\ribbon_bench\.venv\Scripts\python.exe ribbon_bench/full_robot.py --self-check
.\ribbon_bench\.venv\Scripts\python.exe ribbon_bench/full_robot.py --nodes 9 --steps 10 --duration .02 --dt .002 --command-mm 5 --output ribbon_bench/full_robot_n9_cmd5
.\ribbon_bench\.venv\Scripts\python.exe ribbon_bench/render_full_robot.py --input ribbon_bench/full_robot_n9_cmd5 --output ribbon_bench/full_robot_n9_cmd5/full_robot_paper.png
```

原始轨迹在 `trajectory.npz`，摘要在 `summary.json`，问题和决策在 `FULL_SIMULATION_LOG.zh-CN.md`。这些文件中的数值是数值实验记录；材料、摩擦、绳索和接触参数仍需实物测量后校准。
