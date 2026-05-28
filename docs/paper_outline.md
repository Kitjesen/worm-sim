# 论文大纲

## 暂定标题

**Multi-Modal Gait Optimization for a Metameric Pipe-Crawling Robot via Deep Reinforcement Learning in MuJoCo**

管道蠕虫机器人多模态步态的深度强化学习优化

---

## Abstract (200 words)

管道检测机器人需要在狭窄、弯曲管道中可靠前进。蠕虫式运动具有天然管道适应性，
但传统的开环步态设计难以在不同管道条件下达到最优。本文提出一种基于深度强化学习
的多模态步态优化方法，用于刚体链式管道蠕虫机器人。我们建立了与真实硬件 1:1 尺寸
映射的 MuJoCo 仿真模型，包含 5 体节、8 驱动器、被动轮各向异性摩擦和弹簧钢片弹
性连接。在此平台上，使用 PPO 算法自动发现蠕动、蛇形和组合三种步态模式的最优参
数。实验表明：(1) RL 优化的步态在前进速度上超越手调基线 X%；(2) 不同管径和弯曲
半径下，最优步态模式自动切换；(3) 刚体链方案在保持管道通过性的同时，比软体仿真
快 N 倍。本工作为管道蠕虫机器人的步态自动设计和 sim-to-real 部署提供了一条实用路线。

---

## 1. Introduction

### 1.1 背景与动机
- 管道检测的工业需求（石油、天然气、城市管网）
- 蠕虫式运动的优势：体节收缩产生推进力，天然适应管道约束
- 现有方案的局限：软体气动复杂、开环步态难以适应多变管道

### 1.2 相关工作概述
- 管道蠕虫机器人设计（软体 vs 刚体）
- 步态优化方法（手调 vs 优化算法 vs RL）
- 本文定位：RL + 刚体链 + 多模态步态

### 1.3 贡献
1. 提出刚体链+被动轮+弹簧钢片的管道蠕虫仿真模型，与真实硬件 1:1 尺寸映射
2. 设计 CPG+残差 的 RL 动作空间，实现蠕动/蛇形/组合三种步态的自动优化
3. 系统研究管径、弯曲半径对最优步态模式的影响规律
4. 开源仿真平台和训练代码

---

## 2. Related Work

### 2.1 管道蠕虫机器人设计
- **软体方案**：CMMWorm 系列 [Horchler 2015, 2016]，气动驱动 + 弹性体
- **刚体方案**：metameric earthworm [Zhan 2019, Fang 2023, 2025]，模块化关节链
- **混合方案**：Snake-Worm bi-modal robot [2022]，同时具备蛇形和蠕动能力
- **我们的定位**：刚体链 + 被动轮，兼具软体的管道适应性和刚体的仿真效率

### 2.2 被动轮与各向异性摩擦
- 被动轮提供方向性摩擦的机制 [Horchler 2015]
- 摩擦各向异性对蠕动推进的关键作用 [Tirado 2024]
- 软体方案中的 setae/bristle 摩擦 [earthworm skin robot 2024]

### 2.3 蠕虫/蛇形机器人的步态控制
- CPG（中枢模式发生器）控制 [Zhou 2023]
- 开环时序控制器 [CMMWorm Time_Based_Worm.py]
- 多模态步态统一框架 [WSIM robot 2023]

### 2.4 强化学习用于蛇形/蠕虫运动
- DRL 用于蛇形运动步态发现 [2020, IFAC]
- PPO 学习节能蛇形步态 [2021, IEEE T-Mech]
- CPG + RL 混合控制 [2020 ICRA, 2022 IEEE T-RO]
- 接触感知 CPG-RL [2021]
- **Gap**：现有 RL 工作集中在蛇形机器人，少有用于管道蠕虫 + 多模态步态

### 2.5 仿真平台与 Sim-to-Real
- MuJoCo 在机器人学习中的应用
- MJX GPU 加速并行仿真
- 管道蠕虫仿真的特殊挑战（接触丰富、多体耦合）

---

## 3. Robot Model and Simulation

### 3.1 硬件原型描述
- 5 节段 metameric 结构（基于 Fang/Zhan 设计）
- 每节段：刚性壳体 + 左右被动轮
- 节间连接：slide 关节（轴向伸缩）+ yaw 关节（偏航转向）
- 弹簧钢片提供节间弹性回复力

### 3.2 MuJoCo 仿真建模
- 真实 SolidWorks STL mesh 导入（12 个零件文件）
- 碰撞几何：capsule（体节）+ cylinder（轮子）+ plane（地面/管壁）
- 尺寸参数：表格列出所有关键参数（来自 v6.urdf）
  - 18 bodies, 26 DOF, 8 actuators
  - 总质量 ~2.7 kg，体节间距 161.5mm
  - 轮径 25mm，被动自由旋转

### 3.3 被动轮各向异性摩擦
- 原理：轮子自由旋转 → 纵向低阻力；不能侧滑 → 横向高阻力
- MuJoCo 实现：wheel friction=1.5, body friction=0.3, condim=4
- 与软体方案中 setae/bristle 摩擦的对比

### 3.4 弹簧钢片弹性建模
- 物理机制：钢片连接相邻端板，压缩时弓起储能
- 仿真实现：slide joint stiffness 参数等效
- 可视化：per-frame BOX geom 注入，抛物线弓形 profile

### 3.5 管道环境
- 直管：圆柱形 static collider，可变管径
- 弯管：90° 弯曲，可变弯曲半径
- 管壁摩擦参数化

---

## 4. Multi-Modal Gait and RL Formulation

### 4.1 三种步态模式
- **蠕动 (Peristaltic)**：slide 关节收缩波，轮子锚定产生推进
- **蛇形 (Serpentine)**：yaw 关节正弦波，各向异性摩擦产生净前进力
- **组合 (Combined)**：slide + yaw 同时工作

### 4.2 MDP 定义

**State space (25 维)**：
- 关节位置（8）、关节速度（8）
- 投影重力（3）、机体角速度（3）、机体线速度（3）

**Action space**：
- 方案 A：直接关节控制（8 维）
- 方案 B：CPG 参数 + 残差（10 维，推荐）

**Reward**：
- r = w1·v_forward - w2·|v_lateral| - w3·|heading| - w4·energy - w5·action_rate + w6·alive
- 管道模式：v_forward 替换为沿管道中心线的前进速度

**Termination**：翻车（z < 阈值 or 姿态异常）

### 4.3 CPG 动作空间设计
- CPG 生成基础周期波形（6 参数：amp, freq, phase_offset × 2 modes）
- RL 残差在 CPG 基础上微调（4 维 yaw 修正）
- 优势：保证步态周期性，降低探索难度，sim-to-real 友好

### 4.4 训练策略
- 算法：PPO（Proximal Policy Optimization）
- 网络：MLP [128, 128]，tanh 激活
- 并行：MJX + Brax，4096 并行环境
- Curriculum：平地 → 直管 → 弯管
- Domain randomization：摩擦、质量、关节阻尼

---

## 5. Experiments

### 5.1 实验设置
- 训练平台：NVIDIA RTX 5090, MuJoCo MJX
- 训练量：50M timesteps per experiment
- 评估：5 seeds, mean ± std
- 基线对比：
  - Hand-tuned snake (A=0.30, f=0.5Hz)
  - Hand-tuned worm (gait=[0,0,0,1,1], step=0.6s)
  - Hand-tuned combined
  - Random policy

### 5.2 Exp 1：平地步态优化
- RL vs 手调基线：前进速度、能耗、航向稳定性
- 消融实验：reward 各项权重的影响
- 学习到的步态可视化分析

### 5.3 Exp 2：直管推进
- 不同管径（80mm, 100mm, 120mm, 150mm）下的推进速度
- RL 自动发现的步态模式 vs 管径的关系
- 蠕动模式在管道中的推进效率 vs 平地对比

### 5.4 Exp 3：弯管通过
- 不同弯曲半径（0.3m, 0.45m, 0.6m, 0.9m）
- RL 策略在弯道处的步态调整行为分析
- 成功率 vs 弯曲半径
- 与 Riddle 2025 的 critical ROC 结论对比

### 5.5 Exp 4：多模态步态分析
- RL 在不同条件下自动选择的步态模式
- 蠕动 vs 蛇形 vs 组合 的适用场景
- 步态切换时机的可解释性分析

### 5.6 Exp 5：消融实验与敏感性分析
- 被动轮摩擦系数对性能的影响
- 弹簧钢片刚度的作用
- CPG vs 直接控制 vs 残差策略对比
- 观测空间各项的贡献

---

## 6. Results and Discussion

### 6.1 主要发现
- RL 优化后的步态速度/效率提升幅度
- 管径-步态模式映射规律（核心发现）
- 弯管通过策略的可解释分析

### 6.2 与软体方案的对比
- 刚体链 vs CMMWorm（Riddle 2025）在管道任务上的表现
- 仿真效率对比（刚体链的速度优势）
- 物理真实性讨论

### 6.3 局限性
- 当前未做 sim-to-real 真机验证
- 弹簧钢片弹力在仿真中的近似
- MuJoCo 接触模型的固有限制
- 只测试了 5 体节配置

---

## 7. Conclusion

### 7.1 总结
- 提出了刚体链管道蠕虫的 RL 步态优化方案
- RL 自动发现了超越手调的多模态步态
- 揭示了管道条件与最优步态模式的映射规律

### 7.2 Future Work
- Sim-to-real：在 Fang/Zhan 硬件原型上部署
- 6+ 体节的可扩展性
- 传感器反馈闭环（管壁力、IMU）
- 从单一管道到管道网络的路径规划

---

## References (核心引用列表)

### 管道蠕虫机器人设计
[1] Horchler, A.D., Kandhari, A., Daltorio, K.A., et al. "Peristaltic locomotion of a modular mesh-based worm robot: Precision, compliance, and friction." Soft Robotics, 2(3), 2015.
[2] Horchler, A.D., et al. "Worm-like robotic locomotion with a compliant modular mesh." Springer, 2016.
[3] "Design and actuation of a fabric-based worm-like robot (FabricWorm)." MDPI Machines, 2019.

### Metameric earthworm 动力学（方/展团队）
[4] Zhan, X., Fang, H., Xu, J., Wang, K.W. "Planar locomotion of earthworm-like metameric robots." IJRR, 2019.
[5] Fang, H., et al. "Spatial locomotion of a metameric earthworm-like robot: generation and analysis of gaits." Multibody System Dynamics, 2023.
[6] Fang, H., Zhou, Q., et al. "Dynamic modeling and analysis for planar peristaltic locomotion of a metameric earthworm-like robot." Proc. IMechE Part K, 2025.

### 蛇形/蠕虫 多模态步态
[7] "A worm-snake-inspired metameric robot for multi-modal locomotion: Design, modeling, and unified gait control." Mechanism and Machine Theory, 2023.
[8] "Snake-Worm: A bi-modal locomotion robot." J. Bionic Engineering, 2022.
[9] Zhou, Q., et al. "A CPG-based versatile control framework for metameric earthworm-like robotic locomotion." Advanced Science, 2023.

### RL 用于蛇形/蠕虫运动
[10] "Deep reinforcement learning for snake robot locomotion." IFAC-PapersOnLine, 2020.
[11] "An energy-saving snake locomotion gait policy obtained using deep reinforcement learning." IEEE/ASME T-Mech, 2021. arXiv:2103.04511
[12] "Learning to locomote with deep neural-network and CPG-based control in a soft snake robot." ICRA, 2020. arXiv:2001.04059
[13] "Reinforcement learning of CPG-regulated locomotion controller for a soft snake robot." IEEE T-RO, 2022. arXiv:2207.04899
[14] "Learning contact-aware CPG-based locomotion in a soft snake robot." 2021. arXiv:2105.04608

### 各向异性摩擦
[15] Tirado, et al. "Earthworm-inspired soft skin crawling robot." Advanced Science, 2024.
[16] "Frictional anisotropic locomotion and adaptive neural control for a soft crawling robot." Soft Robotics, 2022.
[17] "Multi-material 3D printing of caterpillar-inspired soft crawling robots with anisotropic friction feet." Robotica, 2020.

### 蠕动波优化
[18] "Peristaltic waves as optimal gaits in metameric bio-inspired robots." IEEE RA-L, 2020.
[19] "Addition of a peristaltic wave improves multi-legged locomotion performance on complex terrains." arXiv:2410.01046, 2024.

### 管道机器人
[20] "A snake robot for locomotion in a pipe using trapezium-like travelling wave." Mechanism and Machine Theory, 2021.
[21] "A minimally designed soft crawling robot for robust locomotion in unstructured pipes." Bioinspiration & Biomimetics, 2022.

### MuJoCo 与仿真
[22] "A dynamic simulation of a compliant worm robot amenable to neural control." NSF Proceedings, 2023.
[23] "MuJoCo Playground: An open-source framework for GPU-accelerated robot learning." RSS 2025.

### Sim-to-Real
[24] "Sim-to-real transfer for quadrupedal locomotion via terrain transformer." IEEE T-RO, 2023. arXiv:2212.07740
[25] "Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots." 2025. arXiv:2509.06342

### 综述
[26] "Snake robots: A state-of-the-art review on design, locomotion, control, and real-world applications." Mechanism and Machine Theory, 2025.
[27] "Actuation and design innovations in earthworm-inspired soft robots: A review." Frontiers in Bioengineering, 2023.
[28] "A systematic review of deep reinforcement learning for legged robot locomotion." Robotics (MDPI), 2023.

### 基础
[29] "The mechanics of slithering locomotion." PNAS, 2009.
[30] "A comprehensive study on the locomotion characteristics of a metameric earthworm-like robot." Multibody System Dynamics, 2014.
