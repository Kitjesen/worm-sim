# Worm Robot — Paper Library

论文库，用于支撑蠕虫机器人 RL 训练策略的技术决策。

## 当前研究方向：进化强化学习（Evolutionary RL）

**目标**：让蠕虫机器人运动速度更快，通过种群进化 + RL 梯度优化的混合方法。

### 核心论文

#### 1. EPO — Evolutionary Policy Optimization (2025 CMU) ⭐ 首选方案

- **论文**: `2503.19037_EPO_evolutionary_policy_optimization.pdf`
- **作者**: Jianren Wang, Yifan Su, Abhinav Gupta, Deepak Pathak (CMU Robotics Institute)
- **arXiv**: https://arxiv.org/abs/2503.19037
- **项目页**: https://yifansu1301.github.io/EPO/
- **代码**: https://github.com/lucidrains/evolutionary-policy-optimization

**核心思想**:
- 维护 K 个 agent 的种群，每个 agent 有唯一的 latent embedding (φ_k)，称为"基因"
- 所有 agent **共享同一个 actor-critic 网络** (θ, ψ)，只靠 latent 区分行为 → 省显存
- Actor: π(a|s, φ_k)，Critic: V(s, φ_k)，latent 作为额外输入
- 每代：所有 agent 并行在环境中评估 → 按 fitness 排名 → 选择/交叉/变异 latent
- Master agent (π_1) 用混合更新：on-policy (自己的数据) + off-policy (种群数据, importance sampling)
- 非 master agent 只用 on-policy 更新，保持多样性
- 进化操作只在 latent 空间做（低维），不动网络权重 → 进化算子高效

**关键算法 (Algorithm 1)**:
```
1. 初始化 K 个 agent，每个有 latent φ_k，共享网络 (θ, ψ)
2. 每轮迭代:
   a. 所有 agent 在环境中采集数据，计算 fitness f_k
   b. 如果 fitness 分化足够大 (max-min > γ·median):
      - 选择 top-x 作为精英 (保留)
      - 对精英做 crossover: φ' = 0.5 × (φ_i + φ_j)
      - 做 mutation: φ'' = φ' + N(0, σ²)
      - 替换表现差的 agent
   c. Master agent 混合更新: L = L_on + λ·L_off
   d. 其他 agent 只做 on-policy 更新
```

**实验结果**: 在 ANYmal Walking、Unitree A1 Parkour、DeepMind Humanoid 等任务上全面超越 PPO、SAC、PBT、CEM-RL。尤其在复杂任务（跑酷、双臂操作）上优势显著。

**为什么适合我们**:
- 基于 PPO，和我们现有代码兼容
- 种群共享网络，不需要 K 倍显存
- 进化在 latent 空间做，维度低，GA 操作高效
- 天然支持并行仿真（MJX GPU 批量环境）
- fitness 可以直接设为 forward_speed → 速度最大化

---

#### 2. ERL-Re² — 共享表征的进化 RL (2023 ICLR)

- **论文**: `2210.17375_ERL_Re2_evolutionary_RL.pdf`
- **作者**: Pengyi Li, Hongyao Tang, Jianye Hao, Yan Zheng, Xian Fu, Zhaopeng Meng
- **arXiv**: https://arxiv.org/abs/2210.17375
- **代码**: https://github.com/yeshenpy/ERL-Re2

**核心思想**:
- 种群共享一个非线性 encoder（状态表征），各自有独立的线性 policy head
- 进化操作在行为空间（线性 head 的参数）做，而不是在原始 NN 参数空间
- 一半种群用 off-policy RL 梯度更新（TD3/DDPG），一半用 GA 进化
- 共享 replay buffer，所有 agent 的经验互相利用

**和 EPO 的区别**:
- ERL-Re² 用 off-policy (TD3)，EPO 用 on-policy (PPO)
- ERL-Re² 共享 encoder 但独立 policy head；EPO 共享整个网络但用 latent 区分
- EPO 更新、更 scalable（2025 vs 2023）

---

## 实施计划

### Phase 1: 速度最大化（开环基准）
- 用 CMA-ES 优化开环步态参数 (~20 维)
- fitness = 10 秒后前进距离
- 找到蠕虫的物理速度上限

### Phase 2: EPO 训练 NN 策略
- 基于现有 WormEnvV6 + MJX GPU 环境
- 实现 EPO：K=8~16 agents，共享 actor-critic + latent
- reward = forward_speed（去掉 cmd_vel tracking，纯速度最大化）
- 进化选择淘汰慢的、保留快的 → 逐代提速

### Phase 3: Sim-to-Real
- 将最优策略导出到真机
- Domain randomization + 系统辨识

---

## 其他相关论文（已收集）

| 论文 | 主题 | 文件 |
|------|------|------|
| Isaac Gym | GPU 并行仿真 | `2108.10470_isaac_gym.pdf` |
| Isaac Lab | 机器人学习框架 | `2511.04831_isaac_lab.pdf` |
| Nature 2024 Curriculum | 敏捷运动课程学习 | `nature2024_learning_agility_curriculum.pdf` |
| Sim2Real Systematic | 系统化 sim-to-real | `2509.06342_systematic_sim2real.pdf` |
| Gait-conditioned RL | 步态条件化 RL | `2505.20619_gait_conditioned_RL.pdf` |
| CPG + DRL Snake | CPG+深度RL蛇形 | `2001.04059_CPG_DRL_soft_snake.pdf` |
| Peristaltic Wave | 蠕动波多足 | `2410.01046_peristaltic_wave_multilegged.pdf` |

## 参考资源

- [Awesome Evolutionary RL](https://github.com/yeshenpy/Awesome-Evolutionary-Reinforcement-Learning) — 进化 RL 论文列表
- [EPO Project Page](https://yifansu1301.github.io/EPO/) — 可视化和补充材料
- [PBRL GPU-Scale](https://github.com/Asad-Shahid/PBRL) — 种群并行 RL (Isaac Gym)
