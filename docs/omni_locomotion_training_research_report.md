# Worm V6 全向/多模态运动训练调研报告

Last updated: 2026-06-01

## 0. 结论先行

现在最合理的论文与训练目标不是“像麦克纳姆轮一样的完整全向底盘”，而是：

> 在物理可行速度域内，使用可部署观测的蛇-蠕虫双模态机器人，实现命令条件的多模态平面运动控制。

英文建议表述：

> finite-envelope command-conditioned multimodal planar locomotion

这个表述比直接说 fully omnidirectional locomotion 更稳。蛇形/蠕虫机器人依赖身体-地面摩擦产生推进，横移和纯 yaw 本来就不是独立驱动通道。训练策略必须先学出可行速度包络，再在包络内做 `vx, vy, yaw_rate` 跟踪。

对我们 V6 的直接建议：

1. 保留当前路线：`CMA-ES/CPG-like gait prior + PPO residual + learned latent gait gate`。
2. 不再裸训整个矩形命令空间。先测可行速度域，然后训练包络内命令。
3. 纯 yaw 单独降低命令范围，从 `±0.08 ~ ±0.12 rad/s` 开始，先要求“不蜷缩、不平移漂移”，再扩展到 `±0.25 rad/s`。
4. 混合 `vx+vy` 不要一开始要求满幅对角线。先训练半幅 diagonal，再扩到满幅。
5. gate 论文表述要谨慎：目前 gate 有命令条件中心和奖励正则，不能强说完全自发涌现；应做 fixed-gate / learned-gate / no-gate ablation。

## 1. 我们当前训练状态

最新完整评估来自：

`runs/worm_v6_ppo_flat_random_v59_yaw_preserve_hardcase_from_v56best/best_eval_summary.json`

关键指标：

| 指标 | V59 当前值 | 目标 | 状态 |
|---|---:|---:|---|
| `tracking_gate_passed` | false | true | 未通过 |
| `direction_gate_passed` | false | true | 未通过 |
| `planar_velocity_rmse_m_s` | 0.1331 | <= 0.10 | 未通过 |
| `yaw_rate_rmse_rad_s` | 0.1234 | <= 0.20 | 固定评估达标 |
| `mean_off_axis_speed_m_s` | 0.0261 | <= 0.08 | 达标 |
| `zero_command_mean_speed_m_s` | 0.000017 | <= 0.02 | 达标 |
| `planar_success_rate` | 0.810 | >= 0.85 | 未通过 |
| `yaw_success_rate` | 0.762 | >= 0.70 | 达标 |
| `wrong_planar_sign_count` | 0 | 0 | 达标 |
| `wrong_yaw_sign_count` | 1 | 0 | 未通过 |
| `straight_violation_count` | 4 | <= 1 | 未通过 |
| `stationary_violation_count` | 4 | 越低越好 | 未通过 |

当前能力判断：

- stop `(0,0,0)` 已经很好，不需要优先单独加训。
- 纯 planar 方向符号基本对，但速度幅值和直线 yaw drift 还不够。
- 纯 yaw 可以产生方向正确的转动，但存在平移漂移和蜷缩风险。
- 混合 `vx+vy` 或 `vx+yaw` 仍然容易只表达一个分量，说明 action prior 组合与 residual authority 还没有完全解决。

重要状态：

- 80D observation ABI 保持不变。
- 12D action ABI 保持不变：`11D residual motor action + 1D learned latent gait gate`。
- v29 reward 已加入“纯 yaw 防蜷缩 compactness penalty”，但 V59 训练结果仍是 v28 reward 产物，需要用 v29 重新续训。

## 2. 文献分型：别人到底在训什么

### 2.1 CPG / gait prior + RL 调参

代表工作：

- Satheeshbabu et al., "Learning to Locomote with Deep Neural-Network and CPG-based Control in a Soft Snake Robot", arXiv 2020. Source: https://arxiv.org/abs/2001.04059
- Liu, Onal and Fu, "Integrating Contact-aware Feedback CPG System for Learning-based Soft Snake Robot Locomotion Controllers", arXiv 2023. Source: https://arxiv.org/abs/2309.02781
- Soft snake CPG-regulated RL line, local library includes `2207.04899_RL_CPG_snake_robot.pdf`.

共同点：

- 不把高维身体控制完全交给神经网络裸学。
- 用 CPG 或周期 gait generator 提供稳定节律。
- RL 学的是 CPG 参数、反馈增益、残差或局部修正。
- 重点是让学习器站在“已有可行步态”的基础上优化，而不是从零发现步态。

对 V6 的启发：

- 我们的 CMA-ES gait priors 等价于一组 CPG-like rhythmic motor primitives。
- PPO residual 是合理的，因为它只修正 gait prior，不承担从零生成步态的全部负担。
- 但是现在 residual 的任务过重：既要补速度误差，又要抵消错误 prior，又要解决 yaw drift。下一步应减少 prior 与命令之间的冲突。

### 2.2 直接学习 gait，但目标通常是“有效步态”而不是严格全向速度跟踪

代表工作：

- Shi et al., "Deep Reinforcement Learning for Snake Robot Locomotion", IFAC 2020. Source: https://www.sciencedirect.com/science/article/pii/S2405896320333772
- Local library: `2103.04511_energy_saving_snake_PPO.pdf`, `1904.07788_energy_efficient_snake_RL.pdf`

共同点：

- RL 可以学出蛇形机器人的 terrestrial / aquatic gait。
- 训练目标多是速度、能耗、姿态稳定或环境通过性。
- 很少直接承诺任意 `vx, vy, yaw_rate` 的连续精确跟踪。

对 V6 的启发：

- 我们不能只用“视频看起来会动”作为论文指标。
- 必须区分：
  - gait generation：能不能产生有效步态；
  - command tracking：能不能按给定速度跟踪；
  - feasible envelope：给定命令是否物理可实现。

### 2.3 路径跟踪 / actor-critic optimal tracking

代表工作：

- 用户给出的本地论文：`D:/software/webDownload/Actor-Critic_Framework-Based_on_Optimal_Tracking_Strategy_for_Snake_Robots_with_Reinforcement_Learning_Method.pdf`
- 本地笔记：`docs/paper_actor_critic_tracking_snake_robot_notes.md`

共同点：

- 这类工作更像“给定路径/目标轨迹，蛇机器人调节 gait 参数跟踪路径”。
- 它解决的是路径跟踪或最优跟踪问题，不等同于底层 `vx, vy, yaw_rate` 速度命令跟踪。

对 V6 的启发：

- 我们可以借鉴 tracking error 的写法和 actor-critic 结构表述。
- 但不要照搬问题定义。我们的控制接口更接近速度命令跟踪，而不是给定曲线的路径跟踪。

### 2.4 蠕动机器人 RL 与 peristaltic gait

代表工作：

- Seto et al., "Acquisition of Movement Pattern by Reinforcement Learning in Peristaltic Crawling Robot", local library `2006_tesen_qdsega_peristaltic_RL_robot.pdf`
- Tesen/Ikeda/Saga Q-learning peristaltic crawling line, local library `2010_peristaltic_crawling_Q_learning_environment_change.pdf`
- Saga et al., "Acquisition of Earthworm-Like Movement Patterns of Many-Segmented Peristaltic Crawling Robots", DOI: https://doi.org/10.1177/1729881416657740
- Kandhari et al., "An Analysis of Peristaltic Locomotion for Maximizing Velocity or Minimizing Cost of Transport of Earthworm-Like Robots", local library `2021_kandhari_peristaltic_velocity_cost_transport.pdf`

共同点：

- 早期 peristaltic RL 常是低维、离散、Q-learning 或 actor-critic。
- 很多工作重点是轴向推进、速度/能耗折中，不是横移或全向。
- peristaltic gait 天然适合前后轴向运动，横移和原地 yaw 不是它的强项。

对 V6 的启发：

- 蠕虫模式应该主要承担 `vx` 轴向推进、低速稳定、抗扰和接触锚定。
- 蛇形模式更适合 yaw 和侧向 steering。
- learned gait gate 的合理目标不是“所有命令都自动混合”，而是学会在命令与接触条件下切换不同先验权重。

### 2.5 沙地/坡地：sidewinding 与最小滑移

代表工作：

- Marvi et al., "Sidewinding with Minimal Slip: Snake and Robot Ascent of Sandy Slopes", Science/arXiv 2014. Source: https://arxiv.org/abs/1410.2945
- Gong et al., "Snakes on an Inclined Plane: Learning an Adaptive Sidewinding Motion for Changing Slopes", local library `2013_gong_adaptive_sidewinding_changing_slopes.pdf`
- Rozaidi et al., "HISSbot: Sidewinding with a Soft Snake Robot", local library `2303.15732_hissbot_sidewinding_soft_snake.pdf`

共同点：

- 沙地和坡地里，关键不是最大关节幅值，而是接触段、抬升段、滑移方向和地形摩擦。
- sidewinding 的核心是减少接触点滑移。
- 地形变化会改变最优 gait 参数，固定平地 gait 直接迁移很可能失败。

对 V6 的启发：

- 不应在 flat 未通过时启动 sand/slope 正式训练。
- flat 通过后迁移地形时，奖励要增加 slip proxy、姿态高度/翻滚安全、坡向进展。
- 沙地/坡地评估要包含轨迹、滑移、能耗，而不仅是速度。

### 2.6 腿式机器人速度命令跟踪的经验

代表工作：

- legged_gym / ANYmal 系列开源训练框架。Source: https://github.com/leggedrobotics/legged_gym
- Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots", arXiv 2021. Source: https://arxiv.org/abs/2107.04034
- Margolis et al., "Rapid Locomotion via Reinforcement Learning", IJRR 2024. Source: https://doi.org/10.1177/02783649231224053

共同点：

- policy 输入常包含 commanded `vx, vy, yaw_rate`。
- reward 有线速度跟踪、角速度跟踪、动作平滑、能量、姿态稳定、接触安全。
- 训练不是无限速度空间，而是逐步扩展 command curriculum。
- 高速平移 + 高角速度组合可能物理不可行，必须用可行速度域裁剪。

对 V6 的启发：

- 我们的 80D observation 设计是合理的：命令、编码器、previous action、IMU、phase clock。
- 但是 command range 要从“论文愿望”改成“机器人能做到的包络”。
- 评估要同时看 tracking RMSE 和 direction gate，不能只看 PPO reward。

### 2.7 latent gait / gait representation

代表工作：

- Wu et al., "Learning Multiple Gaits within Latent Space for Quadruped Robots", arXiv 2023. Source: https://arxiv.org/abs/2308.03014
- Mitchell et al., "Gaitor: Learning a Unified Representation Across Gaits for Real-World Quadruped Locomotion", arXiv 2024/local library `2405.19452_gaitor_unified_latent_gaits.pdf`
- Zhang et al., "SYNLOCO: Synthesizing Central Pattern Generator and Reinforcement Learning for Quadruped Locomotion", local library `2310.06606_synloco_CPG_RL_quadruped.pdf`

共同点：

- latent gait 有价值，但论文必须证明 latent 变量对应可解释行为变化。
- 仅展示 gate 数值不够，需要 fixed-gate 对比、gate sweep、ablation。

对 V6 的启发：

- 如果我们说 learned latent gait gate，就要补三个实验：
  1. fixed worm / fixed mixed / fixed snake；
  2. learned gate；
  3. no-gate 或 command-only gate；
  4. gate sweep 与速度/轨迹/能耗曲线。

## 3. 我们当前路线是否合理

当前路线总体合理：

```text
80D deployable observation
        |
        v
SB3 PPO ActorCriticPolicy
        |
        v
12D policy action = 11D residual + 1D gait gate
        |
        v
CMA-ES rhythmic gait prior + residual correction
        |
        v
11D motor command
```

它与文献里的 CPG+RL、motor-primitives+RL、latent-gait 思路一致。

但当前失败点也很清楚：

1. **命令空间过大**
   直接训练 `vx=[-0.25,0.25]`, `vy=[-0.15,0.15]`, `yaw=[-0.25,0.25]` 的组合，对蛇/蠕虫本体太激进。

2. **纯 yaw 是主要瓶颈**
   yaw 指令能转，但带平移漂移，并可能蜷缩。v29 防蜷缩 reward 正是为这个问题加入。

3. **mixed command 不是简单线性叠加**
   `vx+vy` 或 `vx+yaw` 的 motor primitive 会互相干扰。当前 dominant prior 和 componentwise prior 都没有稳定解决这个问题。

4. **gait gate 的论文风险**
   当前 gate 有 command-conditioned center 与 reward target。工程上有效，但论文上必须承认这是“prior-regularized learned gate”，不能夸成完全自发涌现。

5. **奖励太多但课程还不够分层**
   现在奖励项已经覆盖很多目标，但训练仍然在同一个策略里同时解决太多难题。下一步应该拆课程，而不是继续堆 reward。

## 4. 建议的训练路线

### Stage A: 可行速度包络测量

目标：先知道机器人能做到什么，而不是假设它能全向。

命令网格：

```text
vx:  -0.20, -0.10, 0.00, 0.10, 0.20
vy:  -0.12, -0.06, 0.00, 0.06, 0.12
yaw: -0.20, -0.10, 0.00, 0.10, 0.20
```

先不要全组合爆炸扫描，分三类：

1. axis-only: `(vx,0,0)`, `(0,vy,0)`, `(0,0,yaw)`
2. planar diagonal: `(vx,vy,0)`
3. forward-yaw: `(vx,0,yaw)`

输出：

- feasible / marginal / infeasible 标签；
- 每个命令的 measured `body_vx`, `body_vy`, `yaw_rate`；
- planar RMSE、yaw RMSE、off-axis、yaw drift；
- body extent / head-tail / arc ratio，专门看纯 yaw 蜷缩。

### Stage B: 纯 yaw 修复课程

目标：先让 `(0,0,yaw)` 变成可控、不开团、不平移漂移。

训练范围：

```text
yaw = ±0.08, ±0.12
vx = 0
vy = 0
```

奖励重点：

- yaw sign；
- yaw-rate magnitude；
- planar drift penalty；
- v29 body compactness penalty；
- head-tail / arc ratio；
- smooth action；
- stop/yaw alternating，防止策略把 yaw 当成前进波。

通过标准：

- `wrong_yaw_sign_count == 0`
- pure yaw mean planar speed `<= 0.04 m/s`
- body extent ratio 不低于阈值；
- yaw RMSE 在低速 yaw 范围内 `<= 0.08 rad/s`。

### Stage C: 轴向 + 横向半幅课程

目标：修复 planar direction 与 straight drift。

训练范围：

```text
vx = ±0.05, ±0.10, ±0.15
vy = ±0.04, ±0.08
yaw = 0
```

奖励重点：

- signed component tracking；
- zero-yaw heading hold；
- off-axis speed；
- yaw drift；
- 蠕动 slide wave preservation，用来恢复可见蠕虫式轴向作动。

通过标准：

- planar RMSE `<= 0.10 m/s`
- wrong planar sign `== 0`
- straight violation `<= 1`
- lateral left/right 都能稳定通过 6s 固定评估。

### Stage D: 半幅 mixed command

目标：先学半幅 diagonal，不直接上满幅。

训练范围：

```text
vx = ±0.10, ±0.15
vy = ±0.04, ±0.08
yaw = 0
```

以及：

```text
vx = 0.10, 0.15
yaw = ±0.08, ±0.12
vy = 0
```

核心策略：

- mixed command 只在 axis-only 通过后进入。
- prior 只提供稳定节律，mixed 分量尽量交给 residual 学。
- 对 full diagonal 使用 curriculum expansion，不再和 axis-only 同时硬训。

### Stage E: 扩展到论文速度包络

只有 A-D 达标后，再扩展：

```text
vx up to ±0.25
vy up to ±0.15
yaw up to ±0.25
```

如果满幅不可行，论文中应报告 feasible envelope，而不是隐藏失败。

## 5. 下一轮 V60/V61 实施建议

### V60: low-yaw compact repair

目的：先吃到 v29 纯 yaw 防蜷缩奖励，并把训练命令收缩到低速 yaw 可行包络。第一轮不再继续硬训 full continuous omni。

建议命令：

```powershell
python src\v6\train_v6.py `
  --terrain flat `
  --gait-mode random `
  --run-label flat_random_v60_low_yaw_envelope_from_v59best `
  --command-curriculum low_yaw_envelope `
  --timesteps 1200000 `
  --train-chunk-timesteps 200000 `
  --resume runs\worm_v6_ppo_flat_random_v59_yaw_preserve_hardcase_from_v56best\best_model.zip `
  --allow-contract-resume `
  --allow-curriculum-resume `
  --policy-net-arch 512,256,128 `
  --value-net-arch 512,256,128 `
  --learning-rate 3e-5 `
  --n-envs 4 `
  --device cpu
```

说明：

- `--allow-contract-resume` 是因为 v29 reward_contract 比 V59 的 v28 多了 pure-yaw compactness 项。
- `low_yaw_envelope` 是已实现训练课程：30% stop、40% pure yaw `±0.08..±0.12 rad/s`、20% slow axial `±0.05..±0.10 m/s`、10% slow forward + yaw。
- 这不是破坏 ABI：observation/action/actuator contract 不变，只是训练 reward 改了。

### V61: mixed-envelope rejoin

如果 V60 的 pure yaw 不再蜷缩、平移漂移下降，再回到 mixed-planar/yaw-preserve 课程：

```powershell
python src\v6\train_v6.py `
  --terrain flat `
  --gait-mode random `
  --run-label flat_random_v61_mixed_rejoin_from_v60best `
  --command-curriculum mixed_planar_yaw_preserve_repair `
  --timesteps 1400000 `
  --train-chunk-timesteps 200000 `
  --resume runs\worm_v6_ppo_flat_random_v60_low_yaw_envelope_from_v59best\best_model.zip `
  --allow-curriculum-resume `
  --policy-net-arch 512,256,128 `
  --value-net-arch 512,256,128 `
  --learning-rate 3e-5 `
  --n-envs 4 `
  --device cpu
```

这一步再检查 mixed `vx+vy`、`vx+yaw`。如果又出现纯 yaw 蜷缩或 stationary drift，退回 V60 继续低速 yaw 修复。

## 6. 论文实验矩阵建议

### 必做实验

1. **Fixed mode baseline**
   - worm only
   - snake only
   - mixed fixed
   - CMA-ES open-loop

2. **Learned gate ablation**
   - fixed gate
   - learned gate with regularizer
   - learned gate without gate-target regularizer 或退火 gate-target

3. **Command tracking**
   - axis-only 6 directions
   - feasible-envelope scan
   - continuous sweep video
   - trajectory and segment time series

4. **Yaw stability**
   - pure yaw with body compactness metric
   - head-tail distance / body arc / body extent
   - planar drift under pure yaw

5. **Terrain transfer**
   - only after flat passes
   - sand and slope report speed, slip proxy, success rate, energy/action cost

### 不应提前声称

- 不应声称“完整全向”。
- 不应声称“RL 比 CMA-ES 更快”，当前 CMA-ES 仍是速度上界。
- 不应声称 gate 完全自发涌现，除非完成 no-regularizer 和 fixed-gate 消融。

### 可以诚实声称

- 可部署观测契约：没有使用全局位姿/速度作为 policy 输入。
- 多模态 action prior：蠕动、蛇形、混合 gait anchors。
- residual RL：学习命令条件修正。
- finite-envelope planar command tracking：达标后可以写。
- pure-yaw body-shape regularization：针对长体节机器人 yaw 蜷缩问题的训练约束。

## 7. 推荐奖励设计方向

保留：

- body-frame velocity tracking；
- yaw-rate tracking；
- off-axis suppression；
- zero-command speed penalty；
- action smoothness；
- energy/action cost；
- gait gate regularizer，但论文中要说明它是弱正则或做退火/消融。

新增或加强：

- pure yaw body compactness penalty；
- pure yaw planar drift penalty；
- zero-yaw heading hold；
- component-wise tracking，而不是只看投影速度；
- feasible-envelope-based command clipping；
- mixed command curriculum weight，而不是全空间均匀采样。

不建议继续做：

- 在 flat 没过前启动 sand/slope 正式训练；
- 继续加大 yaw 指令到 `±0.5 rad/s`；
- 继续把 full diagonal 和 pure yaw 放在同一早期课程里硬训；
- 单纯提高网络大小期望解决物理不可行命令。

## 8. 推荐验收标准

第一阶段不要直接验收满幅全向，而是验收 low-speed envelope：

```text
vx:  ±0.10, ±0.15
vy:  ±0.04, ±0.08
yaw: ±0.08, ±0.12
```

Low-speed envelope pass:

- wrong planar sign count `== 0`
- wrong yaw sign count `== 0`
- planar RMSE `<= 0.08 m/s`
- yaw RMSE `<= 0.08 rad/s`
- pure yaw planar drift `<= 0.04 m/s`
- zero command speed `<= 0.02 m/s`
- straight yaw drift `<= 0.2 rad / 6s`
- body compactness violation count `== 0`

Full envelope pass:

- planar RMSE `<= 0.10 m/s`
- yaw RMSE `<= 0.20 rad/s`
- mean off-axis speed `<= 0.08 m/s`
- planar success rate `>= 0.85`
- yaw success rate `>= 0.70`
- straight violation count `<= 1`
- stationary violation count `<= 1`

## 9. 最终建议

下一步不要泛化地“继续训练全向”。应该执行：

1. V60：用 v29 reward 和 `low_yaw_envelope` 从 V59 best 继续，验证纯 yaw 防蜷缩是否有效。
2. 做 feasible envelope scan，把不可实现命令从训练目标中移除或降权。
3. 如果 V60 通过，V61 再回到 mixed-planar/yaw-preserve 课程。
4. flat low-speed envelope 通过后，再扩展 full envelope。
5. full envelope 通过后，再启动 sand/slope。

这条路线与蛇/蠕虫机器人文献更一致：先有稳定周期步态和可行速度域，再让 RL 学命令条件修正，而不是让一个 PPO 策略在高维摩擦接触系统里一次性学会完整全向。
