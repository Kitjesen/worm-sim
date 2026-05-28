# 可部署多模态蛇/蠕虫机器人论文计划

## 核心主张

本文研究一个具有伸缩关节和偏航关节的多体节机器人，在平地、沙地、坡地上的蛇形、蠕动和混合运动模式选择规律。控制器必须可实际部署：策略输入只来自关节编码器、每体节 IMU、目标命令、上一时刻动作和相位，不使用 MuJoCo freejoint 线速度、全局位姿或外部定位作为策略观测。

## 方法

- 使用 `src/v6/worm_env_v6.py` 的 80 维可部署观测：
  - command: `cmd_vel_norm`, `cmd_yaw_norm`, `gait_blend`
  - 11 维关节位置，11 维关节速度，11 维上一时刻动作
  - 7 个体节 IMU 的局部重力方向和局部角速度
  - phase clock: `sin`, `cos`
- 使用 PPO 训练 mode-conditioned policy。
- `gait_blend=0.0` 表示蠕动，`gait_blend=1.0` 表示蛇形，中间值表示混合。
- 训练时可打开编码器噪声、IMU 噪声和 1 步动作延迟，作为部署鲁棒性扰动。

## 实验矩阵

主矩阵：

| Terrain | Modes |
| --- | --- |
| flat | worm, snake, mixed, random |
| sand | worm, snake, mixed, random |
| slope | worm, snake, mixed, random |

说明：

- `worm/snake/mixed` 用于固定模式对比。
- `random` 用于训练连续 `gait_blend` 条件策略，并在评估时扫描 `0.0, 0.25, 0.5, 0.75, 1.0`。
- CMA-ES 的 `peristaltic/serpentine/full` 作为开环基线。

## 指标

- 平均前进速度：`mean_speed_mm_s`
- 速度方差：`std_speed_mm_s`
- 横向漂移：`mean_lateral_drift_mm`
- 单位距离动作代价：`mean_action_l2_per_m`
- 单步动作代价：`mean_action_l2_per_step`
- 动作变化代价：`mean_action_rate_l2_per_step`
- 路径效率/打滑 proxy：`mean_path_efficiency`, `mean_slip_proxy`
- 推进效率 proxy：`mean_propulsion_efficiency_m_per_action_l2`
- 成功率，坡地成功率：`success_rate`, `slope_success_rate`
- 沙地打滑/推进效率：`sand_slip_proxy`, `sand_propulsion_efficiency_m_per_action_l2`
- 失效率：`termination_rate`
- CMA-ES 基线速度：`best_speed_mm_s`

## 预期图表

- 三地形固定模式速度柱状图：worm vs snake vs mixed。
- 三地形连续 `gait_blend` 扫描曲线。
- 三地形最优 `gait_blend` 表格。
- 速度与动作代价散点图，说明混合模式是否带来效率优势。
- 实机验证表：flat/sand/slope 的短距离运行、速度估计、视频文件和日志校验结果。

自动生成的图表路径：

- `record/v6/paper_results/fixed_mode_speed.svg`
- `record/v6/paper_results/blend_scan_speed.svg`

## 实机验证要求

每条实机日志必须符合 `record/v6/hardware/hardware_log_template_schema.json`：

- 每行包含完整 80 维策略输入。
- 每行包含 11 维动作。
- 每行包含速度估计、偏航角速度估计和视频文件名。
- 可先用 `record/v6/hardware/hardware_log_template_example.csv` 校验硬件端字段顺序和数值范围。
- 校验命令：

```powershell
python src\v6\validate_hardware_log_v6.py --input record\v6\hardware\<log>.csv
```

## 复现实验命令

快速检查：

```powershell
python src\v6\run_paper_pipeline_v6.py --preset smoke
```

正式训练、评估、扫描、汇总：

```powershell
python src\v6\run_paper_pipeline_v6.py --preset formal --timesteps 1000000 --n-envs 4
```

正式训练并记录代表性视频：

```powershell
python src\v6\run_paper_pipeline_v6.py --preset formal --timesteps 1000000 --n-envs 4 --video
```

只预览正式命令：

```powershell
python src\v6\run_paper_pipeline_v6.py --preset formal --dry-run
```

单独运行部署鲁棒性评估：

```powershell
python src\v6\run_terrain_experiments.py --method robust-eval --terrain flat sand slope --mode worm snake mixed
```

导出可部署策略包：

```powershell
python src\v6\run_terrain_experiments.py --method deploy-export --terrain flat sand slope --mode random
```

用硬件观测日志回放策略动作：

```powershell
python src\v6\deploy_policy_v6.py replay --bundle-dir record\v6\deploy_bundles\flat_random --input-csv record\v6\hardware\hardware_log_template_example.csv
```

从原始硬件传感器日志构造 80 维策略输入日志：

```powershell
python src\v6\build_hardware_obs_v6.py --input-raw record\v6\hardware\raw_hardware_log_template_example.csv --output record\v6\hardware\hardware_log_from_raw_example.csv
```

审计论文目标完成度：

```powershell
python src\v6\audit_paper_goal_v6.py
```

## 完成标准

- `test_deployable_obs_v6.py` 通过，证明策略观测不含仿真线速度/全局位姿。
- 12 个 RL 训练目录存在：三地形乘以 `worm/snake/mixed/random`。
- 9 个固定模式评估 JSON 存在：三地形乘以 `worm/snake/mixed`。
- 9 个鲁棒性评估 JSON 存在：三地形乘以 `worm/snake/mixed`，文件名为 `eval_metrics_robust.json`。
- 3 个连续 blend scan JSON 存在：三地形乘以 `random` policy。
- `record/v6/paper_results/summary.md` 和三份 CSV 汇总表生成。
- `record/v6/paper_results/completion_audit.json` 显示 `complete=true`。
- 三个 `record/v6/deploy_bundles/<terrain>_random` 策略包存在，包含 `policy_actor.pt` 和 `deploy_config.json`。
- flat/sand/slope 三条实机 CSV 通过 `validate_hardware_log_v6.py` 校验，并能对应到视频文件。
