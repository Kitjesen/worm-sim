# Worm Robot Simulation — Changelog

## [V5.1] - 2026-03-11 — 刚体+轮子架构 (worm_v5_1.py)

### 架构重构 — 基于 v6.urdf 真实硬件设计
- **设计思路**: 从 cable composites (327 bodies, 1116 DOF) 重构为刚体+轮子 (16 bodies, 24 DOF)
- **参考**: `D:\robot\snake_worm\v6\urdf\v6.urdf` — SolidWorks 导出的 5 段蠕虫机器人
- **拓扑**: 每个体节 = 1 个 capsule 刚体 + 2 个圆柱轮子, 通过 slide+yaw 关节串联
  ```
  [seg0+wheels] ─slide+yaw─ [seg1+wheels] ─slide+yaw─ [seg2+wheels] ─slide+yaw─ [seg3+wheels] ─slide+yaw─ [seg4+wheels]
     (tail)                                                                                                    (head)
  ```
- **物理尺寸 (匹配 URDF)**:
  - 体节: capsule 80mm, 半径 22mm, 质量 450g
  - 轮子: cylinder 半径 15mm, 厚 10mm, 质量 48g
  - 间距: 105mm center-to-center
  - 总重: 2.73 kg (5 × 0.45 + 10 × 0.048)
- **驱动器**: 18 总 (10 wheel velocity + 4 slide position + 4 yaw position)
- **完全自包含**: 不依赖 exp_runner.py, 内置 XML 生成器

### 三种运动模式
| 模式 | 控制方式 | 速度 (25s) | 航向 |
|------|----------|------------|------|
| worm | Zhan/Fang 离散步态 `[0,0,0,1,1]` + 轮子锚定 | 25.6 mm/s | 0.0° |
| snake | 正弦 yaw 波 + 全轮驱动 | 42.1 mm/s | -0.0° |
| combined | 全轮驱动 + slide 蠕动波 + yaw 波 | **42.5 mm/s** | 1.1° |

**v5.1b 优化 (combined)**: 原 combined 模式沿用 worm 的轮子锚定 (刹车)，导致 snake yaw 无法贡献速度 (仅 25.3 mm/s)。
改为全轮持续驱动 + 纯 slide 蠕动波后，combined 达到 42.5 mm/s，超过 snake alone (42.1 mm/s)。

### 蠕动步态 (worm mode)
- Zhan/Fang 2019 离散逆行蠕动 `{0,0,2|1}`, 步态向量 `[0,0,0,1,1]`
- State 0 (advancing): 驱动轮子前进 (-ω, 右手定则)
- State 1 (anchored): 刹车轮子 (target=0, kv 阻尼 + 地面摩擦锚定)
- Slide 关节: head-side advancing → 伸展 (+), anchored → 收缩 (-)

### 蛇形步态 (snake mode)
- 正弦 yaw 波: amp=0.25 rad (14.3°), freq=0.5 Hz, retrograde head→tail
- 全轮恒速驱动: -3.0 rad/s → 42 mm/s 前进

### 关键技术细节
- **轮子方向**: 正角速度 (around X) 产生 -Y 摩擦 → forward drive 需要负 ω
- **armature=0.001**: 解决 velocity actuator + 极小轮惯量 (5.4e-6 kg·m²) 的数值不稳定
  - 无 armature: forcerange=±2 产生 370,000 rad/s² 加速度 → 轮子飞转 → robot 起飞
  - 有 armature: 等效惯量 1e-3 → 最大加速度 2000 rad/s² → 稳定
- **solver**: 使用 MuJoCo 默认求解器 (非 Newton, 避免深层嵌套链不稳定)

### 对比 V5 (cable composites)
| 指标 | V5 (cables) | V5.1 (rigid) |
|------|-------------|--------------|
| Bodies | 327 | 16 |
| DOF | 1116 | 24 |
| Actuators | 41 | 18 |
| Worm speed | 4.26 mm/s | 25.7 mm/s |
| Sim speed | ~2000 steps/s | ~13000 steps/s |
| 稳定性 | 零 NaN | 零 NaN |

### 文件
- `src/v3/worm_v5_1.py` — V5.1 仿真脚本（新建, 自包含）
- `record/v5_1/videos/worm_v5_1_worm.mp4`
- `record/v5_1/videos/worm_v5_1_snake.mp4`
- `record/v5_1/videos/worm_v5_1_combined.mp4`

---

## [V5.0] - 2026-03-11 — 双模态蠕虫-蛇机器人 (worm_v5.py)

### 架构 — 5 体节 + 4 摆动关节
- **灵感**: Bi, Zhou, Fang (2023) "A worm-snake-inspired metameric robot for multi-modal locomotion"
- **拓扑**: 每两个体节中间一个摆动关节，每个体节有自己独立的两个隔板
  ```
  [plate0-cables-plate1] -swing0- [plate2-cables-plate3] -swing1- [plate4-cables-plate5] -swing2- [plate6-cables-plate7] -swing3- [plate8-cables-plate9]
       体节0                         体节1                         体节2                         体节3                         体节4
  ```
  - 蠕动体节: seg 0,2,4,6,8（各含 cable composites、纵肌、环肌、斜肌）
  - 摆动关节: seg 1,3,5,7（独立刚体，6 DOF + yaw 位置驱动器，无 cable）
- **MuJoCo**: 9 segments, 10 plates, 335 bodies, 1164 DOF
- **驱动器**: 39 总（20 axial + 5 ring + 10 steer + 4 swing yaw position）
- **RL action space**: [5×contraction per 体节] + [4×yaw angle per 摆动关节]

### 摆动关节设计
- 每个关节 6 自由度（3 slide + 3 hinge），通过 `connect` 约束连接两侧板子
  - X slide: stiffness=200（横向稳定）
  - Y slide: stiffness=10（轴向柔顺，传递蠕动力）
  - Z slide: stiffness=500（防下沉）
  - pitch/roll: stiffness=50（防翘曲）
  - yaw: stiffness=0（自由偏航，由 position actuator 控制）

### 三种运动模式
| 模式 | 控制 | 位移 (15s) | 速度 |
|------|------|------------|------|
| worm | 5 段蠕动波 `[0,0,0,1,1]` | 13.0 mm (+Y) | 0.87 mm/s |
| combined | 蠕动 + 4 关节正弦波 | 970 mm | 64.7 mm/s |

- 手动步态仅用于验证架构，RL 训练将学习最优双模态控制策略

### exp_runner.py 修改
- 新增参数: `swing_segments=""` (逗号分隔的摆动关节段索引)
- Cable/ring/tendon 生成跳过摆动关节段
- 摆动关节体、connect 约束、yaw 位置驱动器注入到两种 XML 模板

### 视觉渲染
- 钢片仅渲染蠕动体节（跳过摆动关节段）
- 摆动关节可视化为小球 + capsule 脚
- 沿用 V4.1 渲染方案（黑色厚钢片、xmat 锚定、三层 cable 隐藏）

### 文件
- `src/v3/worm_v5.py` — V5 仿真脚本（新建）
- `src/v3/exp_runner.py` — XML 生成器（修改：摆动关节支持）
- `record/v5/videos/worm_v5_combined.mp4`
- `record/v5/videos/worm_v5_worm.mp4`

---

## [V4.1] - 2026-03-10 — 真实尺寸渲染优化 (worm_v4.py)

### 视觉重构
- **真实机器人尺寸**: 板间距 101.2mm, 板直径 110mm, 钢片长度 115.2mm, 钢片宽度 21mm
- **钢片渲染**: 纯视觉注入 BOX geom（不依赖 cable composite 几何体）
  - 厚度: 0.3mm → 4mm（厚实黑色带子，匹配实物外观）
  - 顶点数: 20 → 40（每条钢片 39 段，消除断裂感）
  - 重叠: 5% → 25%（连续平滑的桶形拱带）
  - 弓度: 28mm（动态缩放 `bow_scale = SEG_LENGTH / body_len`）
- **全黑色调**: 钢片 `[0.10, 0.10, 0.12]` + 板 `[0.08, 0.08, 0.10]`，靠 3D 形状和金属高光区分
  - emission=0.12, specular=0.5, shininess=0.4
- **消除钢片扭转**: 每条钢片两端用板子实际朝向 `d.xmat` 锚定，沿体节线性插值（替代全局 world_x 参考）

### Cable 隐藏（三层）
1. **模型级**: `m.geom_rgba=[0,0,0,0]` + `m.geom_group=4`（按 body name 匹配 cable/ring）
2. **渲染选项**: `vopt.geomgroup[4]=0` + 隐藏 tendon/joint
3. **场景级**: `hide_cable_geoms()` 隐藏所有非板非地面 geom（带 `n_orig` 保护注入的钢片 geom）

### 物理调参
- `axial_muscle_force`: 50 → 100N（可见蠕动压缩，无向上翘曲）
- `bend_stiff=1e8`, `plate_stiff_x=500`（直线模式不变）
- 补光灯: `exp_runner.py` 添加 fill light

### 性能
| 指标 | V4.0 (force=50) | V4.1 (force=100) |
|------|----------------|------------------|
| 位移 (10s) | ~16mm | **40mm** |
| 速度 | 1.6 mm/s | **4.0 mm/s** |
| 航向 | 0.0° | 0.0° |
| 向上翘曲 | 无 | 无 |

### 迭代记录
1. force=50 + 0.8mm 钢片 → 收缩不可见，钢片太暗
2. force=200 + emission=0.7 银色钢片 → 向上翘曲 + 钢片扭转
3. force=200 + 黑色钢片 + xmat 锚定 → 翘曲仍存在
4. force=100 + 4mm 厚钢片 + 40 顶点 + 25% 重叠 → **当前版本**

### 文件
- `src/v3/worm_v4.py` — 主仿真脚本（修改）
- `src/v3/exp_runner.py` — XML 生成器（添加补光灯）
- `record/v4/videos/worm_v4_straight.mp4`

---

## [V3.5] - 2026-02-27 — 转向运动 (对角斜肌)

### 新增
- **对角斜肌 (Diagonal Steering Tendons)**: 2条/段, 连接起始板右侧→终止板左侧(左转)和反方向(右转)
  - 斜肌站点位于板边缘 (`plate_radius × 0.85 = 18.7mm`)
  - 力 200N, 跨距 37.4mm / 65mm轴向 ≈ 30°角 → 有效横向力 ~100N
- **论文参考**: Zhan et al., IJRR 2019 步态分析已记录到 `worm_gait_paper.md`

### 控制方案
- 正弦蠕动波 (1500ms周期) + 恒定左转斜肌激活 (0.5)
- 蠕动波负责前进, 斜肌负责持续横向偏置
- `bend_stiff`: 1e8 → 5e6 (增加横向柔性)
- `plate_stiff_yaw`: 100 → 5 (允许偏航旋转)
- 板间约束: `connect` (仅位置, 无方向约束)

### 性能
| 指标 | V3.4 (直线) | V3.5 (转向) |
|------|-----------|-----------|
| 前进距离 | 109.7mm/15s | **242.8mm/25s** |
| 速度 | 7.3 mm/s | **9.7 mm/s** |
| 航向偏转 | 0° | **~16°** |
| 横向位移 (头) | 0.0mm | **-23.5mm** (左) |
| 头尾横向差 | 0mm | **45.5mm** |
| Z range | 3.9mm | 7.8mm |

### 限制
- 航向偏转为**静态偏置** (~16°), 不随时间累积
- 即 "侧行" (sidewinding) 模式, 非真正的圆周运动 (circular)
- 根因: cable 弯曲刚度 (即使降到5e6) 仍阻止板间偏航旋转累积
- 真正圆周运动需要: 可变形板体 / 螺旋缆绳 / 极低弯曲刚度

### 迭代过程 (10+ 次尝试)
1. 离散步态状态机 (纸上方案) → 0° 转向 (cable太硬)
2. 非对称波相位 → 0° (纵肌力纯轴向, 无横向分量)
3. 软化板间约束 (weld→connect) → 0° (cable仍阻止)
4. 降低cable刚度至1e5 → 0° (纵肌仍纯轴向)
5. 添加偏航驱动器 (kp=200, 50Nm) → 24°但NaN不稳定
6. 偏航驱动器 (kp=20, 5Nm) → 5° 静态偏置
7. **添加对角斜肌** → 首次成功! 横向力分量
8. 恒定斜肌 + 降低cable刚度 → **16° 偏转, 9.7mm/s**

### 文件
- `src/v3/worm_5seg_v3.py` → `worm_5seg_v3_v7_circular.py`
- `record/v3/videos/worm_5seg_v3_circular.mp4`
- `record/v3/plots/circular_gait_trajectory.png`

---

## [V3.4] - 2026-02-27 — 增大纵肌力 (50N)
### 调参
- `axial_muscle_force`: 30N → 50N, 纵向压缩幅度增大

### 性能
| 指标 | V3.3 (30N) | V3.4 (50N) | V2 (38N) |
|------|-----------|-----------|----------|
| 前进距离 | 42.0mm | **109.7mm** | 132.8mm |
| 速度 | 2.8 mm/s | **7.3 mm/s** | 8.9 mm/s |
| 侧偏 | 0.0mm | 0.0mm | 0.0mm |
| Z range | 2.1mm | 3.9mm | 2.17mm |
- 无振动、无NaN

### 文件
- `src/v3/worm_5seg_v3.py` → `worm_5seg_v3_v6_force50.py`
- `record/v3/videos/worm_5seg_v3_v6_force50.mp4`

---

## [V3.3] - 2026-02-27 — 钢片无间隙 (移除 proximity hiding)
### 修正
- 移除 `reshape_to_flat_strips` 中的 proximity hiding 逻辑
- 之前: 靠近隔板 6mm 内的钢片段被设为透明 → 隔板和钢片之间有可见间隙
- 现在: 钢片完整延伸到隔板, 端点紧贴隔板内壁
- 不再需要 `plate_y_positions` 参数

### 文件
- `src/v3/worm_5seg_v3.py` → `worm_5seg_v3_v4_no_gap.py`
- `record/v3/videos/worm_5seg_v3_v4_no_gap.mp4`

---

## [V3.2] - 2026-02-27 — 钢片归位 + 环肌隐藏
### 修正
- `plate_geom_r`: 0.014 → 0.022 (= plate_radius), 钢片回到隔板内侧
- 钢片锚点 strip_circle_r=17mm < plate_geom_r=22mm → 视觉上钢片从隔板内壁伸出
- 环肌元素全部隐藏 (alpha=0):
  - 8×5 ring ball sphere geoms
  - 8×5 ring ball sites
  - 5 ring spatial tendons (红色线)
- 物理不变: 环肌仍正常工作, 仅视觉隐藏

### 性能
- Forward: **42.0mm** / 15s (≈2.8 mm/s, 略优于 V3.1)
- Lateral drift: **0.0mm**
- Z range: **2.1mm**

### 文件
- `src/v3/worm_5seg_v3.py` → `worm_5seg_v3_v3_strips_inside.py`
- `record/v3/videos/worm_5seg_v3_v3_strips_inside.mp4`

---

## [V3.1] - 2026-02-27 — 钢片扁平渲染 (Steel Strip Visual)
### 视觉改进
- 移植 V2 `reshape_to_flat_strips()` 到 V3
- Cable capsule → BOX (3mm宽 × 0.4mm厚): 扁平弹簧钢片效果
- 添加 `<asset>` 材质系统:
  - `MatSteel`: rgba=0.82/0.78/0.72, specular=0.6, shininess=0.8
  - `MatPlane`: checker 纹理地面
- 模型级重着色: 240 capsule geoms → MatSteel material
- 场景级变形: 每帧将 capsule → flat BOX + 径向朝外 + 钢色

### 板附近隐藏
- `plate_geom_r + 3mm` 范围内、距板 Y<6mm 的条带段 alpha=0 (防穿透)

### 性能 (未变)
- Forward: **41.6mm** / 15s (**2.8 mm/s**)
- Lateral drift: **0.0mm**
- Z range: **1.8mm**

### 文件
- `src/v3/worm_5seg_v3.py` → `worm_5seg_v3_v2_steel_strips.py`
- `record/v3/videos/worm_5seg_v3_v2_steel_strips.mp4`

---

## [V3.0] - 2026-02-27 — 五节段拮抗肌蠕虫 (初版)
### 架构
- 基于 V2 独立板架构 + V3 单节段验证的拮抗肌肉系统
- 6 板 × 6 DOF + 5 段 × (8 cable + 4 axial + 1 ring) = 327 bodies, 1116 DOF, 25 actuators
- `implicitfast` 积分器 + Newton/200 + dt=0.0005 (来自 P1.6 振动修复)
- Cable: bend=1e8, twist=4e7, damping=0.12

### 蠕动控制
- 纵肌: 正弦波正半周 (prograde tail→head), T=1500ms
- 环肌: 反相 (正弦波负半周) → 纵肌收缩时环肌放松(锚固), 纵肌放松时环肌收缩(滑行)

### 性能
- Forward: **41.6mm** / 15s (**2.8 mm/s**)
- Lateral drift: **0.0mm**
- Z range: **1.8mm** (稳定)
- 无振动、无NaN

### 与V2对比
| 指标 | V2 | V3 | 变化 |
|------|----|----|------|
| 前进距离 | 132.8mm | 41.6mm | -69% (待调参) |
| 侧偏 | 0.0mm | 0.0mm | 相同 |
| Z range | 2.17mm | 1.8mm | 略好 |
| 驱动器 | 20 axial | 20 axial + 5 ring | +5 ring |
| DOF | 916 | 1116 | +200 (ring balls) |

### 待优化
- 速度比V2低3x: 可能需要增大纵肌力(30→38N)或调整环肌时序
- 环肌是否真正帮助了锚固效果待验证(对比关闭环肌)

### 文件
- `src/v3/worm_5seg_v3.py` → `worm_5seg_v3_v1_initial.py`
- `record/v3/videos/worm_5seg_v3_v1_initial.mp4`
- `record/v3/plots/plate_y_trajectories.png`

---

## [P1.6] - 2026-02-27 — V3 单节段原型 (implicitfast, 当前最佳)
### 振动修复
- **根因分析**: 对比参考论文代码 `3D-Soft-Worm-Robot-Model/`
  - 参考用 RK4 + PGS/500迭代 + bend=8e8 + damping=0.12
  - 我们用 Euler + Newton/100迭代 + bend=5e7 + damping=0.5
  - Euler 1阶积分器每步引入能量误差 → 持续振荡
- **修复**: `integrator="implicitfast"` + `dt=0.0005` + `Newton/200迭代`
- 恢复 cable 刚度: bend=1e8, twist=4e7, damping=0.12

### 效果
- 纵肌 30N×4: 65→32mm (50%压缩) + 环径24→36mm (膨胀50%)
- 环肌 10N: 环径24→11mm (缩54%)
- 弯曲: ±7mm / ±10°
- **弹回**: 压缩后 32→65mm 完全恢复 (之前Euler版弹不回来)
- **无振动、无NaN**

### 参考论文关键发现
- 参考模型用 6条**螺旋缠绕**cable (非直线) + **只有环肌** (无纵肌)
- 环肌收缩 → 中国指套效应 → 轴向同时缩短
- 肌肉力仅 5-10N (我们用30N因为结构不同)

### 文件
- `src/proto/v3_single_seg.py` → `v3_single_seg_v6_implicitfast.py`
- `record/proto/videos/v3_single_seg_v6_implicitfast.mp4`

---

## [P1.1~P1.5] - 2026-02-27 — V3 单节段迭代过程
### P1.1: 初版 (Euler + 38N) — 段塌缩
- 38N×4=152N 压穿底板, top_y 从 65mm 到 -31mm
### P1.2: 降力至 8N — 效果太弱
- 纵肌仅缩短 1.2mm, 环肌减径 3.8mm
### P1.3: 20N — 环肌OK, 纵肌仍弱
### P1.4: 35N + 关节限位 — 纵肌缩4mm
### P1.5: 50N + 添加 pitch/roll/yaw 旋转关节 — 弯曲可用但振动
- 发现缺少旋转DOF导致弯曲不可能
- 移除侧向/旋转关节刚度, 让cable自行提供弹性
### P1.5b: 80N + bend=5e7 — 效果明显但振动严重
- 用户反馈: "为什么一直在振动"

---

## [P0] - 2026-02-27 — Flexcomp 圆筒壳体 (已放弃)
### 验证
- `flexcomp type="direct" dim="2"` + `elastic2d="bend"` 创建闭合圆筒壳
- 84顶点 / 144三角面 / 99 bodies / 326 DOF
- 环肌可工作: 22→2.8mm 半径收缩, 轴向压缩可见
### 放弃原因
- 用户反馈: "效果很差, 也不是我们要的钢片那种效果"
- 壳体看起来像薄膜/皮肤, 不像独立弹簧钢片

### 文件
- `src/proto/shell_cylinder_test.py`
- `record/proto/videos/shell_cylinder_test.mp4`

---

## [V2.1] - 2026-02-27
### Fixed
- Z axis cumulative drift: plates gradually rose 3-5mm over 15s
  - Root cause: `Z_stiff=0` + `Z_damp=2.0` — damping only slowed descent, never restored position
  - Fix: `Z_stiff=20 N/m` + `Z_damp=0.5` — gentle restoring force + faster settling
  - Z baseline now stable at ~1mm throughout simulation

### Performance
- Forward: 132.8mm / 15s (8.9 mm/s)
- Lateral drift: 0.0mm
- Z range: 2.17mm (independent segment height change preserved)

---

## [V2.0] - 2026-02-27
### Architecture Change — Independent Plates
- **Replaced** nested kinematic chain (V1) with flat independent plate bodies
- Each plate has 6 named joints (3 slide + 3 hinge) with per-DOF stiffness control
- Inter-plate coupling via soft weld constraints (`solref=0.010`)

### Key Parameters
- `plate_stiff_x=500` (anti-lateral-drift), `plate_stiff_yaw=100` (anti-rotation)
- `plate_stiff_y=0` (axial free), `plate_stiff_z=20` (gentle Z restore)
- `muscle_force=38N`, `plate_weld_solref="0.010 1.0"`

### Improvements over V1
- Lateral drift: 12.2mm (V1) -> 0.0mm (V2) — eliminated
- Independent height change: ~0mm (V1) -> 2.17mm (V2) — achieved
- Forward: 186.7mm (V1, force=28N) vs 132.8mm (V2, force=38N)
  - V2 at force=28N: 101.3mm (fair comparison; trade-off for independence)

### Development Notes
- Tested 10+ coupling approaches before finding optimal: soft weld + named joints
- Absolute Z stiffness >=30 N/m kills locomotion (equalizes ground contact)
- Cable-only coupling (no weld) causes 125mm lateral drift + 47mm Z oscillation
- Joint equality constraints alone give only 8-41mm forward

---

## [V1.0] - 2026-02-26
### Initial Release — Nested Kinematic Chain
- Architecture: plate0 -> plate1 -> ... -> plate5 (parent-child chain)
- 5 segments, 8 cable strips per segment, 4 axial muscle tendons per segment
- Bulge ring (8 spheres) per segment for volume conservation visual

### Performance
- Forward: 186.7mm / 15s (12.4 mm/s)
- Lateral drift: 12.2mm (after yaw stiffness fix)
- Peristaltic wave: prograde (tail->head), T=1500ms

### Known Limitations
- Kinematic chain prevents per-segment independent height change
  (child plates inherit all parent displacements)
- Cable-plate collision cannot be enabled (kills locomotion)

### Key Parameters
- `num_segments=5`, `seg_length=0.065m`
- `bend_stiff=1e8`, `twist_stiff=4e7`
- `muscle_force=28N`, `ground_friction=1.5`

---

## [Proto] - 2026-02-25
### Single Segment Prototype
- Validated spring steel strip mechanics (8 cables + weld constraints)
- Confirmed axial compression -> radial expansion behavior
- Two plate versions: fixed-base and slide-joint
- Elastic recovery measured: ~95%

### Files
- `worm_segment.py` — V1 single segment (fixed base)
- `worm_segment_v2.py` — V2 single segment (slide joint top plate)
- `spring_steel_demo.py` — Spring steel material demo
