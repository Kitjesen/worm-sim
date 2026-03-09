# Circular Locomotion — 5-Segment Worm MuJoCo Simulation

## 1. 目标

在 MuJoCo 中实现 5 段蠕虫的**圆周运动** (circular locomotion)，对应 Zhan et al. (IJRR 2019) 论文中 n2 != n3 的圆弧步态模式。蠕虫已能直线蠕动前进（rectilinear），需要增加转向能力使 COM 轨迹形成闭合圆。

## 2. 模型架构

```
plate0(tail) ── plate1 ── plate2 ── plate3 ── plate4 ── plate5(head)
   [蓝]                                                    [红]
```

- **6 个独立板** (plate)，每个有 6DOF 弹簧关节 (x/y/z/pitch/roll/yaw)
- 板间通过 `<connect>` 等式约束连接
- 4 条轴向肌肉 tendon / 段 (上/下/左/右)
- 2 条对角转向 tendon / 段 (左斜/右斜)
- tendon 弹簧提供被动弹性 (k=5000 N/m, c=10 N·s/m)
- 无 cable 模式 (`no_cables=1`)：纯板+tendon，~36 DOF，单步 <0.5ms

### 步态

1-anchor 蠕动波: `gait_s0 = [2, 0, 0, 0, 1]` (尾→头)

| 状态 | 含义 | 肌肉激活 |
|------|------|---------|
| 0 | 放松 (extend) | 全 OFF |
| 1 | 收缩 (anchor) | 全 ON |
| 2 | 左弯 (symmetric) | 50% ON |

每步 0.5s，状态向尾部后传 1 段，5 步一周期 (2.5s)。

## 3. 失败的尝试

跨 5 个 session 尝试了以下方法，**全部失败**（轨迹 aspect ratio ~0.02，即纯直线）：

| # | 方法 | 原理 | 失败原因 |
|---|------|------|---------|
| 1 | 对角斜肌差分激活 | State 2/3 左右差分纵肌 | 纵肌力纯轴向，无横向分量 |
| 2 | 扭转摩擦 (condim=4) | 锚固板旋转摩擦锁定 heading | 摩擦不足以产生净转矩 |
| 3 | 弹簧偏航 PD 控制器 | 驱动板间角度到 ±18° | **被 plate_stiff_x=500 阻止** |
| 4 | 累积偏航控制器 | 跟踪全局 heading 目标 | **被 plate_stiff_x=500 阻止** |
| 5 | 体曲率 (body_curvature) | 永久弯曲体形 | **被 plate_stiff_x=500 阻止** |
| 6 | 锚固阻尼 + 偏航耦合 | 锚固板高阻尼 + 等式约束 | **被 plate_stiff_x=500 阻止** |

## 4. 根因发现: plate_stiff_x = 500

### 现象

所有转向实验中，横向位移严格限制在 <1mm，无论施加多大转向力。

### 诊断

每个板的 X 轴 slide joint 有默认弹簧刚度 `plate_stiff_x=500` (N/m)：

```xml
<joint name="p0_x" type="slide" axis="1 0 0" stiffness="500" damping="0.5"/>
```

横向平衡位移 = F / k：
```
0.1N / 500 N/m = 0.0002m = 0.2mm  ← 精确匹配 S1 实验结果
```

### 修复

```python
plate_stiff_x = 0    # 从 500 改为 0
plate_stiff_y = 0    # 已经是 0（前进方向）
```

**设 plate_stiff_x=0 后，所有横向运动立即解锁。**

### 为什么 Y 方向正常而 X 方向不正常？

- `plate_stiff_y=0` 从最早就设为 0（前进方向需要自由滑动）
- `plate_stiff_x=500` 是默认值，从未被覆盖
- 直线蠕动只需要 Y 方向自由，所以一直正常工作
- 圆弧运动需要 X 方向也自由，但从未有人检查过这个参数

## 5. 解决方案: Body-Frame 横向力

### 5.1 原理

在蠕虫 body heading 的垂直方向施加恒力。力随蠕虫转向而旋转，始终垂直于运动方向→产生向心力→圆周运动。

```python
# 计算 body heading (头→尾方向)
heading = atan2(head_x - tail_x, head_y - tail_y)

# 垂直于 heading 的力（左向）
fx = -steer_force * cos(heading)
fy =  steer_force * sin(heading)
```

### 5.2 为什么世界坐标系力不行？

恒定方向的力（如始终 -X）只能产生**抛物线**轨迹，不是圆。当蠕虫转过 90° 后，力方向变成了前进/后退方向，不再提供横向推力。

Body-frame 力始终垂直于运动方向 → 恒定向心加速度 → 圆周运动 (F = mv²/r)。

### 5.3 施力位置比较

| 模式 | 施力位置 | 结果 |
|------|---------|------|
| `body_all` | 所有 6 个板 | R~36m, aspect=0.46（均匀推力≈平移，扭矩小） |
| `body_extend` | 伸展态板 | 中等效果 |
| `body_anchor` | 锚固态板 | 中等效果 |
| **`body_head`** | **仅头板** | **R~1.4m, aspect=0.96（力臂最长→扭矩最大）** |

`body_head` 最优原因：力只作用在头板，距离尾部锚点最远（~325mm），产生最大旋转力矩。其他模式力均匀分布，旋转力矩被抵消。

### 5.4 力的大小扫参

对 `body_head` 模式做了完整力扫参：

| Exp | 力 (N) | Aspect | 周期 (s/rot) | 半径 (mm) | Closure | 旋转数 |
|-----|---------|--------|-------------|-----------|---------|--------|
| S21 | 0.10 | 0.34 | 1655 | 3369 | 1405mm | 1.2/2000s |
| S22 | 0.15 | 0.64 | 1111 | ~4300 | — | 0.9/1000s |
| S24 | 0.25 | 0.58 | 661 | 1595 | 1541mm | 1.5/1000s |
| **S23** | **0.30** | **0.87** | **555** | **1458** | **235mm** | **3.6/2000s** |
| S25 | 0.40 | 0.89 | 413 | 1430 | 1216mm | 2.4/1000s |
| S18 | 2.00 | 0.52 | ~72 | 46 | — | 9.8/700s |

**Sweet spot: 0.3-0.4N**
- <0.25N: 力不够，轨迹椭圆化 (aspect<0.6)
- 0.3N: 最圆 (aspect=0.87)，R=1.5m ≈ 4.5 body lengths
- 0.4N: 也很圆 (aspect=0.89)，更快 (413s/rot vs 555s/rot)
- >0.5N: 力太大，开始原地旋转

## 6. 最优配置

```bash
python src/v3/exp_runner.py S23 --params \
    gait_s0=2,0,0,0,1 \
    state2_mode=symmetric \
    steer_in_state2=0 \
    no_cables=1 \
    constraint_type=none \
    plate_stiff_y=0 \
    plate_stiff_x=0 \
    tendon_stiffness=5000 \
    tendon_damping=10 \
    yaw_mode=none \
    steer_force=0.3 \
    steer_mode=body_head \
    sim_time=2000 \
    save_traj=1
```

### 结果 (S23, 2000s, 3.6 圈)

| 指标 | 值 |
|------|-----|
| Aspect ratio | 0.87 (接近完美圆=1.0) |
| 转弯半径 | 1458mm (~1.5m, ~4.5 body lengths) |
| 周期 | 555 s/rotation |
| Heading rate | 0.65 deg/s (完美线性) |
| Closure (最佳圈) | 235mm (2.6% of circumference) |
| 体完整性 | ht=312.8±1.2mm, bend p95=1.4° |
| 稳定性 | 3.6 圈，轨迹逐圈改善 |
| 仿真速度 | 475s wall time / 2000s sim time |

## 7. 实现代码

### 核心转向逻辑 (`exp_runner.py` L696-750)

```python
steer_f = float(P.get('steer_force', 0))
if steer_f != 0:
    steer_m = P.get('steer_mode', 'extend')
    body_frame = steer_m.startswith("body")
    if body_frame:
        # 计算 body heading
        _hx, _hy = d.xpos[head_id, 0], d.xpos[head_id, 1]
        _tx, _ty = d.xpos[tail_id, 0], d.xpos[tail_id, 1]
        _heading = math.atan2(_hx - _tx, _hy - _ty)
        # 垂直于 heading 的力
        fx = -steer_f * math.cos(_heading)
        fy =  steer_f * math.sin(_heading)
    else:
        fx, fy = -steer_f, 0.0

    effective_mode = steer_m.replace("body_", "") if body_frame else steer_m
    if effective_mode == "head":
        d.xfrc_applied[pids[num_plates - 1], 0] += fx
        d.xfrc_applied[pids[num_plates - 1], 1] += fy
```

### 视频录制

```bash
python src/v3/record_video.py --camera top --duration 30 --output record/v3/videos/circular_top.mp4
python src/v3/record_video.py --camera side --duration 30 --output record/v3/videos/circular_side.mp4
```

## 8. 文件索引

| 文件 | 说明 |
|------|------|
| `src/v3/exp_runner.py` | 参数化实验运行器（headless，快速） |
| `src/v3/record_video.py` | 视频录制（MuJoCo offscreen rendering） |
| `src/v3/plot_trajectory.py` | 轨迹可视化 + 圆度指标 |
| `record/v3/trajectories/S23_traj.npz` | 最优配置 3.6 圈轨迹数据 |
| `record/v3/plots/S23_trajectory.png` | 最优配置轨迹图 |
| `record/v3/videos/` | 运动视频 |

## 9. 物理解释

蠕虫圆周运动的物理机制：

1. **蠕动波提供前进推力**: 1-anchor 步态 [2,0,0,0,1] 通过逆行波产生向前位移（~7m/25s 在直线模式下）
2. **Head 横向力提供向心力**: 0.3N 力作用在头板上，距离尾部锚点 ~325mm，产生旋转力矩
3. **Body-frame 旋转**: 力方向随 heading 旋转，始终垂直于运动方向
4. **蠕动 + 向心力 = 圆周**: 前进速度 v + 横向加速度 a = 半径 r = v²/a

力的大小决定半径：
- F = m * v² / r → r = m * v² / F
- F 太小 → r 太大 → 轨迹展开
- F 太大 → r → 0 → 原地旋转（蠕动波被覆盖）
- Sweet spot (0.3N): r ≈ 1.5m ≈ 4.5 body lengths

## 10. 局限性与后续

### 当前局限
- 头板横向力是**外部施加**的 (`xfrc_applied`)，不是由内部肌肉产生
- 真实蠕虫通过差分纵肌（左侧收缩 > 右侧）产生转向力矩
- 当前模型中纵肌力方向与 Y 轴平行，无法直接产生横向力

### 可能的物理实现
1. **非对称锚固摩擦**: 锚固态一侧接触面积/摩擦更大
2. **头部偏转机构**: 类似舵的头板偏转装置
3. **体曲率 + 差分蠕动**: 弯曲体形 + 不同段不同速度的蠕动波
4. **对角肌肉优化**: 增大斜肌角度（当前 ~30°），增加横向力分量
