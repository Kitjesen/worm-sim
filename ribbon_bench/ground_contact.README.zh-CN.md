# 平地法向罚接触与有历史的库仑摩擦

`ground_contact.py` 是独立 CPU float64 接触定律。它接收固定材料编号的表面采样点；平面固定，法向为世界 `+Z`。它不生成钢带表面、不定位首次碰撞事件，也不单独完成机器人动力学。

```python
trial = evaluate_contact(
    points, oldpoints, history,
    dt=dt, ground_height=height,
    normal_stiffness=kn, tangential_stiffness=kt,
    weights=weights, mu=mu,
)
```

`points`、`oldpoints` 为形状 `(P,3)` 的位置，单位 m。`oldpoints` 和 `history` 始终来自上一个已接受时间步。历史是 `{'elastic_slip_m': (P,2), 'active': (P,)}`；首次可传 `None`。返回的 `history` 为新数组，只有整体时间步接受后才能保存，Newton 与线搜索试算不会修改旧历史。采样点行号必须始终对应同一个材料点，不能按当前最近点排序重建。

`kn`、`kt` 可为正标量或 `(P,)` 数组；默认 `kt=kn`。与非负 `weights` 相乘后的单位必须为每点 N/m。面积权重单位 m² 时刚度密度为 N/m³；长度权重单位 m 时为 N/m²；已给每点刚度则设权重 1。零权重关闭该点。改变网格时保持总面积／长度权重，避免采样点变多导致总接触刚度人为增大。一个时间步内权重与刚度必须保持不变。

## 定律与返回量

间隙为 $g=z-h$，法向力 $N=k_n\max(-g,0)$，法向弹性能 $U_n=\tfrac12k_n\min(g,0)^2$。这是允许微穿透的柔顺模型：总刚度宜从目标最大支承力与目标穿透量选取 $K_n\approx F_{\max}/\delta_{\rm target}$，并通过实际工况核验。自检采用 20 µm 目标支承穿透，仅是自检参数，不是整个机器人的已验证配置。

用上一步切向弹性位移 $e_n$ 与本步切向位移增量构造 $e_{\rm tr}=e_n+\Delta x_t$。若 $k_t\|e_{\rm tr}\|\le\mu N$，则黏着，$e_{n+1}=e_{\rm tr}$、$T=-k_t e_{n+1}$；否则滑动，$e_{n+1}=(\mu N/k_t)e_{\rm tr}/\|e_{\rm tr}\|$、$T=-\mu N e_{\rm tr}/\|e_{\rm tr}\|$。首版黏着／滑动共用一个摩擦系数。

这种黏着允许有限弹性预滑移；停止运动后保留切向支承力，不是纯黏性摩擦，也不是严格零位移的刚性静摩擦约束。`dt` 必须为正，但定律使用位移增量，具有率无关性；动力学残差中的时间尺度由调用方处理。新接触从零弹性历史起算，使用整步切向增量；首碰步需要时间步加密或积分层事件细分，避免在较长跨越步中把空中位移误计为接触位移。若一步内离地后重新接触而终点仍在接触，终点定律不能自行发现该事件。

返回 `force (P,3)` 是地面对采样点的物理力，单位 N；`jacobian (P,3,3)` 是该力对当前采样点位置的导数，单位 N/m。滑动分支包含 $\partial T/\partial z=\mu k_n u$，其中 $u=e_{\rm tr}/\|e_{\rm tr}\|$；对应 $\partial N/\partial x_t=0$，所以切线一般不对称。分离、黏着、滑动各开分支内是解析切线；边界处返回所选分支切线，不声称全程光滑。辅助字段为 `gap_m`、`normal_force_n`、`status` 和 `plastic_increment_m`。

`normal_energy_j` 与 `tangential_energy_j` 是当前储能总量；`friction_dissipation_j` 是本步 $\sum\mu N\|e_{\rm tr}-e_{n+1}\|$ 塑性滑移耗散。离地清空弹性历史；因此单独输出 `history_reset_dissipation_j`，记录被清空的旧切向储能，不能漏记为能量凭空消失，也不要重复加到本步塑性耗散中。端点隐式离散还可能有数值耗散，这些量不保证形成精确连续时间功平衡。

## 接入 Newton

物理广义接触力是 $Q_c=J_c^T f_c$，其中 $J_c=\partial x_c/\partial q$ 必须包含钢带宽厚偏置与材料方向的导数。若残差写成惯性项加内力势能梯度减外力，则接触贡献为 $-Q_c$。其一致切线为

$$
-\left(J_c^T\frac{\partial f_c}{\partial x_c}J_c+
\sum_{i,\alpha}f_{i\alpha}\nabla_q^2x_{i\alpha}\right).
$$

只用 `J.T @ jacobian @ J` 会漏掉非线性表面几何项。此模块只返回点坐标层面的力和切线，几何导数由积分器／钢带适配层负责。摩擦随历史与法向载荷变化，一般不存在可用于全系统线搜索的单一保守势能；使用带物理单位缩放的残差 merit、非对称线性求解及接触分支重算，不把此 Jacobian 强行对称化。每次试算均固定同一 `oldpoints/history`，最终残差满足后再接受历史。

## 自检与范围

```powershell
.\.venv\Scripts\python.exe ground_contact.py --self-check
```

自检覆盖四采样点块体的重力支承代数平衡、目标穿透、黏着阈值与停留保持力、滑动方向／摩擦圆锥、端点功与储能／耗散关系、离地力归零与重置、零摩擦／零权重，以及各平滑分支包含法向耦合列的中心差分。它不替代实际时间积分的落地、接触切换、时间步收敛、钢带变形、刚板运动或爬行验证。

Sano 模型表示钢带整体弯扭，不解析接触处局部压痕。采样表面也仍需进行几何覆盖与网格检查，不能由此宣称钢带自接触或复杂障碍碰撞已实现。
