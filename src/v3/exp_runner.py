"""
Parameter Space Search for Circular Gait Locomotion
====================================================
Runs headless (no video/plot) for speed. Reports key metrics only.
Usage: python exp_runner.py <exp_id> [--params key=val ...]
"""
import mujoco
import numpy as np
import time
import math
import os
import sys
import json

# ======================== Default Parameters ========================
DEFAULTS = dict(
    num_segments=5,
    seg_length=0.065,
    plate_radius=0.022,
    plate_geom_r=0.022,
    plate_thickness=0.003,
    strip_circle_r=0.017,
    num_strips=8,
    num_verts=7,
    strip_r=0.002,
    bow_amount=0.007,
    bend_stiff=1e8,
    twist_stiff=2e6,
    axial_muscle_force=60,
    ring_muscle_force=12,
    ground_friction=1.5,
    steer_muscle_force=200,
    plate_stiff_x=500.0,
    plate_stiff_y=0.0,
    plate_stiff_z=20.0,
    plate_stiff_pitch=5.0,
    plate_stiff_roll=5.0,
    plate_stiff_yaw=0.0,
    plate_damp_x=0.5,
    plate_damp_y=3.0,
    plate_damp_z=0.5,
    plate_damp_pitch=0.5,
    plate_damp_roll=0.5,
    plate_damp_yaw=0.5,
    plate_weld_solref="0.010 1.0",
    plate_weld_solimp="0.90 0.95 0.002 0.5 2",
    # Constraint type: "connect" or "weld" or "none"
    constraint_type="connect",
    # Gait
    step_duration=0.5,
    # State 2 axial level (0-1, used in symmetric mode)
    state2_axial=0.5,
    # State 2 axial mode: "symmetric" (all 4 at state2_axial) or "differential" (left/right only)
    state2_mode="symmetric",
    # Whether to activate diagonal steering tendons in State 2/3 (1=yes, 0=no)
    steer_in_state2=1,
    # Direct yaw torque for State 2/3 (N·m, applied to plates via xfrc_applied, 0=off)
    yaw_torque=0.0,
    # Cable weld solref (controls cable-plate constraint stiffness)
    cable_weld_solref="0.002 1",
    # Cable-plate constraint type: "weld" (6DOF, locks orientation) or "connect" (3DOF, position only)
    cable_constraint="weld",
    # Yaw spring mode: inter-plate PD controller for yaw angle
    # "spring" = read actual plate yaw, apply proportional+derivative torque to reach desired angle
    yaw_mode="none",        # "none", "torque" (xfrc_applied), "spring" (inter-plate PD)
    yaw_angle=18.0,         # desired inter-plate bend angle for State 2/3 (degrees)
    yaw_kp=50.0,            # spring constant (N·m/rad)
    yaw_kd=2.0,             # damping constant (N·m·s/rad)
    # Anchor yaw damping: high damping on anchored (State 1) plates to prevent yaw rotation
    # This creates asymmetric heading accumulation: anchored plates grip ground, bending plates rotate
    anchor_yaw_damp=0.0,    # N·m·s/rad, 0=off
    # Torsional friction: enables condim=4 contacts with spinning friction (critical for heading accumulation)
    torsional_friction=0.0,  # torsional friction coefficient (0=off, condim=3 only)
    # Anchor downforce: push anchored plates down to increase ground friction for heading accumulation
    anchor_downforce=0.0,    # N, applied as -Z force to both plates of State 1 segments
    # Yaw coupling: dynamic equality constraints between consecutive plate yaw joints
    # State 0/1: ACTIVE (plates yaw-coupled, heading transmitted through chain)
    # State 2/3: INACTIVE (plates free to bend)
    yaw_coupling=0,          # 0=off, 1=enable dynamic yaw coupling
    # No-cable mode: skip cable bodies and use plates only (for gait validation)
    no_cables=0,
    # Inter-plate constraint mode for no-cable mode: "weld" or "connect"
    plate_constraint="connect",
    # Inter-plate weld solref for angular coupling (only used when plate_constraint="weld")
    plate_angular_solref="0.05 1",
    # Initial gait state vector (tail→head), comma-separated
    gait_s0="2,0,0,1,1",
    # Body curvature: constant yaw offset per segment (degrees)
    # Creates a permanently curved (banana-shaped) body for circular locomotion.
    # Positive = curve to the left, negative = curve to the right.
    body_curvature=0.0,
    # Lateral steering force (N): applied in -X direction to extending plates (state 0)
    # Simulates differential longitudinal muscle thrust for circular locomotion.
    # Positive = push left (curve left), magnitude in Newtons.
    steer_force=0.0,
    # Steering mode: which gait states receive lateral force
    # "extend" = state 0 only, "all" = all plates always, "anchor" = state 1 only
    steer_mode="extend",
    # Sim
    sim_time=15.0,
    settle_time=1.0,
)


def build_model_xml(exp_id, params):
    """Build MuJoCo XML for the worm model.

    Returns: (xml_str, merged_params_dict)
    """
    P = {**DEFAULTS, **params}

    num_segments = P['num_segments']
    seg_length = P['seg_length']
    plate_radius = P['plate_radius']
    strip_circle_r = P['strip_circle_r']
    num_strips = P['num_strips']
    num_verts = P['num_verts']
    strip_r = P['strip_r']
    bow_amount = P['bow_amount']
    z_center = plate_radius + 0.001
    num_plates = num_segments + 1
    total_length = num_segments * seg_length
    strip_angles = [2 * math.pi * i / num_strips for i in range(num_strips)]
    mid_vert_idx = num_verts // 2
    torsion_f = float(P.get('torsional_friction', 0))
    no_cables = int(P.get('no_cables', 0))

    def strip_verts(angle, seg_idx):
        y_start = seg_idx * seg_length
        verts = []
        for k in range(num_verts):
            t = k / (num_verts - 1)
            y = y_start + t * seg_length
            bow_r = bow_amount * 4.0 * t * (1.0 - t)
            r = strip_circle_r + bow_r
            x = r * math.cos(angle)
            z = z_center + r * math.sin(angle)
            verts.append(f"{x:.6f} {y:.6f} {z:.6f}")
        return "  ".join(verts)

    # --- XML Generation ---
    plates_xml = ""
    for p in range(num_plates):
        y_world = p * seg_length
        plates_xml += f'    <body name="plate{p}" pos="0 {y_world:.5f} 0">\n'
        plates_xml += f'      <joint name="p{p}_x" type="slide" axis="1 0 0" stiffness="{P["plate_stiff_x"]}" damping="{P["plate_damp_x"]}"/>\n'
        plates_xml += f'      <joint name="p{p}_y" type="slide" axis="0 1 0" stiffness="{P["plate_stiff_y"]}" damping="{P["plate_damp_y"]}"/>\n'
        plates_xml += f'      <joint name="p{p}_z" type="slide" axis="0 0 1" stiffness="{P["plate_stiff_z"]}" damping="{P["plate_damp_z"]}"/>\n'
        plates_xml += f'      <joint name="p{p}_pitch" type="hinge" axis="1 0 0" stiffness="{P["plate_stiff_pitch"]}" damping="{P["plate_damp_pitch"]}"/>\n'
        plates_xml += f'      <joint name="p{p}_roll" type="hinge" axis="0 1 0" stiffness="{P["plate_stiff_roll"]}" damping="{P["plate_damp_roll"]}"/>\n'
        plates_xml += f'      <joint name="p{p}_yaw" type="hinge" axis="0 0 1" stiffness="{P["plate_stiff_yaw"]}" damping="{P["plate_damp_yaw"]}"/>\n'
        plates_xml += f'      <geom type="cylinder" size="{P["plate_geom_r"]} {P["plate_thickness"]}" pos="0 0 {z_center}"\n'
        plates_xml += f'            euler="90 0 0" rgba="0.45 0.45 0.50 0.9" mass="0.02" contype="2" conaffinity="2"/>\n'
        # Ground contact foot geoms (provide friction for locomotion)
        if torsion_f > 0:
            # 3 contact spheres spread along X at plate bottom for effective torsional resistance
            # MuJoCo friction format: "slide spin roll" — torsion_f goes in spin (2nd) position
            foot_friction = f'{P["ground_friction"]} {torsion_f} 0.001'
            foot_spread = plate_radius * 0.9  # wide spread for torsional resistance
            for fi, fx in enumerate([-foot_spread, 0.0, foot_spread]):
                plates_xml += f'      <geom name="foot{p}_{fi}" type="sphere" size="0.003" pos="{fx:.5f} 0 0.003"\n'
                plates_xml += f'            rgba="0.3 0.3 0.3 0.5" mass="0.001" friction="{foot_friction}" condim="4" contype="1" conaffinity="1"/>\n'
        else:
            plates_xml += f'      <geom name="foot{p}" type="capsule" size="0.004" fromto="-0.016 0 0.002 0.016 0 0.002"\n'
            plates_xml += f'            rgba="0.85 0.55 0.20 0.9" mass="0.002" friction="{P["ground_friction"]}" contype="1" conaffinity="1"/>\n'
        for mi in range(4):
            sa = strip_angles[mi * 2]
            sx = strip_circle_r * 0.7 * math.cos(sa)
            sz = z_center + strip_circle_r * 0.7 * math.sin(sa)
            plates_xml += f'      <site name="p{p}_s{mi}" pos="{sx:.5f} 0 {sz:.5f}" size="0.0015"/>\n'
        steer_r = plate_radius * 0.85
        steer_z = z_center
        plates_xml += f'      <site name="p{p}_stL" pos="{-steer_r:.5f} 0 {steer_z:.5f}" size="0.0015"/>\n'
        plates_xml += f'      <site name="p{p}_stR" pos="{steer_r:.5f} 0 {steer_z:.5f}" size="0.0015"/>\n'
        # Center site for center-routed tendons (no lateral offset → no yaw coupling)
        plates_xml += f'      <site name="p{p}_center" pos="0 0 {z_center:.5f}" size="0.0015"/>\n'
        plates_xml += f'    </body>\n'

    cables_xml = ""
    for seg in range(num_segments):
        for si, angle in enumerate(strip_angles):
            v = strip_verts(angle, seg)
            prefix = f"c{seg}s{si}"
            cables_xml += f"""
    <body>
      <freejoint/>
      <composite type="cable" prefix="{prefix}" initial="none" vertex="{v}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="bend" value="{P['bend_stiff']}"/>
          <config key="twist" value="{P['twist_stiff']}"/>
          <config key="vmax" value="2"/>
        </plugin>
        <joint armature="0.01" damping="0.25" kind="main"/>
        <geom type="capsule" size="{strip_r}" density="3500"
              friction="{P['ground_friction']}" contype="1" conaffinity="1"/>
      </composite>
    </body>"""

    ring_balls_xml = ""
    for seg in range(num_segments):
        mid_y = seg * seg_length + seg_length / 2
        for si, angle in enumerate(strip_angles):
            mid_r = strip_circle_r + bow_amount
            mx = mid_r * math.cos(angle)
            mz = z_center + mid_r * math.sin(angle)
            ring_balls_xml += f'    <body name="rb{seg}_{si}" pos="{mx:.5f} {mid_y:.5f} {mz:.5f}">\n'
            ring_balls_xml += f'      <freejoint/>\n'
            ring_balls_xml += f'      <geom type="sphere" size="0.002" mass="0.0005" contype="0" conaffinity="0"/>\n'
            ring_balls_xml += f'      <site name="rs{seg}_{si}" pos="0 0 0" size="0.0018"/>\n'
            ring_balls_xml += f'    </body>\n'

    excludes_xml = ""
    for seg in range(num_segments):
        excludes_xml += f'    <exclude body1="plate{seg}" body2="plate{seg+1}"/>\n'

    cable_welds_xml = ""
    cable_ct = P['cable_constraint']
    cable_solref = P['cable_weld_solref']
    for seg in range(num_segments):
        for si in range(num_strips):
            prefix = f"c{seg}s{si}"
            if cable_ct == "connect":
                # Position-only (3DOF) — cable endpoints can rotate freely
                cable_welds_xml += f'    <connect body1="plate{seg}" body2="{prefix}B_first" anchor="0 0 0" solref="{cable_solref}"/>\n'
                cable_welds_xml += f'    <connect body1="plate{seg+1}" body2="{prefix}B_last" anchor="0 0 0" solref="{cable_solref}"/>\n'
            else:
                # Weld (6DOF) — locks cable endpoint orientation to plate
                cable_welds_xml += f'    <weld body1="plate{seg}" body2="{prefix}B_first" solref="{cable_solref}"/>\n'
                cable_welds_xml += f'    <weld body1="plate{seg+1}" body2="{prefix}B_last" solref="{cable_solref}"/>\n'

    plate_welds_xml = ""
    ct = P['constraint_type']
    if ct == "connect":
        for seg in range(num_segments):
            plate_welds_xml += (
                f'    <connect body1="plate{seg}" body2="plate{seg+1}"\n'
                f'          anchor="0 {seg_length:.5f} 0"\n'
                f'          solref="{P["plate_weld_solref"]}" solimp="{P["plate_weld_solimp"]}"/>\n'
            )
    elif ct == "weld":
        for seg in range(num_segments):
            plate_welds_xml += (
                f'    <weld body1="plate{seg}" body2="plate{seg+1}"\n'
                f'          solref="{P["plate_weld_solref"]}" solimp="{P["plate_weld_solimp"]}"/>\n'
            )
    # ct == "none" → no inter-plate constraints

    ring_connects_xml = ""
    for seg in range(num_segments):
        for si in range(num_strips):
            prefix = f"c{seg}s{si}"
            ring_connects_xml += f'    <connect body1="rb{seg}_{si}" body2="{prefix}B_{mid_vert_idx}" anchor="0 0 0" solref="0.003 1"/>\n'

    axial_tendons_xml = ""
    axial_muscles_xml = ""
    tendon_spring_stiff = float(P.get('tendon_stiffness', 0))
    tendon_spring_damp = float(P.get('tendon_damping', 0))
    tendon_routing = str(P.get('tendon_routing', 'offset'))  # "offset" or "center"
    for seg in range(num_segments):
        for mi in range(4):
            tname = f"at{seg}_{mi}"
            mname = f"am{seg}_{mi}"
            # Add tendon spring in no-cable mode for inter-plate elastic coupling
            spring_attrs = ""
            if no_cables and tendon_spring_stiff > 0 and tendon_routing == "offset":
                # Offset routing: springs on muscle tendons (couples axial + yaw)
                spring_attrs = f' stiffness="{tendon_spring_stiff}" damping="{tendon_spring_damp}" springlength="{seg_length:.5f}"'
            axial_tendons_xml += f'    <spatial name="{tname}" width="0.0012"{spring_attrs}>\n'
            axial_tendons_xml += f'      <site site="p{seg+1}_s{mi}"/>\n'
            axial_tendons_xml += f'      <site site="p{seg}_s{mi}"/>\n'
            axial_tendons_xml += f'    </spatial>\n'
            # End segments get 30% stronger muscles to compensate boundary effect
            is_end_seg = (seg == 0 or seg == num_segments - 1)
            seg_force = P["axial_muscle_force"] * 1.3 if is_end_seg else P["axial_muscle_force"]
            axial_muscles_xml += f'    <muscle class="muscle" name="{mname}" tendon="{tname}" force="{seg_force:.0f}" lengthrange="0.03 0.08"/>\n'
        # Center routing: separate spring tendon through plate center (axial only, no yaw coupling)
        if no_cables and tendon_spring_stiff > 0 and tendon_routing == "center":
            ctname = f"ct{seg}"
            spring_attrs = f' stiffness="{tendon_spring_stiff}" damping="{tendon_spring_damp}" springlength="{seg_length:.5f}"'
            axial_tendons_xml += f'    <spatial name="{ctname}" width="0.0015"{spring_attrs}>\n'
            axial_tendons_xml += f'      <site site="p{seg+1}_center"/>\n'
            axial_tendons_xml += f'      <site site="p{seg}_center"/>\n'
            axial_tendons_xml += f'    </spatial>\n'

    ring_tendons_xml = ""
    ring_muscles_xml = ""
    for seg in range(num_segments):
        tname = f"rt{seg}"
        ring_tendons_xml += f'    <spatial name="{tname}" width="0.0015">\n'
        for si in range(num_strips):
            ring_tendons_xml += f'      <site site="rs{seg}_{si}"/>\n'
        ring_tendons_xml += f'      <site site="rs{seg}_0"/>\n'
        ring_tendons_xml += f'    </spatial>\n'
        ring_muscles_xml += f'    <muscle class="muscle" name="rm{seg}" tendon="{tname}" force="{P["ring_muscle_force"]}" lengthrange="0.05 0.20"/>\n'

    steer_tendons_xml = ""
    steer_muscles_xml = ""
    for seg in range(num_segments):
        for direction, tag, s1, s2 in [("L", f"stL{seg}", "stR", "stL"), ("R", f"stR{seg}", "stL", "stR")]:
            mname = f"sm{direction}{seg}"
            steer_tendons_xml += f'    <spatial name="{tag}" width="0.0012">\n'
            steer_tendons_xml += f'      <site site="p{seg}_{s1}"/>\n'
            steer_tendons_xml += f'      <site site="p{seg+1}_{s2}"/>\n'
            steer_tendons_xml += f'    </spatial>\n'
            steer_muscles_xml += f'    <muscle class="muscle" name="{mname}" tendon="{tag}" force="{P["steer_muscle_force"]}" lengthrange="0.03 0.10"/>\n'
    num_steer = num_segments * 2

    if no_cables:
        # Simplified model: plates only, no cables/ring balls
        # Use inter-plate weld for angular coupling
        pc = P.get('plate_constraint', 'connect')
        yaw_coupling = int(P.get('yaw_coupling', 0))
        plate_eq_xml = ""
        for seg in range(num_segments):
            if pc == "weld":
                plate_eq_xml += (
                    f'    <weld body1="plate{seg}" body2="plate{seg+1}"\n'
                    f'          solref="{P.get("plate_angular_solref", "0.05 1")}" solimp="{P["plate_weld_solimp"]}"/>\n'
                )
            else:
                plate_eq_xml += (
                    f'    <connect body1="plate{seg}" body2="plate{seg+1}"\n'
                    f'          anchor="0 {seg_length:.5f} 0"\n'
                    f'          solref="{P["plate_weld_solref"]}" solimp="{P["plate_weld_solimp"]}"/>\n'
                )
        # Dynamic yaw coupling: joint equality between consecutive plate yaw joints
        # These are toggled per-step: active in State 0/1, inactive in State 2/3
        yaw_eq_xml = ""
        if yaw_coupling:
            for seg in range(num_segments):
                yaw_eq_xml += (
                    f'    <joint name="yaw_eq{seg}" joint1="p{seg}_yaw" joint2="p{seg+1}_yaw"\n'
                    f'          polycoef="0 1 0 0 0" solref="0.02 1" solimp="0.9 0.95 0.001" active="false"/>\n'
                )
        full_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="worm_exp_{exp_id}">
  <size memory="1G"/>
  <compiler autolimits="true"><lengthrange useexisting="true" tolrange="0.5"/></compiler>
  <option timestep="0.0005" gravity="0 0 -9.81" solver="Newton" iterations="100" tolerance="1e-8" integrator="implicitfast"/>
  <default>
    <geom solimp=".95 .99 .0001" solref="0.005 1"/>
    <site size="0.002"/>
    <default class="muscle"><muscle ctrllimited="true" ctrlrange="0 1"/></default>
  </default>
  <worldbody>
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <geom type="plane" size="2 2 0.01" condim="{'4' if torsion_f > 0 else '3'}" friction="{P['ground_friction']} {torsion_f} 0.001" contype="3" conaffinity="3"/>
{plates_xml}
  </worldbody>
  <contact>
{excludes_xml}
  </contact>
  <equality>
{plate_eq_xml}{yaw_eq_xml}
  </equality>
  <tendon>
{axial_tendons_xml}
{steer_tendons_xml}
  </tendon>
  <actuator>
{axial_muscles_xml}
{steer_muscles_xml}
  </actuator>
</mujoco>"""
    else:
        full_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="worm_exp_{exp_id}">
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <size memory="2G"/>
  <compiler autolimits="true"><lengthrange useexisting="true" tolrange="0.5"/></compiler>
  <option timestep="0.0005" gravity="0 0 -9.81" solver="Newton" iterations="200" tolerance="1e-8" integrator="implicitfast"/>
  <default>
    <geom solimp=".95 .99 .0001" solref="0.005 1"/>
    <site size="0.002"/>
    <default class="muscle"><muscle ctrllimited="true" ctrlrange="0 1"/></default>
  </default>
  <worldbody>
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <geom type="plane" size="2 2 0.01" condim="{'4' if torsion_f > 0 else '3'}" friction="{P['ground_friction']} {torsion_f} 0.001" contype="3" conaffinity="3"/>
{plates_xml}
{cables_xml}
{ring_balls_xml}
  </worldbody>
  <contact>
{excludes_xml}
  </contact>
  <equality>
{cable_welds_xml}
{plate_welds_xml}
{ring_connects_xml}
  </equality>
  <tendon>
{axial_tendons_xml}
{ring_tendons_xml}
{steer_tendons_xml}
  </tendon>
  <actuator>
{axial_muscles_xml}
{ring_muscles_xml}
{steer_muscles_xml}
  </actuator>
</mujoco>"""

    return full_xml, P


def run_experiment(exp_id, params, save_traj=False):
    full_xml, P = build_model_xml(exp_id, params)

    num_segments = P['num_segments']
    seg_length = P['seg_length']
    num_plates = num_segments + 1
    no_cables = int(P.get('no_cables', 0))

    # --- Save & Load ---
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    bin_dir = os.path.join(project_root, "bin", "v3", "experiments")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, f"exp_{exp_id}.xml")
    with open(xml_path, "w") as f:
        f.write(full_xml)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    num_axial = num_segments * 4
    num_ring = 0 if no_cables else num_segments
    pids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
    head_id = pids[-1]
    tail_id = pids[0]

    # Find yaw joint DOF indices for spring controller
    yaw_mode = str(P.get('yaw_mode', 'none'))
    yaw_dof_ids = []
    yaw_jnt_ids = []
    for p in range(num_plates):
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{p}_yaw")
        yaw_jnt_ids.append(jnt_id)
        yaw_dof_ids.append(m.jnt_dofadr[jnt_id])
    # Find yaw equality constraint IDs for dynamic coupling
    yaw_eq_ids = []
    if int(P.get('yaw_coupling', 0)):
        for seg in range(num_segments):
            eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"yaw_eq{seg}")
            yaw_eq_ids.append(eq_id)

    yaw_angle_rad = math.radians(float(P.get('yaw_angle', 18.0)))
    yaw_kp = float(P.get('yaw_kp', 50.0))
    yaw_kd = float(P.get('yaw_kd', 2.0))

    # Settle
    for _ in range(4000):
        mujoco.mj_step(m, d)
    if np.any(np.isnan(d.qpos)):
        return {"exp_id": exp_id, "error": "NaN after settling", "params": params}

    # --- Gait ---
    gait_str = str(P['gait_s0'])
    s0_gait = [int(x) for x in gait_str.split(',')]
    assert len(s0_gait) == num_segments, f"gait_s0 length {len(s0_gait)} != num_segments {num_segments}"
    TN = num_segments
    nP = 1
    step_dur = P['step_duration']
    state2_axial = P['state2_axial']
    state2_mode = str(P['state2_mode'])
    steer_in_s2 = int(P['steer_in_state2'])

    def get_states(t_act):
        j = int(t_act / step_dur) % TN
        return [s0_gait[(k + j * nP) % TN] for k in range(TN)]

    # --- Simulate ---
    dt = m.opt.timestep
    sim_time = P['sim_time']
    settle_time = P['settle_time']
    total_steps = int(sim_time / dt)
    steer_base = num_axial + num_ring
    yaw_torque_val = float(P['yaw_torque'])

    heading_history = []
    head_xy_history = []
    # Enhanced trajectory recording (when save_traj=True)
    traj_times = []
    traj_plates_xy = []  # list of (num_plates, 2) arrays
    traj_com_xy = []
    traj_ht_dist = []
    traj_max_bend = []
    traj_heading_unwrapped = []
    _traj_heading_accum = 0.0
    _traj_prev_hdg = None
    prev_states = None  # for detecting gait state transitions (yaw coupling ratchet)
    prev_gait_j = -1  # for detecting gait step transitions (cumulative mode)
    cumulative_yaw = 0.0  # total accumulated heading from all completed bends
    t0 = time.time()
    nan_flag = False

    for step in range(total_steps):
        t = step * dt

        if step % 2000 == 0 and step > 0:
            if np.any(np.isnan(d.qpos)):
                nan_flag = True
                break

        # Track every 1000 steps
        if step % 1000 == 0:
            hx, hy = d.xpos[head_id, 0], d.xpos[head_id, 1]
            tx, ty = d.xpos[tail_id, 0], d.xpos[tail_id, 1]
            hdg = math.degrees(math.atan2(hx - tx, hy - ty))
            heading_history.append((t, hdg))
            head_xy_history.append((hx, hy))
            # Enhanced trajectory recording
            if save_traj:
                traj_times.append(t)
                plate_xy = np.array([[d.xpos[pids[p], 0], d.xpos[pids[p], 1]] for p in range(num_plates)])
                traj_plates_xy.append(plate_xy)
                com = plate_xy.mean(axis=0)
                traj_com_xy.append(com)
                ht_d = np.sqrt((hx - tx)**2 + (hy - ty)**2) * 1000
                traj_ht_dist.append(ht_d)
                # Per-plate heading and max bend
                p_hdgs = []
                for pp in range(num_plates - 1):
                    dx = d.xpos[pids[pp+1], 0] - d.xpos[pids[pp], 0]
                    dy = d.xpos[pids[pp+1], 1] - d.xpos[pids[pp], 1]
                    p_hdgs.append(math.degrees(math.atan2(dx, dy)))
                traj_max_bend.append(max(p_hdgs) - min(p_hdgs) if p_hdgs else 0)
                # Unwrapped heading accumulation
                if _traj_prev_hdg is not None:
                    dh = hdg - _traj_prev_hdg
                    if dh > 180: dh -= 360
                    elif dh < -180: dh += 360
                    _traj_heading_accum += dh
                _traj_prev_hdg = hdg
                traj_heading_unwrapped.append(_traj_heading_accum)

        # Control
        d.ctrl[:] = 0
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        if t >= settle_time:
            t_act = t - settle_time
            states = get_states(t_act)
            for seg in range(num_segments):
                s = states[seg]
                # Axial muscles: mi=0(right), mi=1(top), mi=2(left), mi=3(bottom)
                if s == 0:
                    pass  # all zero (relaxed)
                elif s == 1:
                    for mi in range(4):
                        d.ctrl[seg * 4 + mi] = 1.0  # full contraction (anchor)
                elif s == 2:
                    if state2_mode == "differential":
                        # Paper's State 2: left side contracts, right relaxes
                        d.ctrl[seg * 4 + 0] = 0.0  # right OFF
                        d.ctrl[seg * 4 + 1] = 0.0  # top OFF
                        d.ctrl[seg * 4 + 2] = 1.0  # left ON
                        d.ctrl[seg * 4 + 3] = 0.0  # bottom OFF
                    else:
                        for mi in range(4):
                            d.ctrl[seg * 4 + mi] = state2_axial
                elif s == 3:
                    if state2_mode == "differential":
                        # Paper's State 3: right side contracts, left relaxes
                        d.ctrl[seg * 4 + 0] = 1.0  # right ON
                        d.ctrl[seg * 4 + 1] = 0.0  # top OFF
                        d.ctrl[seg * 4 + 2] = 0.0  # left OFF
                        d.ctrl[seg * 4 + 3] = 0.0  # bottom OFF
                    else:
                        for mi in range(4):
                            d.ctrl[seg * 4 + mi] = state2_axial
                # Ring (only when cables are present)
                if num_ring > 0:
                    d.ctrl[num_axial + seg] = 1.0 if s == 0 else 0.0
                # Diagonal steering tendons
                if steer_in_s2:
                    if s == 2:
                        d.ctrl[steer_base + seg * 2] = 1.0
                    elif s == 3:
                        d.ctrl[steer_base + seg * 2 + 1] = 1.0

            # Yaw control modes
            if yaw_mode == "torque" and yaw_torque_val != 0.0:
                # Direct yaw torque via xfrc_applied (bypasses cable system)
                for seg in range(num_segments):
                    s = states[seg]
                    if s == 2:
                        d.xfrc_applied[pids[seg+1], 5] += yaw_torque_val
                        d.xfrc_applied[pids[seg], 5] -= yaw_torque_val
                    elif s == 3:
                        d.xfrc_applied[pids[seg+1], 5] -= yaw_torque_val
                        d.xfrc_applied[pids[seg], 5] += yaw_torque_val

            elif yaw_mode == "spring":
                # Inter-plate PD controller: drives relative yaw to desired angle
                anchor_damp = float(P.get('anchor_yaw_damp', 0.0))
                spring_relax = int(P.get('spring_relax', 0))  # 1=skip spring in State 0 (persist yaw)
                for seg in range(num_segments):
                    s = states[seg]

                    # Skip spring in State 0 if spring_relax=1 (let heading persist)
                    if spring_relax and s == 0:
                        continue

                    desired_rel = 0.0
                    if s == 2:
                        desired_rel = yaw_angle_rad   # left-bend: +θ
                    elif s == 3:
                        desired_rel = -yaw_angle_rad  # right-bend: -θ
                    # State 0 and 1: desired_rel = 0 (straighten)

                    # Read actual plate yaw (qpos at yaw DOF)
                    qpos_head = d.qpos[m.jnt_qposadr[yaw_jnt_ids[seg+1]]]
                    qpos_tail = d.qpos[m.jnt_qposadr[yaw_jnt_ids[seg]]]
                    actual_rel = qpos_head - qpos_tail

                    # Read actual plate yaw velocity
                    qvel_head = d.qvel[yaw_dof_ids[seg+1]]
                    qvel_tail = d.qvel[yaw_dof_ids[seg]]
                    actual_rel_vel = qvel_head - qvel_tail

                    # PD torque with clamping to prevent QACC
                    error = desired_rel - actual_rel
                    torque = yaw_kp * error - yaw_kd * actual_rel_vel
                    # Clamp torque: limit angular acceleration to prevent numerical instability
                    # Plate I ≈ 0.5 * mass * r² ≈ 0.5 * 0.02 * 0.022² ≈ 4.84e-6 kg·m²
                    max_torque = float(P.get('yaw_max_torque', 0.005))  # ~1000 rad/s² for 0.02kg plate
                    torque = max(min(torque, max_torque), -max_torque)

                    # Apply to joint DOFs (equal and opposite)
                    d.qfrc_applied[yaw_dof_ids[seg+1]] += torque
                    d.qfrc_applied[yaw_dof_ids[seg]] -= torque

                    # Anchor damping: State 1 plates resist yaw rotation (ground gripping)
                    # This creates asymmetric heading accumulation for circular locomotion
                    if s == 1 and anchor_damp > 0:
                        # Both plates of the anchored segment get high yaw damping
                        d.qfrc_applied[yaw_dof_ids[seg]] -= anchor_damp * d.qvel[yaw_dof_ids[seg]]
                        d.qfrc_applied[yaw_dof_ids[seg+1]] -= anchor_damp * d.qvel[yaw_dof_ids[seg+1]]

            elif yaw_mode == "cumulative":
                # Absolute cumulative heading controller:
                # Track total heading from all past bends, drive each plate to its absolute target.
                # Each completed bending cycle adds yaw_angle to cumulative offset.
                # Currently-bending segments add an in-progress offset to plates ahead.
                current_j = int(t_act / step_dur) % TN
                if current_j != prev_gait_j:
                    if prev_gait_j >= 0:
                        # Count bends that just completed (transitioned out of State 2/3)
                        old_states = get_states(prev_gait_j * step_dur)
                        for seg in range(num_segments):
                            if old_states[seg] == 2:
                                cumulative_yaw += yaw_angle_rad
                            elif old_states[seg] == 3:
                                cumulative_yaw -= yaw_angle_rad
                    prev_gait_j = current_j

                # Calculate desired absolute yaw for each plate
                # Base: all plates at cumulative heading + body curvature offset
                body_curv_rad = math.radians(float(P.get('body_curvature', 0)))
                desired_abs = [cumulative_yaw + p * body_curv_rad for p in range(num_plates)]
                # Add in-progress bend offset for currently-bending segments
                for seg in range(num_segments):
                    s = states[seg]
                    if s == 2:
                        # Plates ahead of bend get +θ (in-progress)
                        for p in range(seg + 1, num_plates):
                            desired_abs[p] += yaw_angle_rad
                    elif s == 3:
                        for p in range(seg + 1, num_plates):
                            desired_abs[p] -= yaw_angle_rad

                # PD controller: drive each plate toward its absolute target
                max_torque = float(P.get('yaw_max_torque', 0.005))
                for p in range(num_plates):
                    actual = d.qpos[m.jnt_qposadr[yaw_jnt_ids[p]]]
                    vel = d.qvel[yaw_dof_ids[p]]
                    error = desired_abs[p] - actual
                    torque = yaw_kp * error - yaw_kd * vel
                    torque = max(min(torque, max_torque), -max_torque)
                    d.qfrc_applied[yaw_dof_ids[p]] += torque

            # Dynamic yaw coupling: toggle equality constraints based on gait state
            # State 2/3: DEACTIVATE (plates free to bend independently)
            # State 0/1: ACTIVATE with current offset preserved (ratchet mechanism)
            # On transition from bend→coupled: update polycoef[0] = q2-q1 to preserve heading
            if yaw_eq_ids:
                for seg in range(num_segments):
                    s = states[seg]
                    if s == 2 or s == 3:
                        d.eq_active[yaw_eq_ids[seg]] = 0  # free to bend
                    else:
                        # On transition from bending to coupled: lock current offset
                        if prev_states is not None and prev_states[seg] in (2, 3):
                            q1 = d.qpos[m.jnt_qposadr[yaw_jnt_ids[seg]]]
                            q2 = d.qpos[m.jnt_qposadr[yaw_jnt_ids[seg+1]]]
                            # polycoef: q2 = polycoef[0] + polycoef[1]*q1
                            # Set offset so constraint preserves current angle difference
                            m.eq_data[yaw_eq_ids[seg], 0] = q2 - q1
                        d.eq_active[yaw_eq_ids[seg]] = 1  # coupled
                prev_states = states[:]

            # Anchor downforce: push anchored (State 1) plates down to increase ground friction
            # Higher normal force → more sliding friction → effective torsional resistance
            anchor_df = float(P.get('anchor_downforce', 0))
            if anchor_df > 0:
                for seg in range(num_segments):
                    if states[seg] == 1:
                        d.xfrc_applied[pids[seg], 2] -= anchor_df
                        d.xfrc_applied[pids[seg+1], 2] -= anchor_df

            # Lateral steering force: push plates sideways to create curved trajectory
            steer_f = float(P.get('steer_force', 0))
            if steer_f != 0:
                steer_m = P.get('steer_mode', 'extend')
                applied_plates = set()

                # Body-frame modes: compute force perpendicular to body heading
                # heading = atan2(head_x - tail_x, head_y - tail_y)
                # left perpendicular: (-cos(heading), sin(heading))
                body_frame = steer_m.startswith("body")
                if body_frame:
                    _hx = d.xpos[head_id, 0]
                    _hy = d.xpos[head_id, 1]
                    _tx = d.xpos[tail_id, 0]
                    _ty = d.xpos[tail_id, 1]
                    _heading = math.atan2(_hx - _tx, _hy - _ty)
                    # Force perpendicular to heading (leftward)
                    fx = -steer_f * math.cos(_heading)
                    fy = steer_f * math.sin(_heading)
                else:
                    fx = -steer_f
                    fy = 0.0

                # Determine which mode suffix (for body_* modes)
                effective_mode = steer_m.replace("body_", "") if body_frame else steer_m

                if effective_mode == "all":
                    for p in range(num_plates):
                        d.xfrc_applied[pids[p], 0] += fx
                        d.xfrc_applied[pids[p], 1] += fy
                elif effective_mode == "extend":
                    for seg in range(num_segments):
                        if states[seg] == 0:
                            if seg not in applied_plates:
                                d.xfrc_applied[pids[seg], 0] += fx
                                d.xfrc_applied[pids[seg], 1] += fy
                                applied_plates.add(seg)
                            if (seg + 1) not in applied_plates:
                                d.xfrc_applied[pids[seg + 1], 0] += fx
                                d.xfrc_applied[pids[seg + 1], 1] += fy
                                applied_plates.add(seg + 1)
                elif effective_mode == "anchor":
                    for seg in range(num_segments):
                        if states[seg] == 1:
                            if seg not in applied_plates:
                                d.xfrc_applied[pids[seg], 0] += fx
                                d.xfrc_applied[pids[seg], 1] += fy
                                applied_plates.add(seg)
                            if (seg + 1) not in applied_plates:
                                d.xfrc_applied[pids[seg + 1], 0] += fx
                                d.xfrc_applied[pids[seg + 1], 1] += fy
                                applied_plates.add(seg + 1)
                elif effective_mode == "head":
                    d.xfrc_applied[pids[num_plates - 1], 0] += fx
                    d.xfrc_applied[pids[num_plates - 1], 1] += fy

        mujoco.mj_step(m, d)

    elapsed = time.time() - t0

    # --- Results ---
    hx, hy = d.xpos[head_id, 0] * 1000, d.xpos[head_id, 1] * 1000
    tx, ty = d.xpos[tail_id, 0] * 1000, d.xpos[tail_id, 1] * 1000
    z_vals = [d.xpos[pids[p], 2] for p in range(num_plates)]
    z_range = (max(z_vals) - min(z_vals)) * 1000
    heading = math.degrees(math.atan2(d.xpos[head_id, 0] - d.xpos[tail_id, 0],
                                       d.xpos[head_id, 1] - d.xpos[tail_id, 1]))
    head_y0 = num_segments * seg_length * 1000
    fwd = hy - head_y0
    lateral = hx

    # Head-tail distance (body integrity check: should be ~250-325mm if not crumpled)
    ht_dist = math.sqrt((hx - tx)**2 + (hy - ty)**2)

    # Body center of mass displacement
    com_x = np.mean([d.xpos[pids[p], 0] for p in range(num_plates)]) * 1000
    com_y = np.mean([d.xpos[pids[p], 1] for p in range(num_plates)]) * 1000
    com_y0 = num_segments * seg_length / 2 * 1000  # initial COM Y
    com_disp = math.sqrt(com_x**2 + (com_y - com_y0)**2)

    # Per-plate heading: average heading of consecutive plate pairs
    plate_headings = []
    for p in range(num_plates - 1):
        dx = d.xpos[pids[p+1], 0] - d.xpos[pids[p], 0]
        dy = d.xpos[pids[p+1], 1] - d.xpos[pids[p], 1]
        plate_headings.append(math.degrees(math.atan2(dx, dy)))
    avg_heading = np.mean(plate_headings)  # body average heading
    max_bend = max(plate_headings) - min(plate_headings)  # max inter-segment bend

    # Heading change over time (using body average heading for stability)
    if len(heading_history) >= 2:
        hdg_start = heading_history[2][1] if len(heading_history) > 2 else heading_history[0][1]
        hdg_end = heading_history[-1][1]
        hdg_change = hdg_end - hdg_start
    else:
        hdg_change = 0.0

    # Unwrapped heading: accumulate total heading change without ±180° wrapping
    heading_unwrapped = 0.0
    if len(heading_history) >= 2:
        for i in range(1, len(heading_history)):
            delta = heading_history[i][1] - heading_history[i-1][1]
            # Unwrap: if delta jumps by >180°, subtract 360°; if <-180°, add 360°
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            heading_unwrapped += delta

    result = {
        "exp_id": exp_id,
        "params": params,
        "heading_final": round(heading, 2),
        "avg_heading": round(avg_heading, 2),
        "heading_change": round(hdg_change, 2),
        "forward_mm": round(fwd, 1),
        "lateral_mm": round(lateral, 1),
        "ht_dist_mm": round(ht_dist, 1),
        "max_bend_deg": round(max_bend, 1),
        "com_disp_mm": round(com_disp, 1),
        "z_range_mm": round(z_range, 1),
        "heading_total_deg": round(heading_unwrapped, 1),
        "nan": nan_flag,
        "wall_time_s": round(elapsed, 1),
    }

    # Save trajectory data if requested
    if save_traj and traj_times:
        traj_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'record', 'v3', 'trajectories')
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, f'{exp_id}_traj.npz')
        np.savez(traj_path,
                 times=np.array(traj_times),
                 plates_xy=np.array(traj_plates_xy),  # (T, num_plates, 2)
                 com_xy=np.array(traj_com_xy),         # (T, 2)
                 ht_dist=np.array(traj_ht_dist),       # (T,)
                 max_bend=np.array(traj_max_bend),      # (T,)
                 heading_unwrapped=np.array(traj_heading_unwrapped),  # (T,)
                 num_plates=num_plates,
                 seg_length=seg_length)
        result["traj_file"] = traj_path

    return result


if __name__ == "__main__":
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    params = {}
    save_traj = False
    for arg in sys.argv[2:]:
        if arg == '--save_traj':
            save_traj = True
        elif arg.startswith('save_traj') and '=' in arg:
            save_traj = arg.split('=', 1)[1] not in ('0', 'false', 'False')
        elif '=' in arg:
            k, v = arg.split('=', 1)
            k = k.lstrip('-')
            try:
                v = float(v)
                if v == int(v) and 'solref' not in k and 'solimp' not in k:
                    v = int(v) if abs(v) < 1e15 else v
            except ValueError:
                pass
            params[k] = v

    print(f"=== Experiment {exp_id} ===")
    print(f"Params: {params}")
    result = run_experiment(exp_id, params, save_traj=save_traj)
    print(f"\n=== RESULT ===")
    print(json.dumps(result, indent=2))
