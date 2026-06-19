from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "record" / "current" / "worm_v6_advisor_update_ppt"
ASSET_DIR = OUT_DIR / "assets" / "v96"
PPTX_PATH = OUT_DIR / "worm_v6_advisor_update_20260603_v96.pptx"
MANIFEST_JSON = OUT_DIR / "ppt_manifest_v96.json"
MANIFEST_MD = OUT_DIR / "ppt_manifest_v96.md"

V86_VIDEO_DIR = ROOT / "record" / "current" / "flat_omni_v86b_server_robust_forward_left_videos"
V59_VIDEO_DIR = ROOT / "record" / "current" / "flat_omni_v59_dynamic_sweep_hd"
V95_SCAN_DIR = ROOT / "record" / "current" / "flat_omni_v95_server_mixed_positive_vx_hardcase_scan"
V96_SCAN_DIR = ROOT / "record" / "current" / "flat_omni_v96_server_mixed_component_sign_scan"
V96_SERVER_RUN = (
    "/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training/runs/"
    "worm_v6_ppo_flat_random_v96_server_mixed_component_sign_from_v95final_np2"
)
V96_SERVER_LOG = (
    "/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training/"
    "server_logs/v96_flat_mixed_component_sign_20260603_180738.log"
)


COLORS = {
    "ink": RGBColor(28, 31, 38),
    "muted": RGBColor(92, 100, 112),
    "line": RGBColor(221, 226, 234),
    "panel": RGBColor(246, 248, 251),
    "blue": RGBColor(32, 92, 181),
    "teal": RGBColor(0, 126, 135),
    "green": RGBColor(33, 130, 83),
    "red": RGBColor(190, 62, 50),
    "orange": RGBColor(205, 112, 35),
    "purple": RGBColor(95, 78, 158),
    "white": RGBColor(255, 255, 255),
}


VIDEO_ITEMS = [
    ("forward", V86_VIDEO_DIR / "forward_12s_1080p.mp4"),
    ("reverse", V86_VIDEO_DIR / "reverse_12s_1080p.mp4"),
    ("left", V86_VIDEO_DIR / "lateral_left_12s_1080p.mp4"),
    ("right", V86_VIDEO_DIR / "lateral_right_12s_1080p.mp4"),
    ("yaw left", V86_VIDEO_DIR / "yaw_left_12s_1080p.mp4"),
    ("yaw right", V86_VIDEO_DIR / "yaw_right_12s_1080p.mp4"),
    ("hard +vx,+vy", V86_VIDEO_DIR / "hard_forward_left_12s_1080p.mp4"),
    ("hard +vx,-vy", V86_VIDEO_DIR / "hard_forward_right_12s_1080p.mp4"),
    ("continuous sweep", V86_VIDEO_DIR / "continuous_sweep_24s_1080p.mp4"),
    ("long sweep", V59_VIDEO_DIR / "v59_best_continuous_sweep_36s_1080p.mp4"),
]


LITERATURE_ROWS = [
    [
        "Satheeshbabu et al. 2020",
        "CPG + DRL soft snake",
        "CPG keeps rhythmic gait; RL modulates control.",
        "Supports prior + residual instead of naked joint RL.",
    ],
    [
        "Liu, Onal & Fu 2023",
        "Contact-aware CPG soft snake",
        "RL/feedback changes contact-aware CPG terms.",
        "Sand/slope need contact/slip signals, not only more steps.",
    ],
    [
        "Shi, Dear & Kelly 2020",
        "Deep RL snake locomotion",
        "DRL can discover terrestrial/aquatic snake gaits.",
        "Gait discovery is background; strict command tracking is harder.",
    ],
    [
        "Saga et al. 2016; Tesen line",
        "Earthworm peristaltic RL",
        "Q-learning/Actor-Critic for peristaltic crawling.",
        "Peristaltic RL is not new; our claim must be deployable multimodal.",
    ],
    [
        "Marvi et al. 2014; Gong et al. 2013",
        "Sidewinding sand/slope",
        "Slope/sand success depends on slip and contact distribution.",
        "Terrain claim needs slip/contact metrics.",
    ],
    [
        "Kumar et al. 2021; Margolis et al. 2024",
        "Legged velocity-command RL",
        "Obs includes vx/vy/yaw commands; rewards use RMSE and curricula.",
        "Use feasible command envelope, not arbitrary full omni.",
    ],
    [
        "Wu et al. 2023; Gaitor 2024",
        "Latent multi-gait control",
        "Latent gait variables must be ablated and interpreted.",
        "Our learned gait gate needs fixed-gate/no-gate comparisons.",
    ],
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def run_command(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def video_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    raw = run_command([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "json",
        str(path),
    ])
    info = {"exists": True, "path": str(path), "bytes": path.stat().st_size}
    if raw:
        try:
            data = json.loads(raw)
            stream = data.get("streams", [{}])[0]
            info.update({
                "width": int(stream.get("width", 0) or 0),
                "height": int(stream.get("height", 0) or 0),
                "duration_s": float(stream.get("duration", 0.0) or 0.0),
            })
        except Exception:
            pass
    return info


def make_thumbnail(label: str, path: Path, seek_s: float = 4.0) -> Path | None:
    if not path.exists():
        return None
    safe = "".join(ch if ch.isalnum() else "_" for ch in label.lower())
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    out = ASSET_DIR / f"{safe}_{digest}_thumb.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out
    subprocess.run([
        "ffmpeg",
        "-y",
        "-ss", str(seek_s),
        "-i", str(path),
        "-vframes", "1",
        "-q:v", "2",
        str(out),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return out if out.exists() and out.stat().st_size > 0 else None


def load_v95_summary() -> list[dict]:
    path = V95_SCAN_DIR / "v95_acceptance_summary.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("items", [])


def load_v96_summary() -> list[dict]:
    path = V96_SCAN_DIR / "v96_flat_mixed_component_sign_summary.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("items", [])


def set_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.color.rgb = COLORS["line"]


def add_box(slide, x, y, w, h, fill=COLORS["panel"], line=COLORS["line"]):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(0.75)
    return shape


def add_text(slide, text, x, y, w, h, size=18, color=COLORS["ink"],
             bold=False, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_title(slide, title, subtitle=None):
    add_text(slide, title, 0.55, 0.25, 12.2, 0.45, size=24, bold=True)
    if subtitle:
        add_text(slide, subtitle, 0.57, 0.75, 12.1, 0.34, size=11, color=COLORS["muted"])


def add_bullets(slide, items, x, y, w, h, size=13, color=COLORS["ink"]):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = "Microsoft YaHei"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(4)
    return box


def add_table(slide, rows, x, y, w, h, font_size=8):
    table_shape = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)
    )
    table = table_shape.table
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.margin_left = Inches(0.03)
            cell.margin_right = Inches(0.03)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            fill = COLORS["blue"] if r == 0 else (COLORS["white"] if r % 2 else COLORS["panel"])
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            for p in cell.text_frame.paragraphs:
                p.font.name = "Microsoft YaHei"
                p.font.size = Pt(font_size)
                p.font.color.rgb = COLORS["white"] if r == 0 else COLORS["ink"]
                p.font.bold = (r == 0)
    return table_shape


def add_image_if_exists(slide, path: Path, x, y, w, h):
    if not path.exists():
        add_box(slide, x, y, w, h)
        add_text(slide, f"missing image\n{path.name}", x + 0.1, y + 0.2, w - 0.2, h - 0.4,
                 size=9, color=COLORS["muted"], align=PP_ALIGN.CENTER)
        return None
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def add_footer(slide, page, note="Worm V6 advisor update | 2026-06-03"):
    add_text(slide, note, 0.6, 7.15, 9.8, 0.2, size=7, color=COLORS["muted"])
    add_text(slide, str(page), 12.25, 7.15, 0.45, 0.2, size=7, color=COLORS["muted"],
             align=PP_ALIGN.RIGHT)


def build_deck() -> dict:
    ensure_dirs()
    video_infos = []
    thumbs = {}
    for label, path in VIDEO_ITEMS:
        info = video_info(path)
        info["label"] = label
        video_infos.append(info)
        thumbs[label] = make_thumbnail(label, path)

    v95_items = load_v95_summary()
    v96_items = load_v96_summary()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    slide_no = 1

    # 1
    slide = prs.slides.add_slide(blank)
    add_box(slide, 0, 0, 13.333, 7.5, fill=RGBColor(247, 249, 252), line=RGBColor(247, 249, 252))
    add_text(slide, "Worm V6 蛇-蠕虫双模态机器人", 0.75, 0.75, 11.8, 0.6, size=30, bold=True)
    add_text(slide, "导师汇报：视频证据、创新点、论文定位与 V96 继续训练", 0.78, 1.45, 11.2, 0.4,
             size=17, color=COLORS["blue"], bold=True)
    add_box(slide, 0.78, 2.15, 5.8, 3.65, fill=COLORS["white"])
    add_bullets(slide, [
        "当前定位：weak omnidirectional / six-direction primitive controller",
        "未达成：continuous vx/vy/yaw velocity tracking",
        "当前主要失败：mixed vx/vy 分量错号，而不是单纯 reward 低",
        "本轮动作：V96 已完成服务器训练，formal scan 仍未通过",
    ], 1.05, 2.48, 5.25, 2.7, size=15)
    add_image_if_exists(slide, thumbs.get("continuous sweep") or Path("missing"), 7.0, 2.05, 5.55, 3.15)
    add_text(slide, "视频缩略图来自 V86b continuous sweep", 7.02, 5.32, 5.4, 0.22, size=8, color=COLORS["muted"])
    add_footer(slide, slide_no); slide_no += 1

    # 2
    slide = prs.slides.add_slide(blank)
    add_title(slide, "一句话结论", "我们已经有可展示的多方向运动与可部署接口，但严格连续全向仍未通过。")
    cards = [
        ("已有", "80D obs / 12D action ABI 固定；视频、轨迹、scan、文献框架齐全。", COLORS["green"]),
        ("进步", "V95 old projection gate 下 yaw/planar sign 改善，yaw RMSE 约 0.020 rad/s。", COLORS["blue"]),
        ("问题", "mixed vx/vy 分量仍错号，V95 final robust wrong mixed comp = 10。", COLORS["red"]),
        ("当前", "V96 formal scan 全部未通过：wrong mixed component = 8-9。", COLORS["orange"]),
    ]
    for i, (head, body, color) in enumerate(cards):
        x = 0.7 + i * 3.1
        add_box(slide, x, 1.45, 2.75, 4.8, fill=COLORS["white"])
        add_text(slide, head, x + 0.2, 1.75, 2.2, 0.35, size=20, bold=True, color=color)
        add_text(slide, body, x + 0.2, 2.35, 2.25, 2.5, size=14)
    add_footer(slide, slide_no); slide_no += 1

    # 3
    slide = prs.slides.add_slide(blank)
    add_title(slide, "机器人与实机部署接口", "观测只使用现实可采集信号；训练 reward 可用仿真速度，但 policy obs 不用。")
    rows = [
        ["组", "维度", "来源"],
        ["command", "3", "cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm"],
        ["joint_pos", "11", "6 个伸缩 + 5 个偏航编码器"],
        ["joint_vel", "11", "编码器速度估计"],
        ["previous_action", "11", "上一时刻下发电机动作"],
        ["segment IMU gravity", "21", "7 个体节 IMU 重力方向"],
        ["segment IMU gyro", "21", "7 个体节角速度"],
        ["phase clock", "2", "1s 周期 sin/cos"],
    ]
    add_table(slide, rows, 0.7, 1.35, 6.1, 4.85, font_size=9)
    add_box(slide, 7.2, 1.35, 5.35, 4.85, fill=COLORS["white"])
    add_text(slide, "Action", 7.55, 1.72, 2.6, 0.35, size=20, bold=True, color=COLORS["purple"])
    add_bullets(slide, [
        "12D policy action = 11D residual motor action + 1D learned latent gait gate z_g",
        "z_g 连续混合 peristaltic 与 serpentine gait priors",
        "最终电机命令 = gait prior + residual，经 EMA、限幅和电机范围映射",
        "ABI 保持不变，便于后续实机日志回放和部署包导出",
    ], 7.55, 2.25, 4.55, 2.8, size=13)
    add_footer(slide, slide_no); slide_no += 1

    # 4
    slide = prs.slides.add_slide(blank)
    add_title(slide, "方法结构：先验 + 残差 + 潜在 gait gate", "这条路线与 CPG+RL 文献一致：不给网络裸学全部关节，而是学习调制和修正。")
    add_box(slide, 0.75, 1.35, 12.0, 4.8, fill=COLORS["white"])
    pipeline = [
        ("80D deployable obs", 0.95, 2.45, COLORS["blue"]),
        ("PPO ActorCritic\n512-256-128", 3.25, 2.35, COLORS["purple"]),
        ("12D action\n11 residual + z_g", 5.65, 2.35, COLORS["teal"]),
        ("gait priors\nworm/snake", 8.15, 2.35, COLORS["orange"]),
        ("11 motor commands", 10.55, 2.45, COLORS["green"]),
    ]
    for text, x, y, color in pipeline:
        add_box(slide, x, y, 1.85, 0.9, fill=RGBColor(250, 252, 255), line=color)
        add_text(slide, text, x + 0.08, y + 0.15, 1.65, 0.5, size=10, bold=True, color=color,
                 align=PP_ALIGN.CENTER)
        if x < 10:
            add_text(slide, "→", x + 1.92, y + 0.2, 0.45, 0.35, size=24, color=COLORS["muted"])
    add_text(slide, "论文表述", 1.0, 4.15, 1.2, 0.25, size=13, bold=True, color=COLORS["blue"])
    add_text(
        slide,
        "The policy outputs an 11-dimensional residual motor action and a one-dimensional learned latent gait gate z_g, "
        "which continuously blends the peristaltic and serpentine gait priors.",
        2.1, 4.1, 9.65, 0.55, size=13, color=COLORS["ink"],
    )
    add_text(slide, "注意：当前 gate 有命令条件中心与 reward 正则，论文应称 prior-regularized learned gate。", 1.0, 5.15, 10.8, 0.35,
             size=11, color=COLORS["red"])
    add_footer(slide, slide_no); slide_no += 1

    # 5
    slide = prs.slides.add_slide(blank)
    add_title(slide, "视频索引：六方向固定命令", "这些视频用于导师快速看效果；PPT 里放缩略图，视频文件保留为外部附件。")
    gallery = [
        ("forward", 0.65, 1.25), ("reverse", 4.55, 1.25), ("left", 8.45, 1.25),
        ("right", 0.65, 4.05), ("yaw left", 4.55, 4.05), ("yaw right", 8.45, 4.05),
    ]
    for label, x, y in gallery:
        add_box(slide, x, y, 3.35, 2.35, fill=COLORS["white"])
        add_image_if_exists(slide, thumbs.get(label) or Path("missing"), x + 0.1, y + 0.12, 3.15, 1.65)
        add_text(slide, label, x + 0.15, y + 1.82, 1.2, 0.2, size=11, bold=True, color=COLORS["blue"])
        path = dict(VIDEO_ITEMS).get(label)
        add_text(slide, str(path.relative_to(ROOT)) if path else "", x + 0.15, y + 2.04, 3.05, 0.2,
                 size=6, color=COLORS["muted"])
    add_footer(slide, slide_no); slide_no += 1

    # 6
    slide = prs.slides.add_slide(blank)
    add_title(slide, "视频索引：连续速度变化与 mixed hard cases", "continuous sweep 用来看速度命令变化时的响应；hard cases 专门看 diagonal 分量是否错号。")
    for label, x in [("continuous sweep", 0.75), ("hard +vx,+vy", 4.95), ("hard +vx,-vy", 9.15)]:
        add_box(slide, x, 1.4, 3.65, 4.65, fill=COLORS["white"])
        add_image_if_exists(slide, thumbs.get(label) or Path("missing"), x + 0.15, 1.6, 3.35, 1.9)
        add_text(slide, label, x + 0.2, 3.72, 2.8, 0.3, size=14, bold=True, color=COLORS["teal"])
        path = dict(VIDEO_ITEMS).get(label)
        add_text(slide, str(path.relative_to(ROOT)) if path else "", x + 0.2, 4.16, 3.1, 0.48,
                 size=7, color=COLORS["muted"])
    add_text(slide, "解释：这页不是证明已经全向，而是展示当前控制器在速度变化和 hard diagonal 下的真实行为。", 0.8, 6.35, 11.6, 0.35,
             size=11, color=COLORS["red"])
    add_footer(slide, slide_no); slide_no += 1

    # 7
    slide = prs.slides.add_slide(blank)
    add_title(slide, "轨迹与时程图：速度变化下的真实误差", "左图是每体节 XY 轨迹，右图是 commanded vs measured 与 gait gate。")
    add_image_if_exists(slide, V86_VIDEO_DIR / "continuous_sweep_24s_1080p_trajectory.png", 0.55, 1.2, 6.0, 4.8)
    add_image_if_exists(slide, V86_VIDEO_DIR / "continuous_sweep_24s_1080p_telemetry.png", 6.8, 1.2, 6.0, 4.8)
    add_footer(slide, slide_no); slide_no += 1

    # 8
    slide = prs.slides.add_slide(blank)
    add_title(slide, "V95 strict scan 审计", "旧 projection gate 会误判；新 mixed-component gate 后 V95 全部不通过。")
    rows = [["checkpoint", "accepted", "planar RMSE", "yaw RMSE", "wrong mixed", "counterexample"]]
    for item in v95_items:
        rows.append([
            item["label"].replace("v95_", ""),
            str(item["accepted"]).lower(),
            f"{item['planar_rmse_m_s']:.4f}",
            f"{item['yaw_rmse_rad_s']:.4f}",
            str(item["wrong_mixed_component_sign_count"]),
            f"cmd=({item['counter_cmd_vx_m_s']:+.2f},{item['counter_cmd_vy_m_s']:+.3f},{item['counter_cmd_yaw_rad_s']:+.1f}) "
            f"meas=({item['counter_body_vx_m_s']:+.3f},{item['counter_body_vy_m_s']:+.3f},{item['counter_yaw_rate_rad_s']:+.3f})",
        ])
    add_table(slide, rows, 0.55, 1.25, 12.25, 3.2, font_size=7)
    add_bullets(slide, [
        "V95 final robust: planar RMSE 0.0616 m/s, yaw RMSE 0.0201 rad/s，但 wrong mixed component = 10。",
        "核心教训：不能只看沿命令方向的 projection，mixed vx/vy 必须逐分量同号。",
        "因此当前结论不是“全向已完成”，而是“方向投影改善，暴露出更严格的分量级问题”。",
    ], 0.8, 4.85, 11.7, 1.1, size=12)
    add_footer(slide, slide_no); slide_no += 1

    # 9
    slide = prs.slides.add_slide(blank)
    add_title(slide, "V96 结果：projection 过了，但 mixed component 没过", "服务器训练和 formal scan 已完成；结果不能写成全向已解决。")
    rows = [
        ["项", "V96 设置 / 结果"],
        ["resume", "V95 final model"],
        ["run label", "flat_random_v96_server_mixed_component_sign_from_v95final_np2"],
        ["reward", "v36 mixed_component_sign"],
        ["selection", "omni_tracking_scan_v5"],
        ["network", "Actor/Critic 512-256-128"],
        ["sensor robustness", "encoder/IMU noise + 1-step delay + 0.90 saturation"],
        ["train status", "complete; no accepted best_model.zip produced"],
    ]
    add_table(slide, rows, 0.55, 1.2, 5.75, 4.9, font_size=8)
    add_box(slide, 6.55, 1.2, 6.25, 4.9, fill=COLORS["white"])
    add_text(slide, "formal scan 结论", 6.85, 1.5, 2.8, 0.35, size=18, bold=True, color=COLORS["red"])
    scan_rows = [["checkpoint", "accepted", "wrong mixed", "counterexample"]]
    for item in v96_items:
        scan_rows.append([
            item["label"].replace("v96_", "").replace("_model", ""),
            str(item["accepted"]).lower(),
            str(item.get("wrong_mixed_component_sign_count")),
            f"cmd=({item['counter_cmd_vx_m_s']:+.2f},{item['counter_cmd_vy_m_s']:+.3f}) "
            f"meas=({item['counter_body_vx_m_s']:+.3f},{item['counter_body_vy_m_s']:+.3f})",
        ])
    if len(scan_rows) == 1:
        scan_rows.append(["scan", "pending", "n/a", "V96 summary not pulled"])
    add_table(slide, scan_rows, 6.75, 1.95, 5.85, 2.65, font_size=5)
    add_bullets(slide, [
        "进步：formal scan 的 wrong_planar=0、wrong_yaw=0，RMSE 也在阈值内。",
        "短板：mixed vx/vy 仍有 8-9 个分量错号；projection pass 不等于 component pass。",
        "下一轮应改混合分量机制，而不是只继续同一 reward 长训。",
    ], 6.85, 4.9, 5.5, 1.15, size=10)
    add_text(slide, f"Server run: {V96_SERVER_RUN}", 0.75, 6.45, 11.9, 0.22, size=6, color=COLORS["muted"])
    add_footer(slide, slide_no); slide_no += 1

    # 10
    slide = prs.slides.add_slide(blank)
    add_title(slide, "创新点：可以写，但要守住证据边界", "导师汇报建议把创新点和未完成点放在同一页。")
    rows = [
        ["创新点", "当前证据", "边界"],
        ["实机可部署观测契约", "80D obs 明确只用命令、编码器、IMU、previous action、phase", "需要实机日志闭环验证"],
        ["潜在 gait gate", "12D action 中 z_g 控制 worm/snake prior blend", "需 fixed-gate/no-gate ablation"],
        ["蛇形 + 蠕虫先验", "视频中可见 axial slide 与 yaw-wave 的组合", "当前 flat 成功主要是结构先验，不是纯 RL 涌现"],
        ["strict scan 评估", "35 command + robust scan + mixed-component gate", "V95/V96 均未过，问题已定位到 mixed 分量错号"],
        ["弹簧钢片建模路线", "视觉钢片 + 等效回弹力方案已规划", "真实柔性体/参数标定仍待实机实验"],
    ]
    add_table(slide, rows, 0.55, 1.25, 12.25, 5.25, font_size=7)
    add_footer(slide, slide_no); slide_no += 1

    # 11
    slide = prs.slides.add_slide(blank)
    add_title(slide, "为什么一直推进慢", "不是没有训练，而是目标从“能动”升级成“分量级连续速度跟踪”。")
    add_box(slide, 0.7, 1.35, 5.65, 4.95, fill=COLORS["white"])
    add_text(slide, "技术难点", 1.0, 1.7, 2.0, 0.3, size=18, bold=True, color=COLORS["red"])
    add_bullets(slide, [
        "蛇/蠕虫依赖身体-地面摩擦，vx、vy、yaw 不是独立驱动通道。",
        "mixed vx+vy 不是线性叠加，两个 primitive 会互相干扰。",
        "纯 yaw 命令容易蜷缩或带平移漂移，需要 body compactness 与 drift penalty。",
        "视频看起来会动，不等于速度分量跟踪正确。",
    ], 1.0, 2.25, 4.85, 2.6, size=13)
    add_box(slide, 6.9, 1.35, 5.65, 4.95, fill=COLORS["white"])
    add_text(slide, "现在的解决策略", 7.2, 1.7, 3.1, 0.3, size=18, bold=True, color=COLORS["green"])
    add_bullets(slide, [
        "先定义可行速度包络，别一开始追求任意 full omni。",
        "axis-only → mixed planar → mixed yaw → terrain transfer。",
        "每轮训练都以 strict scan 和 counterexample 选模型。",
        "用可部署 ABI 约束观测，提前为实机部署做准备。",
    ], 7.2, 2.25, 4.85, 2.6, size=13)
    add_footer(slide, slide_no); slide_no += 1

    # 12
    slide = prs.slides.add_slide(blank)
    add_title(slide, "文献对比：已有工作怎么训练", "结论：我们的路线应当是 prior-guided residual RL，而不是裸关节端到端。")
    rows = [["论文/方向", "做什么", "关键方法", "对我们的启发"]] + LITERATURE_ROWS
    add_table(slide, rows, 0.35, 1.05, 12.65, 5.85, font_size=6)
    add_footer(slide, slide_no); slide_no += 1

    # 13
    slide = prs.slides.add_slide(blank)
    add_title(slide, "论文主张建议", "把目标命名为有限包络内的命令条件多模态平面运动，而不是无限制全向。")
    add_box(slide, 0.85, 1.35, 11.65, 1.2, fill=RGBColor(238, 244, 255), line=COLORS["blue"])
    add_text(slide, "finite-envelope command-conditioned multimodal planar locomotion", 1.1, 1.72, 10.9, 0.3,
             size=22, bold=True, color=COLORS["blue"], align=PP_ALIGN.CENTER)
    add_bullets(slide, [
        "避免说：fully omnidirectional locomotion 已解决。",
        "可以说：在 flat 上建立了可部署观测/动作契约，并通过 prior + residual + gait gate 推进多方向运动。",
        "等 V97 或后续版本通过 component scan 后，再升级为 continuous body-frame velocity tracking on flat terrain。",
        "sand/slope 暂时作为未来迁移和实机标定目标，不作为已完成结论。",
    ], 1.15, 3.05, 10.9, 2.25, size=15)
    add_footer(slide, slide_no); slide_no += 1

    # 14
    slide = prs.slides.add_slide(blank)
    add_title(slide, "接下来怎么验收", "每一轮都必须给出视频 + JSON/CSV + 轨迹图，而不是只看 reward。")
    rows = [
        ["验收项", "阈值/输出"],
        ["direction gate", "wrong_planar=0, wrong_yaw=0, wrong_mixed_component=0"],
        ["tracking gate", "planar RMSE <= 0.10 m/s; yaw RMSE <= 0.20 rad/s"],
        ["off-axis", "mean off-axis speed <= 0.08 m/s"],
        ["stop", "zero-command mean speed <= 0.02 m/s"],
        ["视频", "6 fixed directions + continuous sweep + hard diagonal cases"],
        ["图", "head/segment XY, per-segment displacement, command vs measured"],
        ["论文边界", "未达 gate 前只称 weak omni prototype"],
    ]
    add_table(slide, rows, 0.9, 1.25, 5.75, 4.95, font_size=9)
    add_box(slide, 7.1, 1.25, 5.25, 4.95, fill=COLORS["white"])
    add_text(slide, "V97 下一轮要跑", 7.4, 1.65, 2.8, 0.35, size=18, bold=True, color=COLORS["teal"])
    add_bullets(slide, [
        "35-command nominal scan",
        "35-command robust scan",
        "六方向 12s 或 20s 视频",
        "continuous sweep 高清视频",
        "trajectory/telemetry PNG",
        "README/progress/PPT 更新并同步 GitHub",
    ], 7.4, 2.2, 4.3, 2.55, size=13)
    add_footer(slide, slide_no); slide_no += 1

    # 15
    slide = prs.slides.add_slide(blank)
    add_title(slide, "钢片/实机物理：怎么和论文连接", "真实机器人是舵机拉绳单向收缩 + 钢片被动回弹，不是双向位置伺服。")
    add_bullets(slide, [
        "当前仿真：用等效 actuator/friction/contact + 视觉钢片表达结构。",
        "更准确方案：钢片力-位移曲线 → 等效非线性弹簧/阻尼；舵机阶跃响应 → 一阶延迟/饱和/死区；接触摩擦 → 地形参数。",
        "论文中不要说已经模拟真实柔性钢片；应说 equivalent passive rebound model and planned hardware identification。",
        "导师可给的数据：钢片拉压曲线、舵机收缩/释放响应、质量几何、轮/地接触摩擦、IMU 静态标定。",
    ], 0.9, 1.45, 11.45, 3.2, size=15)
    add_box(slide, 1.1, 5.15, 10.9, 0.7, fill=RGBColor(255, 248, 235), line=COLORS["orange"])
    add_text(slide, "结论：短期先用 MuJoCo 训练；IsaacLab 作为渲染/并行备选，等 open-loop parity 通过后再迁移正式 RL。", 1.35, 5.35, 10.4, 0.25,
             size=13, bold=True, color=COLORS["orange"])
    add_footer(slide, slide_no); slide_no += 1

    # 16
    slide = prs.slides.add_slide(blank)
    add_title(slide, "汇报时建议这样讲", "把负结果转成清晰的问题定义和下一步实验。")
    add_bullets(slide, [
        "1. 我们不是在做单纯动画，而是在做可部署控制器，所以 obs/action ABI 固定。",
        "2. 当前机器人已能展示多方向运动，但严格连续全向速度跟踪还没通过。",
        "3. V95 暴露出一个更严格问题：projection 方向可以对，但 vx/vy 分量仍可能错号。",
        "4. V96 已完成：projection-level 指标看起来好，但 formal component scan 仍失败。",
        "5. 论文贡献聚焦：deployable contract + prior-guided residual policy + learned gait gate + strict mixed-command evaluation。",
    ], 1.0, 1.45, 11.2, 4.2, size=17)
    add_footer(slide, slide_no); slide_no += 1

    prs.save(PPTX_PATH)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pptx": str(PPTX_PATH),
        "slides": slide_no - 1,
        "video_items": video_infos,
        "v95_scan_dir": str(V95_SCAN_DIR),
        "v96_scan_dir": str(V96_SCAN_DIR),
        "v96_scan_items": v96_items,
        "v96_server_run": V96_SERVER_RUN,
        "v96_server_log": V96_SERVER_LOG,
        "claims_boundary": (
            "Use weak omnidirectional / six-direction primitive controller until "
            "mixed-component strict tracking passes."
        ),
        "literature_count": len(LITERATURE_ROWS),
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Worm V6 Advisor PPT V96 Manifest",
        "",
        f"- PPTX: `{PPTX_PATH}`",
        f"- Slides: {slide_no - 1}",
        f"- V95 scan dir: `{V95_SCAN_DIR}`",
        f"- V96 scan dir: `{V96_SCAN_DIR}`",
        f"- V96 server run: `{V96_SERVER_RUN}`",
        f"- V96 server log: `{V96_SERVER_LOG}`",
        f"- V96 formal scan items: {len(v96_items)}",
        "",
        "## Videos",
    ]
    for item in video_infos:
        status = "ok" if item.get("exists") else "missing"
        dur = item.get("duration_s")
        dur_text = f", {dur:.1f}s" if isinstance(dur, float) else ""
        lines.append(f"- {item['label']}: {status}{dur_text} `{item.get('path', '')}`")
    MANIFEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_deck()
    print(f"wrote {manifest['pptx']}")
    print(f"slides {manifest['slides']}")
    print(f"manifest {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
