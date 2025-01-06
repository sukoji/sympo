"""Create a slide-ready C0 LLM-only vs C3 WBS-quality comparison.

The slide must not penalize C0 for assignment/debate dimensions that are outside
the LLM-only condition. The defensible claim is within the WBS-quality rubric:
C0 also produces a reasonable WBS draft, but C3 improves the Structure judge
score mainly through estimate/buffer realism while preserving hierarchy and
task specificity.
"""
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "comparison_4backbones" / "figures_interpretation"
OUT.mkdir(parents=True, exist_ok=True)

C0_PATH = ROOT / "gemma26_ablation" / "snapshots" / "wbs_snapshot_C0_llm_only_r1_qwen-api_gemma26_ablation_20260423_135834.json"
C3_PATH = ROOT / "gemma26_ablation" / "snapshots" / "wbs_snapshot_C3_3rounds_r2_qwen-api_gemma26_ablation_20260423_144446.json"
SUMMARY_PATH = ROOT / "gemma26_ablation" / "summary_finegrain.csv"

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
font_manager.fontManager.addfont(FONT_PATH)
font_manager.fontManager.addfont(FONT_BOLD_PATH)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_summary_row(condition, run_id):
    with open(SUMMARY_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] == condition and int(r["run_id"]) == run_id:
                return r
    raise ValueError((condition, run_id))


def task_stats(d):
    tasks = d["wbs_tasks"]
    l3 = [t for t in tasks if t.get("level") == "L3"]
    return {
        "total": len(tasks),
        "l3": len(l3),
        "assigned": sum(bool(t.get("assigned_to")) for t in l3),
        "buffer_days": sum(float(t.get("buffer_days") or 0) for t in tasks),
        "deps": sum(len(t.get("dependencies") or []) for t in tasks),
        "debate": len(d.get("debate_log") or []),
    }


def structure_parts(row):
    # Current judge reasons explicitly expose A/B/C sub-scores.
    if row["condition"] == "C0_llm_only":
        return {"Hierarchy": 0.5, "Est./buffer": 0.4, "Task quality": 1.0}
    return {"Hierarchy": 0.5, "Est./buffer": 0.8, "Task quality": 1.0}


def wrap(s, width=28):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def box(ax, xy, wh, fc, ec="#CBD5E1", lw=1.2, radius=0.025):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, size=13, color="#111827", weight="normal", ha="left", va="top", **kwargs):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kwargs)


def task_card(ax, x, y, w, h, title, rows, accent, note=None):
    box(ax, (x, y), (w, h), "#FFFFFF", ec="#CBD5E1", lw=1.0)
    text(ax, x + 0.018, y + h - 0.028, title, size=15, weight="bold", color=accent)
    y0 = y + h - 0.082
    headers = ["Task", "Role", "Duration", "Deps/Buffer"]
    widths = [0.46, 0.19, 0.17, 0.16]
    cx = x + 0.018
    for head, ww in zip(headers, widths):
        text(ax, cx, y0, head, size=11.5, weight="bold", color="#475569")
        cx += w * ww
    ax.plot([x + 0.014, x + w - 0.014], [y0 - 0.022, y0 - 0.022], color="#E2E8F0", lw=1)
    yy = y0 - 0.048
    for r in rows:
        cx = x + 0.018
        values = [wrap(r[0], 22), wrap(r[1], 13), wrap(r[2], 10), wrap(r[3], 12)]
        colors = ["#111827", "#111827", "#111827", "#111827"]
        for val, ww, c in zip(values, widths, colors):
            text(ax, cx, yy, val, size=11.5, color=c)
            cx += w * ww
        yy -= 0.070
    if note:
        text(ax, x + 0.018, y + 0.026, note, size=11.5, color=accent, weight="bold")


def metric_pill(ax, x, y, w, h, label, left, right, left_color="#991B1B", right_color="#166534"):
    box(ax, (x, y), (w, h), "#F8FAFC", ec="#CBD5E1", lw=1.0)
    text(ax, x + 0.014, y + h - 0.018, label, size=11.0, color="#475569", weight="bold")
    text(ax, x + w * 0.32, y + 0.022, left, size=16, color=left_color, weight="bold", ha="center", va="bottom")
    text(ax, x + w * 0.76, y + 0.022, right, size=16, color=right_color, weight="bold", ha="center", va="bottom")
    ax.plot([x + w * 0.52, x + w * 0.52], [y + 0.018, y + h - 0.018], color="#CBD5E1", lw=1)


def score_bars(ax, x, y, w, h, c0_parts, c3_parts):
    box(ax, (x, y), (w, h), "#FFFFFF", ec="#CBD5E1", lw=1.0)
    text(ax, x + 0.018, y + h - 0.030, "Structure judge 내부 근거", size=15, weight="bold", color="#111827")
    yy = y + h - 0.085
    for label in ["Hierarchy", "Est./buffer", "Task quality"]:
        c0v = c0_parts[label]
        c3v = c3_parts[label]
        text(ax, x + 0.018, yy + 0.010, label, size=11.5, weight="bold", color="#475569")
        ax.add_patch(plt.Rectangle((x + 0.150, yy), w * 0.27 * c0v, 0.018, color="#991B1B", alpha=0.85))
        ax.add_patch(plt.Rectangle((x + 0.150, yy - 0.028), w * 0.27 * c3v, 0.018, color="#166534", alpha=0.90))
        text(ax, x + 0.150 + w * 0.28, yy + 0.012, f"{c0v:.1f}", size=10.5, color="#991B1B", va="center")
        text(ax, x + 0.150 + w * 0.28, yy - 0.016, f"{c3v:.1f}", size=10.5, color="#166534", va="center")
        yy -= 0.060
    text(ax, x + 0.018, y + 0.026, "차이는 assignment/debate가 아니라 WBS 구조 평가 안의 estimation/buffer realism", size=11.5, color="#334155")


def main():
    c0 = load_json(C0_PATH)
    c3 = load_json(C3_PATH)
    c0_row = load_summary_row("C0_llm_only", 1)
    c3_row = load_summary_row("C3_3rounds", 2)
    s0 = task_stats(c0)
    s3 = task_stats(c3)
    c0_parts = structure_parts(c0_row)
    c3_parts = structure_parts(c3_row)

    fig, ax = plt.subplots(figsize=(18.2, 11.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    text(ax, 0.035, 0.975, "LLM-only 생성의 한계: WBS 품질 지표 안에서만 비교", size=22, weight="bold")
    text(ax, 0.035, 0.940, "공정 비교: assignment/debate는 C0 조건 밖이므로 제외, Structure judge와 WBS row 근거로 비교", size=13.5, color="#64748B")

    # Column headers
    box(ax, (0.035, 0.815), (0.43, 0.075), "#FEF2F2", ec="#FCA5A5", lw=1.5)
    box(ax, (0.535, 0.815), (0.43, 0.075), "#ECFDF5", ec="#86EFAC", lw=1.5)
    text(ax, 0.055, 0.862, "C0: LLM-only baseline", size=19, weight="bold", color="#991B1B")
    text(ax, 0.555, 0.862, "C3: SYMPo 3-round orchestration", size=19, weight="bold", color="#166534")
    text(ax, 0.055, 0.833, "단일 WBS draft 생성 조건", size=13, color="#7F1D1D")
    text(ax, 0.555, 0.833, "동일 backbone + C3 orchestration 조건", size=13, color="#14532D")

    # Metrics
    metric_y = 0.700
    metric_w = 0.215
    metric_h = 0.085
    metric_pill(ax, 0.035, metric_y, metric_w, metric_h, "WBS Structure Judge", f"{float(c0_row['judge_structure']):.3f}", f"{float(c3_row['judge_structure']):.3f}")
    metric_pill(ax, 0.277, metric_y, metric_w, metric_h, "MECE / Granularity", "1.0 / 1.0", "1.0 / 1.0")
    metric_pill(ax, 0.520, metric_y, metric_w, metric_h, "Buffer ratio", f"{float(c0_row['buffer_ratio_pct']):.1f}%", f"{float(c3_row['buffer_ratio_pct']):.1f}%")
    metric_pill(ax, 0.762, metric_y, metric_w, metric_h, "Dependency links", f"{s0['deps']}", f"{s3['deps']}")

    # WBS samples
    c0_rows = [
        ("고객사 CS 워크플로우 분석", "Planner", "4d", "1 / 0d"),
        ("Salesforce 연동 인터페이스 설계", "Backend", "5d", "1 / 0d"),
    ]
    c3_rows = [
        ("Figma 기반 와이어프레임 제작", "Designer", "5d", "1 / 0d"),
        ("DB 스키마 및 ERD 설계", "Backend", "3d", "1 / +1d"),
    ]
    task_card(ax, 0.035, 0.405, 0.43, 0.260, "공통 산출물 샘플: WBS row", c0_rows, "#991B1B", note="C0도 역할·기간·의존성 기반의 WBS draft는 생성함")
    task_card(ax, 0.535, 0.405, 0.43, 0.260, "공통 산출물 샘플: WBS row", c3_rows, "#166534", note="C3는 WBS row에 buffer realism이 추가됨")

    # WBS-quality rubric evidence cards.
    box(ax, (0.035, 0.205), (0.43, 0.165), "#FFF7ED", ec="#FDBA74", lw=1.2)
    text(ax, 0.055, 0.340, "C0 structure judge reason", size=15, weight="bold", color="#9A3412")
    text(ax, 0.055, 0.304, "A=0.5 B=0.4 C=1.0", size=14, weight="bold", color="#991B1B")
    text(ax, 0.055, 0.270, "3 L1s + good L3 coverage, task titles are domain-specific", size=12.2, color="#111827")
    text(ax, 0.055, 0.239, "감점 근거: all estimates 1-10d but no buffer", size=12.2, color="#7F1D1D")

    box(ax, (0.535, 0.205), (0.43, 0.165), "#F0FDF4", ec="#86EFAC", lw=1.2)
    text(ax, 0.555, 0.340, "C3 structure judge reason", size=15, weight="bold", color="#166534")
    text(ax, 0.555, 0.304, "A=0.5 B=0.8 C=1.0", size=14, weight="bold", color="#166534")
    text(ax, 0.555, 0.270, "3 L1s limits hierarchy score, but L3 depth is perfect", size=12.2, color="#111827")
    text(ax, 0.555, 0.239, "개선 근거: buffer is 12.6%, tasks are highly specific", size=12.2, color="#14532D")

    box(ax, (0.035, 0.055), (0.93, 0.125), "#F8FAFC", ec="#CBD5E1", lw=1.1)
    text(ax, 0.055, 0.150, "해석 포인트", size=15, weight="bold", color="#111827")
    bullets = [
        "C0도 MECE, granularity, schedule feasibility는 모두 1.0이므로 WBS draft 자체는 성립한다.",
        "유의미한 품질 차이는 Structure judge 내부의 estimation/buffer realism: C0 0.4 → C3 0.8이다.",
    ]
    yy = 0.118
    for b in bullets:
        text(ax, 0.055, yy, "• " + b, size=12.8, color="#334155")
        yy -= 0.041

    # Bottom source note.
    text(
        ax,
        0.965,
        0.025,
        "Source: eval_results/gemma26_ablation snapshots, C0 r1 vs C3 r2",
        size=10.5,
        color="#64748B",
        ha="right",
        va="bottom",
    )

    fig.savefig(OUT / "figG_llm_only_limitation_output_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Markdown summary for slide speaker notes / report traceability.
    summary_md = OUT / "llm_only_vs_c3_sample_summary.md"
    summary_md.write_text(
        f"""# LLM-only Limitation Sample: C0 vs C3

## Source snapshots

- C0: `{C0_PATH.relative_to(ROOT)}`
- C3: `{C3_PATH.relative_to(ROOT)}`

## Fair comparison principle

- C0는 LLM-only WBS draft 조건이므로 assignment/debate 차원을 결함으로 감점하지 않는다.
- C0에서 assignment/debate judge가 N/A인 것은 산출물 실패가 아니라 실험 조건의 범위 차이다.
- 따라서 슬라이드 주장은 "LLM-only가 WBS를 못 만든다"가 아니라 "C3가 WBS Structure rubric 안에서 estimate/buffer realism을 개선했다"로 둔다.

## Comparable WBS-quality evidence

| 항목 | C0 LLM-only | C3 SYMPo |
|---|---:|---:|
| Total tasks | {s0['total']} | {s3['total']} |
| L3 tasks | {s0['l3']} | {s3['l3']} |
| Dependency links | {s0['deps']} | {s3['deps']} |
| MECE score | {float(c0_row['mece_score']):.1f} | {float(c3_row['mece_score']):.1f} |
| Granularity fitness | {float(c0_row['granularity_fitness']):.1f} | {float(c3_row['granularity_fitness']):.1f} |
| Schedule feasibility | {float(c0_row['schedule_feasibility']):.1f} | {float(c3_row['schedule_feasibility']):.1f} |
| Buffer ratio | {float(c0_row['buffer_ratio_pct']):.1f}% | {float(c3_row['buffer_ratio_pct']):.1f}% |
| Structure judge | {float(c0_row['judge_structure']):.3f} | {float(c3_row['judge_structure']):.3f} |

## Structure judge sub-score evidence

| 항목 | C0 LLM-only | C3 SYMPo |
|---|---:|---:|
| A: hierarchy / depth | {c0_parts['Hierarchy']:.1f} | {c3_parts['Hierarchy']:.1f} |
| B: estimate / buffer realism | {c0_parts['Est./buffer']:.1f} | {c3_parts['Est./buffer']:.1f} |
| C: task specificity | {c0_parts['Task quality']:.1f} | {c3_parts['Task quality']:.1f} |

## Output-level interpretation

- C0도 계층형 WBS 초안, 역할명, 기간, 의존성을 생성하며 rule-based WBS quality 지표는 높다.
- 유의미한 차이는 Structure judge 내부의 estimate/buffer realism이다. C0는 no buffer로 B=0.4, C3는 12.6% buffer로 B=0.8이다.
- Assignment/debate는 C0 조건 밖이므로 이 슬라이드의 성능 주장 근거로 쓰지 않는다.
- 슬라이드 메시지: **LLM-only도 WBS draft는 가능하지만, C3는 동일 backbone에서 일정/버퍼 현실성이 더 높은 WBS로 개선된다.**
""",
        encoding="utf-8",
    )

    print(OUT / "figG_llm_only_limitation_output_sample.png")
    print(summary_md)


if __name__ == "__main__":
    main()
