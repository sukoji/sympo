"""
Generate high-quality README figures (no PPT / no broken fonts).

Outputs:
  docs/assets/figures/en/*.png
  docs/assets/figures/ko/*.png
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_EN = ROOT / "docs" / "assets" / "figures" / "en"
OUT_KO = ROOT / "docs" / "assets" / "figures" / "ko"

C_BG = "#ffffff"
C_SURFACE = "#f8fafc"
C_BORDER = "#e2e8f0"
C_TEXT = "#0f172a"
C_MUTED = "#64748b"
C_PRIMARY = "#2563eb"
C_PRIMARY_SOFT = "#dbeafe"
C_INDIGO = "#4f46e5"
C_VIOLET = "#7c3aed"
C_TEAL = "#0d9488"
STAGE_COLORS = ["#2563eb", "#3b82f6", "#6366f1", "#8b5cf6", "#1d4ed8"]

DPI = 260
PAD = 0.55


def _available_fonts() -> set[str]:
    return {f.name for f in mpl.font_manager.fontManager.ttflist}


def _pick_font(candidates: list[str], available: set[str], fallback: str = "DejaVu Sans") -> str:
    return next((c for c in candidates if c in available), fallback)


def _resolve_fonts() -> tuple[str, str]:
    available = _available_fonts()
    en_font = _pick_font(
        ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        available,
    )
    ko_font = _pick_font(
        ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"],
        available,
        en_font,
    )
    return en_font, ko_font


@contextmanager
def _font_context(lang: str, en_font: str, ko_font: str):
    primary = ko_font if lang == "ko" else en_font
    backup = en_font if lang == "ko" else ko_font
    old = mpl.rcParams.copy()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [primary, backup, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": C_BG,
        "axes.facecolor": C_BG,
        "savefig.facecolor": C_BG,
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    })
    try:
        yield primary
    finally:
        mpl.rcParams.update(old)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=PAD, transparent=False)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def _rounded_card(ax, xy, w, h, *, color=C_SURFACE, edge=C_BORDER, radius=0.10, lw=1.0, zorder=2):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=color,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _header(ax, title: str, subtitle: str, *, y_title=0.90, y_sub=0.82):
    ax.text(0.06, y_title, title, fontsize=20, fontweight="bold", color=C_TEXT, transform=ax.transAxes)
    ax.text(0.06, y_sub, subtitle, fontsize=11, color=C_MUTED, transform=ax.transAxes)


def build_pipeline(lang: str, out: Path):
    stages_en = [
        ("01", "Input", "PRD · team · meetings"),
        ("02", "Draft", "3-level WBS"),
        ("03", "Route", "Skill-based routing"),
        ("04", "Debate", "Buffers · risks · R&R"),
        ("05", "Assign", "Final plan lock-in"),
    ]
    stages_ko = [
        ("01", "입력", "PRD · 팀 · 회의록"),
        ("02", "초안", "3단계 WBS"),
        ("03", "라우팅", "스킬 기반 라우팅"),
        ("04", "토론", "버퍼 · 리스크 · R&R"),
        ("05", "배정", "최종 계획 확정"),
    ]
    title_en = "symPO orchestration pipeline"
    title_ko = "symPO 오케스트레이션 파이프라인"
    subtitle_en = "Multi-agent flow from product context to debated, assignable WBS"
    subtitle_ko = "제품 컨텍스트에서 토론·배정 가능한 WBS까지"

    stages = stages_ko if lang == "ko" else stages_en
    title = title_ko if lang == "ko" else title_en
    subtitle = subtitle_ko if lang == "ko" else subtitle_en

    fig_h = 5.8 if lang == "ko" else 5.4
    fig, ax = plt.subplots(figsize=(14.0, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _rounded_card(ax, (0.03, 0.06), 0.94, 0.88, color=C_SURFACE, radius=0.12)
    _header(ax, title, subtitle, y_title=0.86, y_sub=0.78)

    n = len(stages)
    gap = 0.032
    card_w = (0.86 - gap * (n - 1)) / n
    x0 = 0.07
    y = 0.18
    h = 0.50

    for i, (num, name, desc) in enumerate(stages):
        x = x0 + i * (card_w + gap)
        color = STAGE_COLORS[i]
        _rounded_card(ax, (x, y), card_w, h, color="#ffffff", edge=color, radius=0.08, lw=1.6)

        badge_y = y + h - 0.10
        ax.add_patch(plt.Circle(
            (x + card_w / 2, badge_y), 0.022, color=color,
            transform=ax.transAxes, zorder=4,
        ))
        ax.text(
            x + card_w / 2, badge_y, num,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", transform=ax.transAxes, zorder=5,
        )
        ax.text(
            x + card_w / 2, y + h * 0.58, name,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C_TEXT, transform=ax.transAxes,
        )
        ax.text(
            x + card_w / 2, y + h * 0.28, desc,
            ha="center", va="center", fontsize=9, color=C_MUTED,
            transform=ax.transAxes, linespacing=1.35,
        )

        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + card_w + gap * 0.72, y + h * 0.48),
                xytext=(x + card_w + gap * 0.28, y + h * 0.48),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.6, shrinkA=0, shrinkB=0),
                xycoords=ax.transAxes,
            )

    _save(fig, out / "pipeline.png")


def build_outputs(lang: str, out: Path):
    title_en = "Deliverables at a glance"
    title_ko = "산출물 한눈에"
    left_en, right_en = "Per-member task cards", "Schedule timeline"
    left_ko, right_ko = "팀원별 태스크 카드", "일정 타임라인"

    members_en = [
        ("Alex", "4 tasks · 18 days", ["API design", "Auth", "Deploy"]),
        ("Jordan", "3 tasks · 14 days", ["Dashboard", "E2E tests"]),
        ("Sam", "5 tasks · 22 days", ["Data pipe", "Model eval", "Monitoring"]),
    ]
    members_ko = [
        ("이동헌", "4 tasks · 18일", ["API 설계", "인증", "배포"]),
        ("장선열", "3 tasks · 14일", ["대시보드", "E2E 테스트"]),
        ("윤수빈", "5 tasks · 22일", ["데이터 파이프", "모델 평가", "모니터링"]),
    ]
    gantt = [(1, 5), (2, 4), (1, 7)]  # (start_week, duration_weeks)
    colors = ["#2563eb", "#7c3aed", "#0d9488"]

    title = title_ko if lang == "ko" else title_en
    left_title = left_ko if lang == "ko" else left_en
    right_title = right_ko if lang == "ko" else right_en
    members = members_ko if lang == "ko" else members_en

    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _rounded_card(ax, (0.03, 0.05), 0.94, 0.90, color=C_SURFACE, radius=0.12)
    ax.text(0.06, 0.88, title, fontsize=20, fontweight="bold", color=C_TEXT, transform=ax.transAxes)

    # Left panel
    _rounded_card(ax, (0.06, 0.12), 0.41, 0.68, color="#ffffff", radius=0.10)
    ax.text(0.09, 0.74, left_title, fontsize=12, fontweight="bold", color=C_TEXT, transform=ax.transAxes)

    row_h = 0.155
    row_gap = 0.035
    row_top = 0.58
    for i, (name, meta, tasks) in enumerate(members):
        y = row_top - i * (row_h + row_gap)
        _rounded_card(ax, (0.09, y), 0.35, row_h, color=C_PRIMARY_SOFT, edge=colors[i], radius=0.06, lw=1.4)
        ax.text(0.115, y + row_h * 0.72, name, fontsize=12, fontweight="bold", color=C_TEXT, transform=ax.transAxes)
        ax.text(0.115, y + row_h * 0.30, meta, fontsize=9, color=C_MUTED, transform=ax.transAxes)
        task_line = "  ·  ".join(tasks)
        ax.text(0.26, y + row_h * 0.50, task_line, fontsize=8.8, color=C_MUTED, transform=ax.transAxes, va="center")

    # Right panel — real mini-gantt axes
    _rounded_card(ax, (0.52, 0.12), 0.42, 0.68, color="#ffffff", radius=0.10)
    panel = fig.add_axes([0.545, 0.20, 0.385, 0.54])
    panel.set_facecolor("#ffffff")
    for spine in panel.spines.values():
        spine.set_visible(True)
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.0)
    panel.spines["top"].set_visible(False)
    panel.spines["right"].set_visible(False)

    panel.set_title(right_title, loc="left", fontsize=12, fontweight="bold", color=C_TEXT, pad=14)
    panel.set_xlim(0.5, 8.5)
    panel.set_ylim(-0.5, len(members) - 0.5)
    panel.set_yticks(range(len(members)))
    panel.set_yticklabels([m[0] for m in members], fontsize=10)
    panel.invert_yaxis()
    panel.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    panel.set_xticklabels([f"W{w}" for w in range(1, 9)], fontsize=8.5, color=C_MUTED)
    panel.set_xlabel("Week", fontsize=9, color=C_MUTED, labelpad=8)
    panel.grid(axis="x", color=C_BORDER, linestyle="-", linewidth=0.8, alpha=0.7)
    panel.tick_params(axis="y", length=0, pad=8)
    panel.tick_params(axis="x", colors=C_MUTED, pad=4)

    for i, ((name, _, _), (start, dur), color) in enumerate(zip(members, gantt, colors)):
        panel.barh(
            i, dur, left=start, height=0.42, color=color, alpha=0.88,
            edgecolor="none", zorder=3,
        )

    _save(fig, out / "outputs.png")


def build_validation(lang: str, out: Path):
    title_en = "Validation snapshot"
    title_ko = "검증 결과 요약"
    subtitle_en = "Human survey (N=65) · 9 Likert items · practitioners with project experience"
    subtitle_ko = "인간 설문 (N=65) · 9개 리커트 문항 · 프로젝트 경험 실무자"

    cats_en = ["WBS quality", "R&R fit", "Feasibility", "Overall"]
    cats_ko = ["WBS 완성도", "R&R 적정성", "실현 가능성", "전체 평균"]
    scores = [4.51, 4.44, 4.48, 4.48]
    pos_rates = [92.6, 90.8, 89.2, 91.6]

    debate_en = ["Assign only", "+1 round", "+3 rounds"]
    debate_ko = ["배정만", "+1 라운드", "+3 라운드"]
    debate_scores = [0.58, 0.70, 0.73]

    title = title_ko if lang == "ko" else title_en
    subtitle = subtitle_ko if lang == "ko" else subtitle_en
    cats = cats_ko if lang == "ko" else cats_en
    debate_labels = debate_ko if lang == "ko" else debate_en
    debate_note_en = "Debate rounds lift judge score\n(Gemma-26B MoE, representative run)"
    debate_note_ko = "토론 라운드 증가 시 Judge 점수 상승\n(Gemma-26B MoE 대표값)"
    debate_note = debate_note_ko if lang == "ko" else debate_note_en

    fig = plt.figure(figsize=(14.0, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14, wspace=0.42)

    fig.text(0.07, 0.92, title, fontsize=20, fontweight="bold", color=C_TEXT, ha="left")
    fig.text(0.07, 0.845, subtitle, fontsize=11, color=C_MUTED, ha="left")

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_BG)

    y = np.arange(len(cats))
    bar_colors = [C_PRIMARY, C_INDIGO, C_TEAL, C_VIOLET]
    bars = ax1.barh(y, scores, color=bar_colors, height=0.48, alpha=0.92)
    ax1.set_xlim(0, 5.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(cats, fontsize=11)
    ax1.set_xlabel("Mean score ( / 5.00 )", fontsize=10, color=C_MUTED, labelpad=10)
    ax1.axvline(4.0, color="#cbd5e1", ls="--", lw=1.2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="x", colors=C_MUTED, pad=6)
    ax1.tick_params(axis="y", pad=10)
    for i, (bar, pr) in enumerate(zip(bars, pos_rates)):
        ax1.text(
            bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
            f"{scores[i]:.2f}   ·   {pr:.0f}% positive",
            va="center", fontsize=10, color=C_TEXT,
        )

    x = np.arange(len(debate_labels))
    ax2.bar(x, debate_scores, color=[C_BORDER, C_INDIGO, C_PRIMARY], width=0.52, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(debate_labels, fontsize=10)
    ax2.set_ylim(0, 0.92)
    ax2.set_ylabel("LLM judge overall", fontsize=10, color=C_MUTED, labelpad=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", pad=8)
    ax2.text(0.5, 0.97, debate_note, ha="center", va="top", transform=ax2.transAxes,
             fontsize=9, color=C_MUTED, linespacing=1.4)
    for i, v in enumerate(debate_scores):
        ax2.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold", color=C_TEXT)

    _save(fig, out / "validation.png")


def build_hero(lang: str, out: Path):
    tag_en = "Multi-agent WBS orchestration"
    tag_ko = "멀티에이전트 WBS 오케스트레이션"
    badges_en = ["5-stage pipeline", "N=65 survey", "LangGraph · MCP"]
    badges_ko = ["5단계 파이프라인", "설문 N=65", "LangGraph · MCP"]
    tag = tag_ko if lang == "ko" else tag_en
    badges = badges_ko if lang == "ko" else badges_en

    fig, ax = plt.subplots(figsize=(14.0, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _rounded_card(ax, (0.03, 0.12), 0.94, 0.76, color=C_PRIMARY_SOFT, edge=C_BORDER, radius=0.16)
    ax.text(0.07, 0.62, "sym", fontsize=36, fontweight="bold", color=C_PRIMARY, va="center", transform=ax.transAxes)
    ax.text(0.19, 0.62, "PO", fontsize=36, fontweight="bold", color=C_TEXT, va="center", transform=ax.transAxes)
    ax.text(0.07, 0.32, tag, fontsize=12, color=C_MUTED, va="center", transform=ax.transAxes)

    bx = 0.48
    bw = 0.155
    gap = 0.018
    for i, label in enumerate(badges):
        x = bx + i * (bw + gap)
        _rounded_card(ax, (x, 0.28), bw, 0.44, color="#ffffff", edge=C_BORDER, radius=0.08)
        ax.text(x + bw / 2, 0.50, label, ha="center", va="center", fontsize=9.5, color=C_TEXT, transform=ax.transAxes)

    _save(fig, out / "hero.png")


def main():
    en_font, ko_font = _resolve_fonts()
    print(f"fonts: en={en_font!r}, ko={ko_font!r}")
    for lang, folder in [("en", OUT_EN), ("ko", OUT_KO)]:
        with _font_context(lang, en_font, ko_font):
            build_pipeline(lang, folder)
            build_outputs(lang, folder)
            build_validation(lang, folder)
            build_hero(lang, folder)


if __name__ == "__main__":
    main()
