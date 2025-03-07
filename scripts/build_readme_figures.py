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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_EN = ROOT / "docs" / "assets" / "figures" / "en"
OUT_KO = ROOT / "docs" / "assets" / "figures" / "ko"

# Brand palette — light, GitHub-friendly
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

DPI = 240


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
    fig.savefig(path, bbox_inches="tight", pad_inches=0.35, transparent=False)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def _rounded_card(ax, xy, w, h, color=C_SURFACE, edge=C_BORDER, radius=0.08):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=color,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(box)
    return box


def build_pipeline(lang: str, out: Path):
    stages_en = [
        ("01", "Input", "PRD · team · meetings"),
        ("02", "Draft", "3-level WBS draft"),
        ("03", "Route", "Skill-based agent call"),
        ("04", "Debate", "Buffers · risks · R&R"),
        ("05", "Assign", "Final plan lock-in"),
    ]
    stages_ko = [
        ("01", "입력", "PRD · 팀원 · 회의록"),
        ("02", "초안", "3단계 WBS 생성"),
        ("03", "라우팅", "스킬 기반 에이전트 호출"),
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

    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _rounded_card(ax, (0.02, 0.08), 0.96, 0.84, color=C_SURFACE)
    ax.text(0.05, 0.82, title, fontsize=18, fontweight="bold", color=C_TEXT, transform=ax.transAxes)
    ax.text(0.05, 0.72, subtitle, fontsize=10.5, color=C_MUTED, transform=ax.transAxes)

    n = len(stages)
    gap = 0.018
    card_w = (0.88 - gap * (n - 1)) / n
    x0 = 0.06
    y = 0.22
    h = 0.42

    for i, (num, name, desc) in enumerate(stages):
        x = x0 + i * (card_w + gap)
        color = STAGE_COLORS[i]
        _rounded_card(ax, (x, y), card_w, h, color="#ffffff", edge=color)
        ax.add_patch(plt.Circle((x + 0.04, y + h - 0.08), 0.028, color=color, transform=ax.transAxes, zorder=3))
        ax.text(x + 0.04, y + h - 0.08, num, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", transform=ax.transAxes, zorder=4)
        ax.text(x + card_w / 2, y + h * 0.52, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color=C_TEXT, transform=ax.transAxes)
        ax.text(x + card_w / 2, y + h * 0.22, desc, ha="center", va="center",
                fontsize=8.2, color=C_MUTED, transform=ax.transAxes)

        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + card_w + gap * 0.35, y + h * 0.5),
                xytext=(x + card_w + gap * 0.05, y + h * 0.5),
                arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.4),
                xycoords=ax.transAxes,
            )

    _save(fig, out / "pipeline.png")


def build_outputs(lang: str, out: Path):
    title_en = "Deliverables at a glance"
    title_ko = "산출물 한눈에"
    left_en, right_en = "Per-member task cards", "Schedule timeline"
    left_ko, right_ko = "팀원별 태스크 카드", "일정 타임라인"

    members_en = [
        ("Alex", "4 tasks · 18d", ["API design", "Auth module", "Deploy"]),
        ("Jordan", "3 tasks · 14d", ["Dashboard UI", "E2E tests"]),
        ("Sam", "5 tasks · 22d", ["Data pipeline", "Model eval", "Monitoring"]),
    ]
    members_ko = [
        ("이동헌", "4 tasks · 18d", ["API 설계", "인증 모듈", "배포"]),
        ("장선열", "3 tasks · 14d", ["대시보드 UI", "E2E 테스트"]),
        ("윤수빈", "5 tasks · 22d", ["데이터 파이프", "모델 평가", "모니터링"]),
    ]
    colors = ["#2563eb", "#7c3aed", "#0d9488"]

    title = title_ko if lang == "ko" else title_en
    left_title = left_ko if lang == "ko" else left_en
    right_title = right_ko if lang == "ko" else right_en
    members = members_ko if lang == "ko" else members_en

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _rounded_card(ax, (0.02, 0.06), 0.96, 0.88, color=C_SURFACE)
    ax.text(0.05, 0.86, title, fontsize=18, fontweight="bold", color=C_TEXT, transform=ax.transAxes)

    # Left panel — cards
    _rounded_card(ax, (0.05, 0.12), 0.42, 0.68, color="#ffffff")
    ax.text(0.07, 0.74, left_title, fontsize=11, fontweight="bold", color=C_TEXT, transform=ax.transAxes)
    for i, (name, meta, tasks) in enumerate(members):
        y = 0.62 - i * 0.2
        _rounded_card(ax, (0.07, y), 0.38, 0.16, color=C_PRIMARY_SOFT, edge=colors[i])
        ax.text(0.09, y + 0.11, name, fontsize=11, fontweight="bold", color=C_TEXT, transform=ax.transAxes)
        ax.text(0.09, y + 0.04, meta, fontsize=8.5, color=C_MUTED, transform=ax.transAxes)
        pill_x = 0.22
        for j, t in enumerate(tasks):
            ax.text(pill_x, y + 0.11 - j * 0.035, f"• {t}", fontsize=7.8, color=C_MUTED, transform=ax.transAxes)

    # Right panel — gantt
    _rounded_card(ax, (0.52, 0.12), 0.43, 0.68, color="#ffffff")
    ax.text(0.54, 0.74, right_title, fontsize=11, fontweight="bold", color=C_TEXT, transform=ax.transAxes)
    weeks = np.arange(1, 9)
    for i, (name, _, _) in enumerate(members):
        y = 0.58 - i * 0.16
        start = 0.55 + 0.06 * i
        width = 0.12 + 0.08 * (2 - i)
        bar = FancyBboxPatch(
            (start, y), width, 0.07,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            facecolor=colors[i], edgecolor="none", alpha=0.85,
            transform=ax.transAxes,
        )
        ax.add_patch(bar)
        ax.text(0.54, y + 0.035, name, fontsize=9, color=C_TEXT, transform=ax.transAxes, va="center")
    ax.text(0.54, 0.16, "W1", fontsize=7.5, color=C_MUTED, transform=ax.transAxes)
    ax.text(0.72, 0.16, "W4", fontsize=7.5, color=C_MUTED, transform=ax.transAxes)
    ax.text(0.88, 0.16, "W8", fontsize=7.5, color=C_MUTED, transform=ax.transAxes)
    ax.plot([0.54, 0.92], [0.18, 0.18], color=C_BORDER, lw=1, transform=ax.transAxes)

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
    debate_scores = [0.58, 0.70, 0.73]  # Gemma26 representative

    title = title_ko if lang == "ko" else title_en
    subtitle = subtitle_ko if lang == "ko" else subtitle_en
    cats = cats_ko if lang == "ko" else cats_en
    debate_labels = debate_ko if lang == "ko" else debate_en
    debate_note_en = "Debate rounds lift judge score (Gemma-26B MoE, representative run)"
    debate_note_ko = "토론 라운드 증가 시 Judge 점수 상승 (Gemma-26B MoE 대표값)"

    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in (ax1, ax2):
        ax.set_facecolor(C_BG)

    fig.suptitle(title, fontsize=18, fontweight="bold", color=C_TEXT, x=0.06, ha="left", y=0.98)
    fig.text(0.06, 0.90, subtitle, fontsize=10, color=C_MUTED)

    y = np.arange(len(cats))
    bars = ax1.barh(y, scores, color=[C_PRIMARY, C_INDIGO, C_TEAL, C_VIOLET], height=0.55, alpha=0.9)
    ax1.set_xlim(0, 5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(cats, fontsize=10)
    ax1.set_xlabel("Mean score ( / 5.00 )", fontsize=9, color=C_MUTED)
    ax1.axvline(4.0, color=C_BORDER, ls="--", lw=1)
    ax1.spines[["top", "right"]].set_visible(False)
    for i, (bar, pr) in enumerate(zip(bars, pos_rates)):
        ax1.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height() / 2,
                 f"{scores[i]:.2f}  ·  {pr:.0f}% positive",
                 va="center", fontsize=9, color=C_TEXT)

    ax2.bar(debate_labels, debate_scores, color=[C_BORDER, C_INDIGO, C_PRIMARY], width=0.55)
    ax2.set_ylim(0, 0.85)
    ax2.set_ylabel("LLM judge overall", fontsize=9, color=C_MUTED)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.5, 0.78, debate_note_ko if lang == "ko" else debate_note_en,
             ha="center", transform=ax2.transAxes, fontsize=8.5, color=C_MUTED)
    for i, v in enumerate(debate_scores):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color=C_TEXT)

    _save(fig, out / "validation.png")


def build_hero(lang: str, out: Path):
    """Compact hero strip for README top."""
    tag_en = "Multi-agent WBS orchestration"
    tag_ko = "멀티에이전트 WBS 오케스트레이션"
    badges_en = ["5-stage pipeline", "N=65 survey", "LangGraph · MCP"]
    badges_ko = ["5단계 파이프라인", "설문 N=65", "LangGraph · MCP"]
    tag = tag_ko if lang == "ko" else tag_en
    badges = badges_ko if lang == "ko" else badges_en

    fig, ax = plt.subplots(figsize=(12.5, 2.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _rounded_card(ax, (0.02, 0.10), 0.96, 0.80, color=C_PRIMARY_SOFT, edge=C_BORDER, radius=0.14)
    ax.text(0.06, 0.58, "sym", fontsize=32, fontweight="bold", color=C_PRIMARY, va="center", transform=ax.transAxes)
    ax.text(0.16, 0.58, "PO", fontsize=32, fontweight="bold", color=C_TEXT, va="center", transform=ax.transAxes)
    ax.text(0.06, 0.30, tag, fontsize=11.5, color=C_MUTED, va="center", transform=ax.transAxes)

    bx = 0.46
    bw = 0.165 if lang == "en" else 0.155
    for i, label in enumerate(badges):
        x = bx + i * (bw + 0.012)
        _rounded_card(ax, (x, 0.30), bw, 0.40, color="#ffffff", edge=C_BORDER, radius=0.06)
        ax.text(x + bw / 2, 0.50, label, ha="center", va="center", fontsize=8.5, color=C_TEXT, transform=ax.transAxes)

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
