"""Build a slide-style summary table from comparison_4backbones Figure 5 data.

The values are centroids over C2/C3/C5:
- x-axis of Figure 5: LLM Judge mean
- y-axis of Figure 5: AutoScore mean
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, patches


HERE = Path(__file__).resolve().parent
SRC = HERE / "summary_4backbones.csv"
OUT_DIR = HERE / "figure5"
OUT_DIR.mkdir(exist_ok=True)

CONDS = {"C2_1round", "C3_3rounds", "C5_5rounds"}
SYNTH_SRC = HERE / "qwen35_a3b_synthetic_summary.csv"
ORDER = ["gemma26", "qwen35_a3b", "qwen", "gemini", "gemma"]
LABELS = {
    "gemma26": "Gemma-\n4-26B-\nA4B",
    "qwen35_a3b": "Qwen3.5-\n35B-\nA3B",
    "qwen": "Qwen3-\n14B",
    "gemini": "Gemini 3.1\nFlash Lite",
    "gemma": "Gemma-4-\nE4B-it",
}


def _set_font() -> None:
    for path in [
        "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def _read_centroids() -> dict[str, dict[str, float]]:
    values = {bb: {"llm": [], "auto": []} for bb in ORDER}
    with SRC.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            bb = row["backbone"]
            if bb not in values or row["condition"] not in CONDS:
                continue
            values[bb]["llm"].append(float(row["Overall_mean"]))
            values[bb]["auto"].append(float(row["AutoOverall_mean"]))
    if SYNTH_SRC.exists():
        with SYNTH_SRC.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["backbone"] != "qwen35_a3b" or row["condition"] not in CONDS:
                    continue
                values["qwen35_a3b"]["llm"].append(float(row["Overall_mean_LLM"]))
                values["qwen35_a3b"]["auto"].append(float(row["AutoOverall_mean"]))
    return {
        bb: {
            "llm": sum(values[bb]["llm"]) / len(values[bb]["llm"]),
            "auto": sum(values[bb]["auto"]) / len(values[bb]["auto"]),
        }
        for bb in ORDER
    }


def _save_csv(centroids: dict[str, dict[str, float]]) -> None:
    with (OUT_DIR / "fig5_centroid_summary_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *[LABELS[bb].replace("\n", " ") for bb in ORDER]])
        writer.writerow(["AutoScore 평균 점수", *[f"{centroids[bb]['auto']:.3f}" for bb in ORDER]])
        writer.writerow(["LLM Judge 평균 점수", *[f"{centroids[bb]['llm']:.3f}" for bb in ORDER]])


def _draw_table(centroids: dict[str, dict[str, float]]) -> None:
    _set_font()

    fig, ax = plt.subplots(figsize=(12.6, 4.2), dpi=180)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)
    ax.axis("off")

    header_bg = "#EEF1FF"
    row_label_bg = "#F7F8FF"
    cell_bg = "#FFFFFF"
    grid = "#D1D5DB"
    text = "#111827"
    accent = "#2563EB"
    focus_border = "#DC2626"

    widths = [0.92, 1.02, 1.08, 1.02, 1.08, 1.02]
    xs = [0]
    for width in widths[:-1]:
        xs.append(xs[-1] + width)
    heights = [1.05, 0.98, 0.97]
    ys = [3 - heights[0], 3 - heights[0] - heights[1], 0]

    # Header row.
    for col, x in enumerate(xs):
        ax.add_patch(
            patches.Rectangle(
                (x, ys[0]),
                widths[col],
                heights[0],
                facecolor=header_bg,
                edgecolor=grid,
                linewidth=1.1,
            )
        )
    for i, bb in enumerate(ORDER, start=1):
        ax.text(
            xs[i] + widths[i] / 2,
            ys[0] + heights[0] / 2,
            LABELS[bb],
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=text,
            linespacing=1.05,
        )

    rows = [
        ("AutoScore\n평균 점수", "auto"),
        ("LLM Judge\n평균 점수", "llm"),
    ]
    for r, (label, key) in enumerate(rows, start=1):
        y = ys[r]
        for col, x in enumerate(xs):
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    widths[col],
                    heights[r],
                    facecolor=row_label_bg if col == 0 else cell_bg,
                    edgecolor=grid,
                    linewidth=1.1,
                )
            )
        ax.text(
            xs[0] + widths[0] / 2,
            y + heights[r] / 2,
            label,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=text,
            linespacing=1.15,
        )
        for i, bb in enumerate(ORDER, start=1):
            color = accent if bb == "gemma26" else text
            ax.text(
                xs[i] + widths[i] / 2,
                y + heights[r] / 2,
                f"{centroids[bb][key]:.3f}",
                ha="center",
                va="center",
                fontsize=19,
                fontweight="bold" if bb == "gemma26" else "normal",
                color=color,
            )

    # Focus outline for the selected local backbone column.
    ax.add_patch(
        patches.Rectangle(
            (xs[1], 0),
            widths[1],
            3,
            facecolor="none",
            edgecolor=focus_border,
            linewidth=3.4,
            zorder=10,
        )
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "fig5_centroid_summary_table.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    centroids = _read_centroids()
    _save_csv(centroids)
    _draw_table(centroids)
    print(OUT_DIR / "fig5_centroid_summary_table.png")


if __name__ == "__main__":
    main()
