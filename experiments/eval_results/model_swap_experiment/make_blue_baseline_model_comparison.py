import csv
import glob
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval_results"
OUT_DIR = Path(__file__).resolve().parent / "wbs_taskmgr_baseline_selective_comparison_20260426"

BASELINE_CSV = EVAL_DIR / "gemma26_ablation" / "summary_qwen-api_gemma26_ablation_20260423_171707.csv"

EXPERIMENTS = [
    {
        "name": "Gemma4-26B baseline",
        "agent": "All agents",
        "model": "Gemma4-26B",
        "csv": BASELINE_CSV,
        "condition": "C3_3rounds",
        "note": "기준 조건: 모든 에이전트 Gemma4-26B, C3 토의 라운드",
    },
    {
        "name": "WBS Gen Qwen 4B",
        "agent": "WBS Gen",
        "model": "Qwen 4B",
        "csv": EVAL_DIR / "wbsgen_qwen_experiment" / "summary_qwen-api_wbsgen_qwen4b_20260426_040226.csv",
        "condition": None,
        "note": "WBS 생성 에이전트만 Qwen 4B로 교체",
    },
    {
        "name": "WBS Gen EXAONE 7.8B",
        "agent": "WBS Gen",
        "model": "EXAONE 7.8B",
        "csv": EVAL_DIR / "wbsgen_exaone_experiment" / "summary_qwen-api_wbsgen_exaone78b_20260426_051534.csv",
        "condition": None,
        "note": "WBS 생성 에이전트만 EXAONE 7.8B로 교체",
    },
    {
        "name": "Task Manager Qwen 4B",
        "agent": "Task Manager",
        "model": "Qwen 4B",
        "csv": EVAL_DIR / "summary_qwen-api_taskmgr_qwen4b_20260426_160116.csv",
        "condition": None,
        "note": "Task Manager만 Qwen 4B로 교체",
    },
    {
        "name": "Task Manager EXAONE 7.8B",
        "agent": "Task Manager",
        "model": "EXAONE 7.8B",
        "csv_glob": str(EVAL_DIR / "summary_qwen-api_taskmgr_exaone78b_*.csv"),
        "condition": None,
        "note": "Task Manager만 EXAONE 7.8B로 교체",
    },
]

METRIC_KEYS = [
    "elapsed_sec",
    "total_tasks",
    "success_rate",
    "workload_gini",
    "est_tokens",
    "est_cost_usd",
    "autoscore_final",
    "autoscore_allocation",
    "judge_structure",
    "judge_assignment",
    "judge_debate",
    "judge_overall",
]

STYLE = {
    "baseline": "#0B3D91",
    "blue_1": "#1D4ED8",
    "blue_2": "#2563EB",
    "blue_3": "#3B82F6",
    "blue_4": "#60A5FA",
    "muted": "#CBD5E1",
    "muted_dark": "#93C5FD",
    "line": "#E2E8F0",
    "text": "#111827",
    "edge": "#374151",
}

PALETTE_BY_NAME = {
    "Gemma4-26B baseline": STYLE["baseline"],
    "WBS Gen Qwen 4B": STYLE["blue_1"],
    "WBS Gen EXAONE 7.8B": STYLE["blue_2"],
    "Task Manager Qwen 4B": STYLE["blue_3"],
    "Task Manager EXAONE 7.8B": STYLE["blue_4"],
}
HATCH_BY_NAME = {
    "Gemma4-26B baseline": "",
    "WBS Gen Qwen 4B": "",
    "WBS Gen EXAONE 7.8B": "///",
    "Task Manager Qwen 4B": "\\\\\\",
    "Task Manager EXAONE 7.8B": "///",
}
MARKERS = ["o", "s", "D", "^", "P"]


def _set_font():
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
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def _latest(pattern):
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _float(value):
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _read_rows(path, condition=None):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if condition:
        rows = [row for row in rows if row.get("condition") == condition]
    for row in rows:
        for key in METRIC_KEYS:
            row[key] = _float(row.get(key))
        row["run_id"] = int(_float(row.get("run_id"))) if not math.isnan(_float(row.get("run_id"))) else 0
    return rows


def _records():
    records = []
    for exp in EXPERIMENTS:
        csv_path = exp.get("csv")
        if exp.get("csv_glob"):
            csv_path = _latest(exp["csv_glob"])

        record = {
            "name": exp["name"],
            "agent": exp["agent"],
            "model": exp["model"],
            "note": exp["note"],
            "source_csv": "",
            "status": "pending",
            "n_runs": 0,
        }
        if csv_path and Path(csv_path).exists():
            rows = _read_rows(Path(csv_path), exp.get("condition"))
            if rows:
                record["status"] = "done"
                record["n_runs"] = len(rows)
                record["source_csv"] = str(Path(csv_path).relative_to(ROOT))
                valid_rows = [
                    row
                    for row in rows
                    if not np.isnan(row["total_tasks"]) and row["total_tasks"] > 0 and row["success_rate"] > 0
                ]
                record["valid_runs"] = len(valid_rows)
                record["output_valid_rate"] = len(valid_rows) / len(rows)
                for key in METRIC_KEYS:
                    vals = np.array([row[key] for row in rows], dtype=float)
                    record[f"{key}_mean"] = float(np.nanmean(vals))
                    record[f"{key}_std"] = float(np.nanstd(vals, ddof=0))
                record["elapsed_min_mean"] = record["elapsed_sec_mean"] / 60.0
                record["elapsed_min_std"] = record["elapsed_sec_std"] / 60.0
                if record["valid_runs"] == 0:
                    record["invalid_output"] = True
                    record["display_judge_overall_mean"] = 0.0
                    record["display_judge_structure_mean"] = 0.0
                    record["display_judge_assignment_mean"] = np.nan
                    record["display_judge_debate_mean"] = np.nan
                    record["display_autoscore_final_mean"] = 0.0
                    record["display_note"] = "invalid output"
                else:
                    record["invalid_output"] = False
                    record["display_note"] = ""
                    for key in [
                        "judge_overall",
                        "judge_structure",
                        "judge_assignment",
                        "judge_debate",
                        "autoscore_final",
                    ]:
                        record[f"display_{key}_mean"] = record[f"{key}_mean"]
        for key in METRIC_KEYS:
            record.setdefault(f"{key}_mean", np.nan)
            record.setdefault(f"{key}_std", np.nan)
        record.setdefault("elapsed_min_mean", np.nan)
        record.setdefault("elapsed_min_std", np.nan)
        record.setdefault("valid_runs", 0)
        record.setdefault("output_valid_rate", np.nan)
        record.setdefault("invalid_output", False)
        record.setdefault("display_note", "")
        for key in [
            "judge_overall",
            "judge_structure",
            "judge_assignment",
            "judge_debate",
            "autoscore_final",
        ]:
            record.setdefault(f"display_{key}_mean", record.get(f"{key}_mean", np.nan))
        records.append(record)
    return records


def _fmt(value, digits=3):
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _bar_label(ax, bars, values, digits=2, dy=0.012):
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=STYLE["text"],
        )


def _style_axis(ax, title=None):
    if title:
        ax.set_title(title, fontsize=20, fontweight="bold", color=STYLE["text"], pad=14)
    ax.grid(axis="y", color=STYLE["line"], linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", labelsize=13, colors=STYLE["text"])


def _color_for(record):
    return PALETTE_BY_NAME.get(record["name"], STYLE["muted"])


def _hatch_for(record):
    return HATCH_BY_NAME.get(record["name"], "")


def _display_metric(record, metric_key):
    return record.get(f"display_{metric_key}", record.get(metric_key, np.nan))


def _save_aggregate(records):
    keys = [
        "name",
        "agent",
        "model",
        "status",
        "n_runs",
        "valid_runs",
        "output_valid_rate",
        "judge_overall_mean",
        "judge_structure_mean",
        "judge_assignment_mean",
        "judge_debate_mean",
        "autoscore_final_mean",
        "autoscore_allocation_mean",
        "display_judge_overall_mean",
        "display_judge_structure_mean",
        "display_judge_assignment_mean",
        "display_judge_debate_mean",
        "display_autoscore_final_mean",
        "elapsed_min_mean",
        "total_tasks_mean",
        "workload_gini_mean",
        "est_tokens_mean",
        "est_cost_usd_mean",
        "note",
        "display_note",
        "source_csv",
    ]
    with (OUT_DIR / "aggregate_model_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in keys})


def _copy_sources(records):
    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    for record in records:
        if record["source_csv"]:
            src = ROOT / record["source_csv"]
            shutil.copy2(src, raw_dir / src.name)
    for pattern in [
        "experiment_qwen-api_taskmgr_exaone78b_*.json",
        "wbs_snapshot_H_taskmgr_exaone78b_r*_qwen-api_taskmgr_exaone78b_*.json",
    ]:
        for path in sorted(EVAL_DIR.glob(pattern)):
            shutil.copy2(path, raw_dir / path.name)


def _save_score_comparison(records):
    done = [r for r in records if r["status"] == "done"]
    metrics = [
        ("Overall", "judge_overall_mean"),
        ("Structure", "judge_structure_mean"),
        ("Assignment", "judge_assignment_mean"),
        ("Debate", "judge_debate_mean"),
        ("Auto", "autoscore_final_mean"),
    ]
    x = np.arange(len(metrics))
    width = 0.14
    fig, ax = plt.subplots(figsize=(17.5, 8.2), dpi=180)
    for idx, rec in enumerate(done):
        vals = [_display_metric(rec, key) for _, key in metrics]
        pos = x + (idx - (len(done) - 1) / 2) * width
        bars = ax.bar(
            pos,
            vals,
            width=width,
            label=rec["name"],
            color=_color_for(rec),
            edgecolor=STYLE["edge"],
            linewidth=1.1,
            hatch=_hatch_for(rec),
        )
        _bar_label(ax, bars, vals, digits=2)
        if rec.get("invalid_output"):
            ax.text(
                pos[0],
                0.04,
                "invalid\n0 tasks",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#DC2626",
            )
    ax.set_ylim(0, 1.06)
    ax.set_xticks(x, [m[0] for m in metrics], fontsize=15, fontweight="bold")
    ax.set_ylabel("Score", fontsize=15, fontweight="bold", color=STYLE["text"])
    _style_axis(ax, "Baseline 포함 백본 교체 점수 비교")
    ax.text(
        0.01,
        0.91,
        "Blue shades compare baseline and selective agent-model swaps; hatch marks distinguish agent roles.",
        transform=ax.transAxes,
        fontsize=12.5,
        color=STYLE["text"],
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=12, ncols=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_selective_baseline_score_comparison.png")
    plt.close(fig)


def _save_runtime_resource(records):
    done = [r for r in records if r["status"] == "done"]
    panels = [
        ("평균 수행 시간", "elapsed_min_mean", "min/run", 1),
        ("평균 토큰 사용량", "est_tokens_mean", "tokens/run", 0),
        ("평균 비용", "est_cost_usd_mean", "USD/run", 4),
        ("평균 산출 Task 수", "total_tasks_mean", "tasks/run", 1),
    ]
    labels = [r["name"] for r in done]
    x = np.arange(len(done))
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=180)
    for ax, (title, key, ylabel, digits) in zip(axes.flat, panels):
        vals = [r[key] for r in done]
        bars = ax.bar(
            x,
            vals,
            color=[_color_for(r) for r in done],
            edgecolor=STYLE["edge"],
            linewidth=1.1,
        )
        for rec, bar in zip(done, bars):
            bar.set_hatch(_hatch_for(rec))
        offset = max([v for v in vals if not np.isnan(v)] or [1]) * 0.02
        _bar_label(ax, bars, vals, digits=digits, dy=offset)
        ax.set_xticks(x, labels, rotation=15, ha="right", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color=STYLE["text"])
        _style_axis(ax, title)
    fig.suptitle("속도 및 리소스 관점 비교", fontsize=24, fontweight="bold", color=STYLE["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_selective_runtime_resource_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _save_efficiency_scatter(records):
    done = [r for r in records if r["status"] == "done"]
    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=180)
    max_tasks = max([r["total_tasks_mean"] for r in done if not np.isnan(r["total_tasks_mean"])] or [1])
    for idx, rec in enumerate(done):
        size = 230 + 620 * (rec["total_tasks_mean"] / max_tasks)
        ax.scatter(
            rec["elapsed_min_mean"],
            rec["display_judge_overall_mean"],
            s=size,
            c=_color_for(rec),
            marker=MARKERS[idx % len(MARKERS)],
            edgecolor=STYLE["edge"],
            linewidth=1.3,
            alpha=0.9,
            label=rec["name"],
        )
        ax.text(
            rec["elapsed_min_mean"] + 0.12,
            rec["display_judge_overall_mean"] + 0.004,
            rec["name"],
            fontsize=12.5,
            fontweight="bold",
            color=STYLE["text"],
        )
    ax.set_xlabel("Mean elapsed time (min/run)", fontsize=15, fontweight="bold", color=STYLE["text"])
    ax.set_ylabel("Mean LLM judge overall", fontsize=15, fontweight="bold", color=STYLE["text"])
    _style_axis(ax, "성능-속도-산출규모 효율 비교")
    ax.text(
        0.02,
        0.94,
        "Marker size = average task count",
        transform=ax.transAxes,
        fontsize=13,
        color=STYLE["text"],
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_selective_efficiency_scatter.png")
    plt.close(fig)


def _save_assignment_gini(records):
    done = [r for r in records if r["status"] == "done"]
    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=180)
    for idx, rec in enumerate(done):
        if np.isnan(rec["display_judge_assignment_mean"]):
            continue
        ax.scatter(
            rec["workload_gini_mean"],
            rec["display_judge_assignment_mean"],
            s=520,
            c=_color_for(rec),
            marker=MARKERS[idx % len(MARKERS)],
            edgecolor=STYLE["edge"],
            linewidth=1.3,
            alpha=0.9,
            label=rec["name"],
        )
        ax.text(
            rec["workload_gini_mean"] + 0.006,
            rec["display_judge_assignment_mean"] + 0.006,
            rec["name"],
            fontsize=12.5,
            fontweight="bold",
            color=STYLE["text"],
        )
    ax.set_xlabel("Mean workload gini (lower is more balanced)", fontsize=15, fontweight="bold", color=STYLE["text"])
    ax.set_ylabel("Mean judge assignment", fontsize=15, fontweight="bold", color=STYLE["text"])
    _style_axis(ax, "배정 품질과 업무 편중도 비교")
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_selective_assignment_vs_workload_gini.png")
    plt.close(fig)


def _role_subset(records, agent):
    return [
        r
        for r in records
        if r["status"] == "done" and (r["agent"] == "All agents" or r["agent"] == agent)
    ]


def _save_role_score(records, agent, file_prefix, title):
    subset = _role_subset(records, agent)
    metrics = [
        ("Overall", "judge_overall_mean"),
        ("Structure", "judge_structure_mean"),
        ("Assignment", "judge_assignment_mean"),
        ("Auto", "autoscore_final_mean"),
    ]
    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(15.5, 7.8), dpi=180)
    for idx, rec in enumerate(subset):
        vals = [_display_metric(rec, key) for _, key in metrics]
        pos = x + (idx - (len(subset) - 1) / 2) * width
        bars = ax.bar(
            pos,
            vals,
            width=width,
            label=rec["name"],
            color=_color_for(rec),
            edgecolor=STYLE["edge"],
            linewidth=1.1,
            hatch=_hatch_for(rec),
        )
        _bar_label(ax, bars, vals, digits=2)
        if rec.get("invalid_output"):
            ax.text(
                pos[0],
                0.04,
                "invalid\n0 tasks",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#DC2626",
            )
    ax.set_ylim(0, 1.06)
    ax.set_xticks(x, [m[0] for m in metrics], fontsize=15, fontweight="bold")
    ax.set_ylabel("Score", fontsize=15, fontweight="bold", color=STYLE["text"])
    _style_axis(ax, title)
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "role_separated" / f"{file_prefix}_score_comparison.png")
    plt.close(fig)


def _save_role_resource(records, agent, file_prefix, title):
    subset = _role_subset(records, agent)
    panels = [
        ("수행 시간", "elapsed_min_mean", "min/run", 1),
        ("토큰 사용량", "est_tokens_mean", "tokens/run", 0),
        ("비용", "est_cost_usd_mean", "USD/run", 4),
        ("Task 수", "total_tasks_mean", "tasks/run", 1),
    ]
    labels = [r["name"] for r in subset]
    x = np.arange(len(subset))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), dpi=180)
    for ax, (panel_title, key, ylabel, digits) in zip(axes.flat, panels):
        vals = [r[key] for r in subset]
        bars = ax.bar(
            x,
            vals,
            color=[_color_for(r) for r in subset],
            edgecolor=STYLE["edge"],
            linewidth=1.1,
        )
        for rec, bar in zip(subset, bars):
            bar.set_hatch(_hatch_for(rec))
        offset = max([v for v in vals if not np.isnan(v)] or [1]) * 0.02
        _bar_label(ax, bars, vals, digits=digits, dy=offset)
        ax.set_xticks(x, labels, rotation=12, ha="right", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color=STYLE["text"])
        _style_axis(ax, panel_title)
    fig.suptitle(title, fontsize=23, fontweight="bold", color=STYLE["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "role_separated" / f"{file_prefix}_runtime_resource.png", bbox_inches="tight")
    plt.close(fig)


def _save_role_efficiency(records, agent, file_prefix, title):
    subset = _role_subset(records, agent)
    fig, ax = plt.subplots(figsize=(12.8, 7.6), dpi=180)
    max_tasks = max([r["total_tasks_mean"] for r in subset if not np.isnan(r["total_tasks_mean"])] or [1])
    for idx, rec in enumerate(subset):
        size = 260 + 720 * (rec["total_tasks_mean"] / max_tasks)
        ax.scatter(
            rec["elapsed_min_mean"],
            rec["display_judge_overall_mean"],
            s=size,
            c=_color_for(rec),
            marker=MARKERS[idx % len(MARKERS)],
            edgecolor=STYLE["edge"],
            linewidth=1.3,
            alpha=0.92,
            label=rec["name"],
        )
        ax.text(
            rec["elapsed_min_mean"] + 0.12,
            rec["display_judge_overall_mean"] + 0.004,
            rec["name"],
            fontsize=12.5,
            fontweight="bold",
            color=STYLE["text"],
        )
    ax.set_xlabel("Mean elapsed time (min/run)", fontsize=15, fontweight="bold", color=STYLE["text"])
    ax.set_ylabel("Mean LLM judge overall", fontsize=15, fontweight="bold", color=STYLE["text"])
    _style_axis(ax, title)
    ax.text(
        0.02,
        0.94,
        "Marker size = average task count",
        transform=ax.transAxes,
        fontsize=13,
        color=STYLE["text"],
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "role_separated" / f"{file_prefix}_efficiency_scatter.png")
    plt.close(fig)


def _save_role_separated_figures(records):
    role_dir = OUT_DIR / "role_separated"
    role_dir.mkdir(exist_ok=True)

    _save_role_score(records, "WBS Gen", "wbsgen", "WBS Gen 백본 교체 점수 비교")
    _save_role_resource(records, "WBS Gen", "wbsgen", "WBS Gen 백본 교체 리소스 비교")
    _save_role_efficiency(records, "WBS Gen", "wbsgen", "WBS Gen 성능-속도 효율")

    _save_role_score(records, "Task Manager", "taskmanager", "Task Manager 백본 교체 점수 비교")
    _save_role_resource(records, "Task Manager", "taskmanager", "Task Manager 백본 교체 리소스 비교")
    _save_role_efficiency(records, "Task Manager", "taskmanager", "Task Manager 성능-속도 효율")

    md = """# Role-Separated Figures

## WBS Gen

- `wbsgen_score_comparison.png`: Gemma4-26B baseline vs WBS Gen Qwen/EXAONE 점수 비교
- `wbsgen_runtime_resource.png`: WBS Gen 교체 조건의 시간, 토큰, 비용, task 수 비교
- `wbsgen_efficiency_scatter.png`: WBS Gen 교체 조건의 성능-속도-산출규모 효율 비교

## Task Manager

- `taskmanager_score_comparison.png`: Gemma4-26B baseline vs Task Manager Qwen/EXAONE 점수 비교
- `taskmanager_runtime_resource.png`: Task Manager 교체 조건의 시간, 토큰, 비용, task 수 비교
- `taskmanager_efficiency_scatter.png`: Task Manager 교체 조건의 성능-속도-산출규모 효율 비교
"""
    (role_dir / "SUMMARY.md").write_text(md, encoding="utf-8")


def _save_summary(records):
    rows = "\n".join(
        "| {name} | {runs} | {overall} | {structure} | {assignment} | {debate} | {auto} | {time} | {tokens} | {cost} | {tasks} | {gini} |".format(
            name=r["name"],
            runs=r["n_runs"] if r["status"] == "done" else "-",
            overall=_fmt(r["display_judge_overall_mean"]),
            structure=_fmt(r["display_judge_structure_mean"]),
            assignment=_fmt(r["display_judge_assignment_mean"]),
            debate=_fmt(r["display_judge_debate_mean"]),
            auto=_fmt(r["display_autoscore_final_mean"]),
            time=_fmt(r["elapsed_min_mean"], 1),
            tokens=_fmt(r["est_tokens_mean"], 0),
            cost=_fmt(r["est_cost_usd_mean"], 4),
            tasks=_fmt(r["total_tasks_mean"], 1),
            gini=_fmt(r["workload_gini_mean"]),
        )
        for r in records
    )
    done = [r for r in records if r["status"] == "done"]
    best_overall = max(done, key=lambda r: r["display_judge_overall_mean"]) if done else None
    valid_done = [r for r in done if not r.get("invalid_output")]
    fastest = min(valid_done, key=lambda r: r["elapsed_min_mean"]) if valid_done else None

    md = f"""# Baseline-Inclusive Model Swap Comparison

## 요약 테이블

| Condition | Runs | Judge Overall | Structure | Assignment | Debate | Auto | Time min | Tokens | Cost USD | Tasks | Workload Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## 해석 포인트

- 기준선은 `Gemma4-26B baseline`으로 표기했다. 이는 C3 조건에서 모든 에이전트를 Gemma4-26B로 둔 결과다.
- 현재 최고 judge overall은 `{best_overall['name'] if best_overall else '-'}`의 `{_fmt(best_overall['display_judge_overall_mean'] if best_overall else np.nan)}`이다.
- 현재 가장 빠른 완료 조건은 `{fastest['name'] if fastest else '-'}`의 `{_fmt(fastest['elapsed_min_mean'] if fastest else np.nan, 1)} min/run`이다.
- Task Manager EXAONE은 점수가 나오더라도 raw output의 JSON 주석, 미허용 assignee, 일부 L3 미배정 같은 schema-following 문제를 같이 해석해야 한다.
- `0 tasks`처럼 유효 WBS를 만들지 못한 조건은 figure에서 표시용 점수를 0 또는 N/A로 보정했다. 원본 judge 값은 CSV에 보존하되 정상 산출물 점수로 해석하지 않는다.
- `Debate` 점수는 교체하지 않은 토론 에이전트의 영향도 포함하므로, 백본 선택 근거에서는 `Overall`, `Assignment`, `Time`, `Token/Cost`, `Workload Gini`를 함께 보는 편이 더 타당하다.

## Figure

- `fig1_selective_baseline_score_comparison.png`: baseline 포함 품질 점수 비교
- `fig2_selective_runtime_resource_comparison.png`: 수행 시간, 토큰, 비용, 산출 task 수 비교
- `fig3_selective_efficiency_scatter.png`: 성능-속도-산출규모 효율 비교
- `fig4_selective_assignment_vs_workload_gini.png`: 배정 점수와 업무 편중도 비교
- `role_separated/`: WBS Gen 실험과 Task Manager 실험을 분리한 figure 세트

원본 CSV와 EXAONE Task Manager raw JSON/snapshot은 `raw/`에 복사했다.
"""
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")


def main():
    _set_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _records()
    _save_aggregate(records)
    _copy_sources(records)
    _save_score_comparison(records)
    _save_runtime_resource(records)
    _save_efficiency_scatter(records)
    _save_assignment_gini(records)
    _save_role_separated_figures(records)
    _save_summary(records)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
