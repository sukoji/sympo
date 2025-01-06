import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval_results"
OUT_DIR = Path(__file__).resolve().parent / "wbs_taskmgr_model_comparison_20260426"

EXPERIMENTS = [
    {
        "agent": "Baseline",
        "model": "Gemma4-26B baseline",
        "status": "done",
        "csv": EVAL_DIR / "gemma26_ablation" / "summary_qwen-api_gemma26_ablation_20260423_171707.csv",
        "condition": "C3_3rounds",
        "note": "모든 에이전트 Gemma4-26B, C3 조건",
    },
    {
        "agent": "WBS Gen",
        "model": "Qwen 4B",
        "status": "done",
        "csv": EVAL_DIR / "wbsgen_qwen_experiment" / "summary_qwen-api_wbsgen_qwen4b_20260426_040226.csv",
        "condition": None,
        "note": "WBS Gen만 Qwen 4B, 나머지는 Gemma 26B",
    },
    {
        "agent": "WBS Gen",
        "model": "EXAONE 7.8B",
        "status": "done",
        "csv": EVAL_DIR / "wbsgen_exaone_experiment" / "summary_qwen-api_wbsgen_exaone78b_20260426_051534.csv",
        "condition": None,
        "note": "WBS Gen만 EXAONE 7.8B, 나머지는 Gemma 26B",
    },
    {
        "agent": "Task Manager",
        "model": "Qwen 4B",
        "status": "done",
        "csv": EVAL_DIR / "summary_qwen-api_taskmgr_qwen4b_20260426_160116.csv",
        "condition": None,
        "note": "Task Manager만 Qwen 4B, 나머지는 Gemma 26B",
    },
    {
        "agent": "Task Manager",
        "model": "EXAONE 7.8B",
        "status": "done",
        "csv": EVAL_DIR / "summary_qwen-api_taskmgr_exaone78b_20260426_180428.csv",
        "condition": None,
        "note": "Task Manager만 EXAONE 7.8B, 나머지는 Gemma 26B",
    },
    {
        "agent": "WBS Gen",
        "model": "Qwen LoRA",
        "status": "pending",
        "csv": None,
        "note": "8083/8084 서버 정상화 후 실행 예정",
    },
    {
        "agent": "Task Manager",
        "model": "Qwen LoRA",
        "status": "pending",
        "csv": None,
        "note": "8083/8084 서버 정상화 후 실행 예정",
    },
]

METRIC_KEYS = [
    "elapsed_sec",
    "total_tasks",
    "autoscore_final",
    "autoscore_allocation",
    "workload_gini",
    "judge_structure",
    "judge_assignment",
    "judge_debate",
    "judge_overall",
]


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


def _read_csv(path, condition=None):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if condition:
        rows = [row for row in rows if row.get("condition") == condition]
    for row in rows:
        for key in METRIC_KEYS:
            row[key] = float(row[key]) if row.get(key) else np.nan
        row["run_id"] = int(float(row["run_id"])) if row.get("run_id") else 0
    return rows


def _aggregate():
    records = []
    for exp in EXPERIMENTS:
        record = {
            "agent": exp["agent"],
            "model": exp["model"],
            "status": exp["status"],
            "note": exp["note"],
            "n_runs": 0,
        }
        if exp["status"] == "done":
            rows = _read_csv(exp["csv"], exp.get("condition"))
            record["n_runs"] = len(rows)
            for key in METRIC_KEYS:
                vals = np.array([r[key] for r in rows], dtype=float)
                record[f"{key}_mean"] = float(np.nanmean(vals))
                record[f"{key}_std"] = float(np.nanstd(vals, ddof=0))
            record["elapsed_min_mean"] = record["elapsed_sec_mean"] / 60.0
            record["source_csv"] = str(exp["csv"].relative_to(ROOT))
        else:
            for key in METRIC_KEYS:
                record[f"{key}_mean"] = np.nan
                record[f"{key}_std"] = np.nan
            record["elapsed_min_mean"] = np.nan
            record["source_csv"] = ""
        records.append(record)
    return records


def _fmt(value, digits=3):
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _write_stats_csv(records):
    keys = [
        "agent",
        "model",
        "status",
        "n_runs",
        "judge_overall_mean",
        "judge_structure_mean",
        "judge_assignment_mean",
        "judge_debate_mean",
        "autoscore_final_mean",
        "autoscore_allocation_mean",
        "workload_gini_mean",
        "total_tasks_mean",
        "elapsed_min_mean",
        "note",
        "source_csv",
    ]
    with (OUT_DIR / "aggregate_model_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in keys})


def _copy_sources():
    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    for exp in EXPERIMENTS:
        if exp["status"] == "done" and exp["csv"]:
            shutil.copy2(exp["csv"], raw_dir / exp["csv"].name)


def _save_matrix(records):
    agents = ["WBS Gen", "Task Manager"]
    models = ["Gemma4-26B baseline", "Qwen 4B", "EXAONE 7.8B", "Qwen LoRA"]
    judge_data = np.full((len(agents), len(models)), np.nan)
    auto_data = np.full((len(agents), len(models)), np.nan)
    status = [["pending" for _ in models] for _ in agents]
    baseline = next((r for r in records if r["agent"] == "Baseline"), None)
    if baseline:
        for i in range(len(agents)):
            judge_data[i, 0] = baseline["judge_overall_mean"]
            auto_data[i, 0] = baseline["autoscore_final_mean"]
            status[i][0] = "done"
    for rec in records:
        if rec["agent"] in agents and rec["model"] in models:
            i = agents.index(rec["agent"])
            j = models.index(rec["model"])
            status[i][j] = rec["status"]
            judge_data[i, j] = rec["judge_overall_mean"]
            auto_data[i, j] = rec["autoscore_final_mean"]

    fig, ax = plt.subplots(figsize=(15.6, 6.4), dpi=180)
    im = ax.imshow(judge_data, cmap="YlGnBu", vmin=0.55, vmax=0.82)
    ax.set_xticks(np.arange(len(models)), models, fontsize=14, fontweight="bold")
    ax.set_yticks(np.arange(len(agents)), agents, fontsize=14, fontweight="bold")
    ax.set_title("WBS / Task Manager 백본 교체 실험 현황", fontsize=21, fontweight="bold", pad=18)
    ax.text(
        0.0,
        1.05,
        "cell format: LLM Judge / AutoScore",
        transform=ax.transAxes,
        fontsize=12.5,
        color="#4B5563",
        fontweight="bold",
    )

    for i in range(len(agents)):
        for j in range(len(models)):
            if status[i][j] == "done":
                ax.text(
                    j,
                    i,
                    f"{judge_data[i, j]:.3f}\nAuto {auto_data[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="#111827",
                )
            else:
                ax.text(j, i, "pending", ha="center", va="center", fontsize=14, fontweight="bold", color="#6B7280")
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#9CA3AF", linewidth=0))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="LLM Judge Overall")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_agent_model_completion_matrix.png")
    plt.close(fig)


def _save_wbs_comparison(records):
    baseline = next(r for r in records if r["agent"] == "Baseline")
    wbs = [baseline] + [r for r in records if r["agent"] == "WBS Gen" and r["status"] == "done" and r["model"] in ("Qwen 4B", "EXAONE 7.8B")]
    labels = [
        "Gemma4-26B\nBaseline" if r["agent"] == "Baseline" else r["model"]
        for r in wbs
    ]
    judge_vals = [r["judge_overall_mean"] for r in wbs]
    auto_vals = [r["autoscore_final_mean"] for r in wbs]

    x = np.arange(len(wbs))
    width = 0.28
    judge_color = "#63A7F5"
    auto_color = "#2563EB"

    fig, ax = plt.subplots(figsize=(12.2, 6.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars_judge = ax.bar(
        x - width / 2,
        judge_vals,
        width=width,
        label="LLM Judge",
        color=judge_color,
        edgecolor="#2563EB",
        linewidth=0.9,
    )
    bars_auto = ax.bar(
        x + width / 2,
        auto_vals,
        width=width,
        label="AutoScore",
        color=auto_color,
        edgecolor="#1D4ED8",
        linewidth=0.9,
    )

    for bars, vals in ((bars_judge, judge_vals), (bars_auto, auto_vals)):
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.017,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color="#1F2937",
            )

    ax.set_ylim(0.56, 0.86)
    ax.set_xticks(x, labels, fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Mean Score", fontsize=13, fontweight="bold", color="#111827")
    ax.set_title("WBS Gen 백본 모델 점수 비교", fontsize=19, fontweight="bold", color="#111827", pad=18)
    ax.legend(
        frameon=False,
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.99),
        handlelength=1.8,
    )
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=11.5, colors="#374151")
    ax.tick_params(axis="x", colors="#111827", pad=7)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_wbsgen_model_score_comparison.png")
    plt.close(fig)


def _save_taskmgr_comparison(records):
    baseline = next(r for r in records if r["agent"] == "Baseline")
    task = [baseline] + [r for r in records if r["agent"] == "Task Manager" and r["status"] == "done" and r["model"] in ("Qwen 4B", "EXAONE 7.8B")]
    labels = [r["model"] for r in task]
    metrics = [
        ("Overall", "judge_overall_mean"),
        ("Structure", "judge_structure_mean"),
        ("Assignment", "judge_assignment_mean"),
        ("Auto", "autoscore_final_mean"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    colors = ["#1D4ED8", "#94A3B8", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=180)
    for idx, rec in enumerate(task):
        vals = [rec[key] for _, key in metrics]
        offset = (idx - (len(task) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=labels[idx], color=colors[idx], edgecolor="#111827")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [m[0] for m in metrics], fontsize=14, fontweight="bold")
    ax.set_ylabel("score", fontsize=13, fontweight="bold")
    ax.set_title("Task Manager 백본 비교: Gemma baseline vs Qwen 4B vs EXAONE 7.8B", fontsize=19, fontweight="bold", pad=18)
    ax.text(0.02, 0.93, "Baseline은 모든 에이전트 Gemma4-26B, 나머지는 Task Manager만 교체", transform=ax.transAxes, fontsize=12, color="#4B5563")
    ax.legend(frameon=False, fontsize=12)
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_taskmanager_model_score_comparison.png")
    plt.close(fig)


def _save_qwen_role_comparison(records):
    qwen = [r for r in records if r["model"] == "Qwen 4B" and r["status"] == "done"]
    metrics = [
        ("Overall", "judge_overall_mean"),
        ("Assignment", "judge_assignment_mean"),
        ("Tasks/100", "total_tasks_mean"),
        ("Time/20m", "elapsed_min_mean"),
    ]
    labels = [r["agent"] for r in qwen]
    colors = ["#2563EB", "#F59E0B"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=180)
    for idx, rec in enumerate(qwen):
        vals = [
            rec["judge_overall_mean"],
            rec["judge_assignment_mean"],
            rec["total_tasks_mean"] / 100.0,
            rec["elapsed_min_mean"] / 20.0,
        ]
        bars = ax.bar(x + (idx - 0.5) * width, vals, width=width, label=labels[idx], color=colors[idx], edgecolor="#111827")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x, [m[0] for m in metrics], fontsize=14, fontweight="bold")
    ax.set_title("동일 Qwen 4B의 교체 위치별 영향", fontsize=21, fontweight="bold", pad=18)
    ax.text(0.02, 0.93, "WBS Gen 교체는 생성 구조 품질, Task Manager 교체는 배정 품질을 주로 흔듦", transform=ax.transAxes, fontsize=12, color="#4B5563")
    ax.legend(frameon=False, fontsize=12)
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_qwen4b_role_swap_comparison.png")
    plt.close(fig)


def _save_efficiency_scatter(records):
    done = [r for r in records if r["status"] == "done"]
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=180)
    colors = {"Baseline": "#1D4ED8", "WBS Gen": "#10B981", "Task Manager": "#F59E0B"}
    markers = {"Gemma4-26B baseline": "*", "Qwen 4B": "o", "EXAONE 7.8B": "s"}
    for rec in done:
        ax.scatter(
            rec["elapsed_min_mean"],
            rec["judge_overall_mean"],
            s=rec["total_tasks_mean"] * 9,
            c=colors.get(rec["agent"], "#6B7280"),
            marker=markers.get(rec["model"], "o"),
            edgecolor="#111827",
            linewidth=1.2,
            alpha=0.86,
        )
        ax.text(
            rec["elapsed_min_mean"] + 0.08,
            rec["judge_overall_mean"] + 0.006,
            f"{rec['agent']} / {rec['model']}",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xlim(6.5, 20.0)
    ax.set_ylim(0.58, 0.81)
    ax.set_xlabel("mean elapsed time (min/run)", fontsize=13, fontweight="bold")
    ax.set_ylabel("mean judge overall", fontsize=13, fontweight="bold")
    ax.set_title("성능-시간-산출규모 트레이드오프", fontsize=21, fontweight="bold", pad=18)
    ax.text(0.02, 0.93, "원 크기 = 평균 task 수", transform=ax.transAxes, fontsize=12, color="#4B5563")
    ax.grid(color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_efficiency_tradeoff_scatter.png")
    plt.close(fig)


def _write_summary(records):
    done_rows = "\n".join(
        "| {agent} | {model} | {n_runs} | {overall} | {structure} | {assignment} | {debate} | {auto} | {tasks} | {time} | {gini} |".format(
            agent=r["agent"],
            model=r["model"],
            n_runs=r["n_runs"],
            overall=_fmt(r["judge_overall_mean"]),
            structure=_fmt(r["judge_structure_mean"]),
            assignment=_fmt(r["judge_assignment_mean"]),
            debate=_fmt(r["judge_debate_mean"]),
            auto=_fmt(r["autoscore_final_mean"]),
            tasks=_fmt(r["total_tasks_mean"], 1),
            time=_fmt(r["elapsed_min_mean"], 1),
            gini=_fmt(r["workload_gini_mean"]),
        )
        for r in records
        if r["status"] == "done"
    )
    pending_rows = "\n".join(
        f"| {r['agent']} | {r['model']} | {r['note']} |"
        for r in records
        if r["status"] != "done"
    )
    wbs_qwen = next(r for r in records if r["agent"] == "WBS Gen" and r["model"] == "Qwen 4B")
    wbs_exaone = next(r for r in records if r["agent"] == "WBS Gen" and r["model"] == "EXAONE 7.8B")
    task_qwen = next(r for r in records if r["agent"] == "Task Manager" and r["model"] == "Qwen 4B")
    task_exaone = next(r for r in records if r["agent"] == "Task Manager" and r["model"] == "EXAONE 7.8B")
    baseline = next(r for r in records if r["agent"] == "Baseline")

    md = f"""# WBS / Task Manager Backbone Model Comparison

## 범위

이 폴더는 현재 완료된 백본 교체 실험을 모델별, 에이전트별로 비교한다. 모든 완료 실험은 `C3_3rounds` 조건을 유지하고, 지정된 agent만 해당 모델로 교체했다.
`Gemma4-26B baseline`은 모든 에이전트를 Gemma4-26B로 둔 기준선이다.

## 완료 실험 요약

| Agent | Model | Runs | Judge Overall | Structure | Assignment | Debate | Auto | Tasks | Time min | Workload Gini |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{done_rows}

## 아직 미완료/대기 조건

| Agent | Model | 상태 |
|---|---|---|
{pending_rows}

## 해석

- WBS Gen 교체에서는 **Qwen 4B 평균 overall {_fmt(wbs_qwen['judge_overall_mean'])}**가 **EXAONE 7.8B 평균 overall {_fmt(wbs_exaone['judge_overall_mean'])}**보다 높다.
- 그러나 Gemma4-26B baseline overall **{_fmt(baseline['judge_overall_mean'])}**과 비교하면 WBS Gen 교체 모델 둘 다 최종 overall을 넘지 못했다.
- EXAONE WBS는 평균 task 수가 **{_fmt(wbs_exaone['total_tasks_mean'], 1)}**로 작고, 실험 로그상 `assigned_role` 타입 오류와 L1/L2 구조 부족이 반복됐다.
- Qwen 4B WBS는 평균 task 수가 **{_fmt(wbs_qwen['total_tasks_mean'], 1)}**로 크지만 run 간 분산이 크다. 생성량은 많지만 안정성 해석이 필요하다.
- Task Manager-Qwen 4B는 평균 overall **{_fmt(task_qwen['judge_overall_mean'])}**로 수치상 양호하다. 다만 병목은 assignment 평균 **{_fmt(task_qwen['judge_assignment_mean'])}**이며, judge reason상 skill-fit과 workload imbalance가 반복적으로 지적된다.
- Task Manager-EXAONE 7.8B는 overall **{_fmt(task_exaone['judge_overall_mean'])}**로 Qwen 4B와 유사하지만, baseline보다 느리고 assignment도 baseline보다 낮다.

## Figure

- `fig1_agent_model_completion_matrix.png`: agent-model 완료/대기 매트릭스
- `fig2_wbsgen_model_score_comparison.png`: Gemma baseline, Qwen 4B, EXAONE 7.8B의 WBS Gen 비교
- `fig3_taskmanager_model_score_comparison.png`: Gemma baseline, Qwen 4B, EXAONE 7.8B의 Task Manager 비교
- `fig4_qwen4b_role_swap_comparison.png`: Qwen 4B를 WBS Gen에 넣었을 때와 Task Manager에 넣었을 때 비교
- `fig5_efficiency_tradeoff_scatter.png`: 평균 성능, 소요 시간, task 규모 비교

## Raw

원본 summary CSV는 `raw/` 폴더에 복사했다. 집계 테이블은 `aggregate_model_comparison.csv`에 저장했다.
"""
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")


def main():
    _set_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _aggregate()
    _write_stats_csv(records)
    _copy_sources()
    _save_matrix(records)
    _save_wbs_comparison(records)
    _save_taskmgr_comparison(records)
    _save_qwen_role_comparison(records)
    _save_efficiency_scatter(records)
    _write_summary(records)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
