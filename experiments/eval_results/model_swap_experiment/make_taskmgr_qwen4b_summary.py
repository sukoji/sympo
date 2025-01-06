import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval_results"
CSV_PATH = EVAL_DIR / "summary_qwen-api_taskmgr_qwen4b_20260426_160116.csv"
SNAPSHOT_GLOB = "wbs_snapshot_H_taskmgr_qwen4b_r*_qwen-api_taskmgr_qwen4b_*.json"
OUT_DIR = Path(__file__).resolve().parent / "taskmgr_qwen4b_summary_20260426_160116"


def _set_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def _read_rows():
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key in {
                "run_id",
                "elapsed_sec",
                "total_tasks",
                "interaction_turns",
                "est_tokens",
            }:
                row[key] = int(float(value)) if value else 0
            elif key in {
                "judge_structure",
                "judge_assignment",
                "judge_debate",
                "judge_overall",
                "autoscore_final",
                "autoscore_quality",
                "autoscore_allocation",
                "autoscore_orchestration",
                "workload_gini",
                "planning_score",
                "mece_score",
                "granularity_fitness",
                "comm_efficiency",
                "est_cost_usd",
            }:
                row[key] = float(value) if value else np.nan
    return rows


def _read_snapshots():
    snapshots = []
    for path in sorted(EVAL_DIR.glob(SNAPSHOT_GLOB)):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        snapshots.append((path, data))
    return snapshots


def _mean(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmean(vals))


def _std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanstd(vals, ddof=0))


def _member_loads(snapshot):
    members = {m["member_id"]: m["name"] for m in snapshot.get("team_members", [])}
    loads = {mid: 0.0 for mid in members}
    for task in snapshot.get("wbs_tasks", []):
        if task.get("level") != "L3":
            continue
        assignees = task.get("assigned_to") or []
        if not assignees:
            continue
        share = float(task.get("estimated_days") or 0) / len(assignees)
        for mid in assignees:
            loads[mid] = loads.get(mid, 0.0) + share
    return {members.get(mid, mid): load for mid, load in loads.items()}


def _save_score_profile(rows):
    labels = ["Structure", "Assignment", "Debate", "Overall"]
    keys = ["judge_structure", "judge_assignment", "judge_debate", "judge_overall"]
    means = [_mean(rows, k) for k in keys]
    stds = [_std(rows, k) for k in keys]
    colors = ["#2F80ED", "#EB5757", "#27AE60", "#111827"]

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=180)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.62, edgecolor="#1F2937", linewidth=0.8)
    ax.axhspan(0.75, 1.0, color="#E8F5EF", zorder=0)
    ax.axhline(0.75, color="#16A34A", linestyle="--", linewidth=1.4)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, labels, fontsize=15, fontweight="bold")
    ax.set_ylabel("LLM Judge Score", fontsize=14, fontweight="bold")
    ax.set_title("Task Manager 백본 교체: Qwen 4B 평가 프로파일", fontsize=21, fontweight="bold", pad=18)
    ax.text(0.02, 0.94, "C3 조건 유지: WBS/토론/최종정리 = Gemma 26B, Task Manager만 Qwen 4B", transform=ax.transAxes, fontsize=12, color="#4B5563")
    ax.text(1, means[1] + 0.08, "병목: 배정 품질", ha="center", fontsize=13, color="#B91C1C", fontweight="bold")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_qwen4b_taskmanager_score_profile.png")
    plt.close(fig)


def _save_run_dashboard(rows):
    runs = [r["run_id"] for r in rows]
    elapsed_min = [r["elapsed_sec"] / 60 for r in rows]
    overall = [r["judge_overall"] for r in rows]
    assignment = [r["judge_assignment"] for r in rows]
    tasks = [r["total_tasks"] for r in rows]
    gini = [r["workload_gini"] for r in rows]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=180)
    fig.suptitle("Qwen 4B Task Manager: 3회 반복 안정성 점검", fontsize=21, fontweight="bold", y=0.98)

    axs[0, 0].plot(runs, overall, marker="o", linewidth=3, color="#111827", label="Overall")
    axs[0, 0].plot(runs, assignment, marker="o", linewidth=3, color="#EB5757", label="Assignment")
    axs[0, 0].set_ylim(0.55, 0.85)
    axs[0, 0].set_xticks(runs)
    axs[0, 0].set_title("Judge 점수 추이", fontsize=15, fontweight="bold")
    axs[0, 0].legend(frameon=False)

    axs[0, 1].bar(runs, elapsed_min, color="#6B7280", edgecolor="#111827")
    axs[0, 1].set_title("실험 소요 시간", fontsize=15, fontweight="bold")
    axs[0, 1].set_ylabel("minutes")
    axs[0, 1].set_xticks(runs)
    for x, y in zip(runs, elapsed_min):
        axs[0, 1].text(x, y + 0.25, f"{y:.1f}", ha="center", fontsize=11, fontweight="bold")

    axs[1, 0].bar(runs, tasks, color="#2F80ED", edgecolor="#111827")
    axs[1, 0].set_title("생성 WBS 규모", fontsize=15, fontweight="bold")
    axs[1, 0].set_ylabel("tasks")
    axs[1, 0].set_xticks(runs)
    for x, y in zip(runs, tasks):
        axs[1, 0].text(x, y + 0.5, str(y), ha="center", fontsize=11, fontweight="bold")

    axs[1, 1].plot(runs, gini, marker="D", linewidth=3, color="#F59E0B")
    axs[1, 1].set_ylim(0.35, 0.58)
    axs[1, 1].set_title("업무량 불균형 지표", fontsize=15, fontweight="bold")
    axs[1, 1].set_ylabel("workload gini")
    axs[1, 1].set_xticks(runs)
    axs[1, 1].text(0.04, 0.88, "낮을수록 균형적", transform=axs[1, 1].transAxes, fontsize=11, color="#6B7280")

    for ax in axs.flat:
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "fig2_qwen4b_taskmanager_run_dashboard.png")
    plt.close(fig)


def _save_heatmap(rows):
    metrics = [
        ("Structure", "judge_structure"),
        ("Assignment", "judge_assignment"),
        ("Debate", "judge_debate"),
        ("Auto Score", "autoscore_final"),
        ("Auto Allocation", "autoscore_allocation"),
        ("MECE", "mece_score"),
        ("Granularity", "granularity_fitness"),
        ("Comm Eff.", "comm_efficiency"),
    ]
    data = np.array([[r[k] for _, k in metrics] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=180)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), [m[0] for m in metrics], rotation=25, ha="right", fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(len(rows)), [f"Run {r['run_id']}" for r in rows], fontsize=12, fontweight="bold")
    ax.set_title("Qwen 4B Task Manager: 품질 지표 매트릭스", fontsize=20, fontweight="bold", pad=16)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] < 0.45 else "#111827"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("score", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_qwen4b_taskmanager_quality_matrix.png")
    plt.close(fig)


def _save_workload(snapshots):
    latest_path, latest = snapshots[-1]
    loads = _member_loads(latest)
    names = list(loads)
    values = [loads[n] for n in names]
    colors = ["#EB5757" if v == max(values) else "#27AE60" if v == min(values) else "#2F80ED" for v in values]
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=180)
    ax.barh(names, values, color=colors, edgecolor="#111827")
    ax.invert_yaxis()
    ax.set_title("Run 3 배정 결과: 팀원별 업무량 분포", fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("assigned estimated days", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    for name, value in zip(names, values):
        ax.text(value + 0.5, name, f"{value:.1f}d", va="center", fontsize=12, fontweight="bold")
    legend = [
        Patch(facecolor="#EB5757", edgecolor="#111827", label="최대 부하"),
        Patch(facecolor="#27AE60", edgecolor="#111827", label="최소 부하"),
        Patch(facecolor="#2F80ED", edgecolor="#111827", label="기타"),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_qwen4b_taskmanager_workload_distribution.png")
    plt.close(fig)


def _write_markdown(rows, snapshots):
    mean_overall = _mean(rows, "judge_overall")
    mean_structure = _mean(rows, "judge_structure")
    mean_assignment = _mean(rows, "judge_assignment")
    mean_debate = _mean(rows, "judge_debate")
    mean_auto = _mean(rows, "autoscore_final")
    mean_elapsed = _mean(rows, "elapsed_sec") / 60
    mean_tasks = _mean(rows, "total_tasks")
    mean_gini = _mean(rows, "workload_gini")
    snapshots_text = "\n".join(f"- `{p.name}`" for p, _ in snapshots)
    run_rows = "\n".join(
        "| {run_id} | {total_tasks} | {elapsed:.1f} | {js:.2f} | {ja:.2f} | {jd:.2f} | {jo:.3f} | {auto:.3f} | {gini:.3f} |".format(
            run_id=r["run_id"],
            total_tasks=r["total_tasks"],
            elapsed=r["elapsed_sec"] / 60,
            js=r["judge_structure"],
            ja=r["judge_assignment"],
            jd=r["judge_debate"],
            jo=r["judge_overall"],
            auto=r["autoscore_final"],
            gini=r["workload_gini"],
        )
        for r in rows
    )
    md = f"""# Qwen 4B Task Manager Backbone Swap Summary

## 실험 조건

- 조건: `C3_3rounds`
- 교체 지점: `Task Manager`만 Qwen 4B로 교체
- 유지 지점: WBS Gen, 토론 에이전트, 최종 정리, judge는 기존 설정 유지
- 모델 endpoint: `http://127.0.0.1:8082`
- raw CSV: `{CSV_PATH.name}`

## 핵심 결과

- 평균 LLM Judge Overall: **{mean_overall:.3f}**
- 평균 Structure / Assignment / Debate: **{mean_structure:.3f} / {mean_assignment:.3f} / {mean_debate:.3f}**
- 평균 Auto Score: **{mean_auto:.3f}**
- 평균 WBS 규모: **{mean_tasks:.1f} tasks**
- 평균 소요 시간: **{mean_elapsed:.1f} min/run**
- 평균 workload gini: **{mean_gini:.3f}**

해석: Qwen 4B Task Manager는 JSON 구조와 팀원 ID 복사는 정상적으로 수행했다. 다만 judge reason 기준으로 assignment는 skill-fit과 workload balance가 병목이다. Debate 점수는 높지만 이는 후속 Gemma 기반 토론 단계의 보정 효과가 섞여 있으므로, Task Manager 단독 우수성으로 과해석하면 안 된다.

## Run별 지표

| Run | Tasks | Time min | Structure | Assignment | Debate | Overall | Auto | Workload Gini |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{run_rows}

## 산출물 품질 메모

- 모든 run에서 `bad assignee`는 발견되지 않음. 즉 Qwen 4B가 허용된 팀원 ID를 그대로 사용했다.
- 모든 L3 task에 담당자가 배정됨.
- judge가 반복적으로 지적한 약점은 QA/검증 성격 task의 skill mismatch와 업무량 불균형이다.
- WBS 생성은 Gemma 26B가 담당했으므로, 이 실험은 WBS 생성력 비교가 아니라 Task Manager 배정 판단력 비교로 해석해야 한다.

## Figure

- `fig1_qwen4b_taskmanager_score_profile.png`: judge score profile
- `fig2_qwen4b_taskmanager_run_dashboard.png`: run별 안정성 대시보드
- `fig3_qwen4b_taskmanager_quality_matrix.png`: 품질 지표 heatmap
- `fig4_qwen4b_taskmanager_workload_distribution.png`: Run 3 업무량 분포

## Snapshot

{snapshots_text}
"""
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")


def main():
    _set_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_rows()
    snapshots = _read_snapshots()
    _save_score_profile(rows)
    _save_run_dashboard(rows)
    _save_heatmap(rows)
    _save_workload(snapshots)
    _write_markdown(rows, snapshots)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
