import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "eval_results" / "wbsgen_qwen_experiment"
FIG_DIR = EXP_DIR / "figures"
SNAP_DIR = EXP_DIR / "snapshots"
SUMMARY = EXP_DIR / "summary_qwen-api_wbsgen_qwen4b_20260426_040226.csv"
GEMMA_SUMMARY = ROOT / "eval_results" / "gemma26_ablation" / "summary_qwen-api_gemma26_ablation_20260423_171707.csv"
GEMMA_SNAP_DIR = ROOT / "eval_results" / "gemma26_ablation" / "snapshots"


def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumSquare_acR.ttf",
        "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_snapshot_metrics():
    rows = []
    for path in sorted(SNAP_DIR.glob("wbs_snapshot_H_wbsgen_qwen4b_r*_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = data["wbs_tasks"]
        ids = {t["task_id"] for t in tasks}
        by_id = {t["task_id"]: t for t in tasks}
        level_counts = Counter(t["level"] for t in tasks)
        l2_tasks = [t for t in tasks if t["level"] == "L2"]
        l3_tasks = [t for t in tasks if t["level"] == "L3"]
        l3_by_l2 = Counter(t.get("parent_id") for t in l3_tasks)
        l3_by_l1 = Counter(by_id.get(t.get("parent_id"), {}).get("parent_id") for t in l3_tasks)
        missing_deps = sum(
            1
            for t in tasks
            for dep in (t.get("dependencies") or [])
            if dep not in ids
        )
        max_l1_share = max(l3_by_l1.values()) / max(len(l3_tasks), 1)
        repair = data.get("wbs_repair_stats", {})
        judge = data.get("llm_judge", {})
        rows.append(
            {
                "run": f"Run {data['run_id']}",
                "run_id": data["run_id"],
                "L1": level_counts["L1"],
                "L2": level_counts["L2"],
                "L3": level_counts["L3"],
                "total_tasks": len(tasks),
                "l2_without_l3": sum(1 for t in l2_tasks if l3_by_l2[t["task_id"]] == 0),
                "avg_l3_per_l2": len(l3_tasks) / max(len(l2_tasks), 1),
                "max_l1_l3_share": max_l1_share,
                "missing_deps": missing_deps,
                "retry_l3_count": repair.get("retry_l3_count", 0),
                "synthetic_l3_count": repair.get("synthetic_l3_count", 0),
                "judge_structure": judge.get("structure", {}).get("score"),
                "judge_assignment": judge.get("assignment", {}).get("score"),
                "judge_debate": judge.get("debate", {}).get("score"),
                "judge_overall": judge.get("overall"),
            }
        )
    return pd.DataFrame(rows)


def load_summary():
    df = pd.read_csv(SUMMARY)
    df["run"] = "Run " + df["run_id"].astype(str)
    return df


def load_gemma_c3_reference():
    summary = pd.read_csv(GEMMA_SUMMARY)
    c3 = summary[summary["condition"] == "C3_3rounds"].copy()
    snap_rows = []
    for path in sorted(GEMMA_SNAP_DIR.glob("wbs_snapshot_C3_3rounds_r*_qwen-api_gemma26_ablation_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter(t["level"] for t in data["wbs_tasks"])
        snap_rows.append({"L1": counts["L1"], "L2": counts["L2"], "L3": counts["L3"], "total_tasks": len(data["wbs_tasks"])})
    snap = pd.DataFrame(snap_rows)
    return {
        "L1": snap["L1"].mean(),
        "L2": snap["L2"].mean(),
        "L3": snap["L3"].mean(),
        "total_tasks": snap["total_tasks"].mean(),
        "autoscore_final": c3["autoscore_final"].mean(),
        "judge_overall": c3["judge_overall"].mean(),
        "judge_structure": c3["judge_structure"].mean(),
        "elapsed_sec": c3["elapsed_sec"].mean(),
    }


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=13)


def draw_structure(snapshot_df, ref):
    labels = list(snapshot_df["run"]) + ["Gemma C3\n평균"]
    l1 = list(snapshot_df["L1"]) + [ref["L1"]]
    l2 = list(snapshot_df["L2"]) + [ref["L2"]]
    l3 = list(snapshot_df["L3"]) + [ref["L3"]]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    colors = {"L1": "#314E8A", "L2": "#5F8CCB", "L3": "#D58A24"}
    ax.bar(x, l1, label="L1 Phase", color=colors["L1"], width=0.66)
    ax.bar(x, l2, bottom=l1, label="L2 기능그룹", color=colors["L2"], width=0.66)
    bottom = [a + b for a, b in zip(l1, l2)]
    ax.bar(x, l3, bottom=bottom, label="L3 작업패키지", color=colors["L3"], width=0.66)
    for i, total in enumerate([a + b + c for a, b, c in zip(l1, l2, l3)]):
        ax.text(i, total + 2, f"{total:.0f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.axhspan(32, 38, color="#BDBDBD", alpha=0.18, label="Gemma C3 정상 범위")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("태스크 수", fontsize=15)
    ax.set_title("Qwen WBS Gen 산출 구조 변동성", fontsize=21, fontweight="bold", pad=18)
    ax.legend(ncol=4, loc="upper left", fontsize=12, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_structure_variability.png", bbox_inches="tight")
    plt.close(fig)


def draw_repairs(snapshot_df):
    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    x = range(len(snapshot_df))
    ax.bar(x, snapshot_df["retry_l3_count"], color="#D58A24", width=0.6, label="재프롬프트로 확보한 L3")
    ax.bar(
        x,
        snapshot_df["synthetic_l3_count"],
        bottom=snapshot_df["retry_l3_count"],
        color="#B91C1C",
        hatch="///",
        edgecolor="#7F1D1D",
        width=0.6,
        label="템플릿 합성 L3",
    )
    for i, row in snapshot_df.iterrows():
        ax.text(i, row["retry_l3_count"] + row["synthetic_l3_count"] + 1.3, f"{int(row['retry_l3_count'] + row['synthetic_l3_count'])}", ha="center", fontsize=14, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, snapshot_df["max_l1_l3_share"] * 100, color="#111827", marker="o", linewidth=2.8, label="최대 L1 집중도")
    for i, v in enumerate(snapshot_df["max_l1_l3_share"] * 100):
        ax2.text(i, v + 2.2, f"{v:.0f}%", ha="center", fontsize=12, color="#111827")
    ax.set_xticks(list(x))
    ax.set_xticklabels(snapshot_df["run"], fontsize=14)
    ax.set_ylabel("복구 의존 L3 수", fontsize=15)
    ax2.set_ylabel("특정 L1에 몰린 L3 비율", fontsize=15)
    ax.set_title("Qwen WBS Gen의 구조 복구 의존도", fontsize=21, fontweight="bold", pad=18)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=12, frameon=False)
    style_ax(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_repair_and_concentration.png", bbox_inches="tight")
    plt.close(fig)


def draw_score_gap(summary_df, ref):
    labels = list(summary_df["run"]) + ["Gemma C3\n평균"]
    auto = list(summary_df["autoscore_final"]) + [ref["autoscore_final"]]
    judge = list(summary_df["judge_overall"]) + [ref["judge_overall"]]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    ax.bar([i - width / 2 for i in x], auto, width, color="#5F8CCB", label="AutoScore")
    ax.bar([i + width / 2 for i in x], judge, width, color="#D58A24", label="LLM Judge")
    for i, (a, j) in enumerate(zip(auto, judge)):
        ax.text(i - width / 2, a + 0.018, f"{a:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.text(i + width / 2, j + 0.018, f"{j:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("점수", fontsize=15)
    ax.set_title("점수만 보면 숨겨지는 산출 품질 문제", fontsize=21, fontweight="bold", pad=18)
    ax.text(0, 0.08, "Run1: 구조 결함을 Judge가 감점", fontsize=13, color="#7F1D1D", fontweight="bold")
    ax.text(1.2, 0.18, "Run2/3: 범위 축소에도 점수는 상승", fontsize=13, color="#7F1D1D", fontweight="bold")
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_score_gap.png", bbox_inches="tight")
    plt.close(fig)


def draw_problem_matrix(snapshot_df):
    mat = snapshot_df[["L1", "L3", "retry_l3_count", "missing_deps", "judge_structure", "judge_assignment"]].copy()
    mat.index = snapshot_df["run"]
    norm = pd.DataFrame(index=mat.index)
    norm["L1 부족/과소"] = (6 - mat["L1"]).clip(lower=0) / 3
    norm["L3 변동"] = (mat["L3"] - mat["L3"].mean()).abs() / max(mat["L3"].mean(), 1)
    norm["복구 의존"] = mat["retry_l3_count"] / max(mat["retry_l3_count"].max(), 1)
    norm["의존성 오류"] = mat["missing_deps"] / max(mat["missing_deps"].max(), 1)
    norm["구조 감점"] = 1 - mat["judge_structure"]
    norm["배정 감점"] = 1 - mat["judge_assignment"]
    norm = norm.clip(0, 1)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    im = ax.imshow(norm.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(norm.columns)))
    ax.set_xticklabels(norm.columns, fontsize=13)
    ax.set_yticks(range(len(norm.index)))
    ax.set_yticklabels(norm.index, fontsize=14)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=12, color="#111827")
    ax.set_title("산출 샘플별 문제 신호 매트릭스", fontsize=21, fontweight="bold", pad=18)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("문제 강도", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_problem_matrix.png", bbox_inches="tight")
    plt.close(fig)


def write_report(snapshot_df, summary_df, ref):
    rows = []
    for _, row in snapshot_df.iterrows():
        srow = summary_df[summary_df["run_id"] == row["run_id"]].iloc[0]
        rows.append(
            f"| {row['run']} | {int(row['L1'])}/{int(row['L2'])}/{int(row['L3'])} | "
            f"{int(row['retry_l3_count'])}+{int(row['synthetic_l3_count'])} | "
            f"{int(row['missing_deps'])} | {row['max_l1_l3_share']*100:.1f}% | "
            f"{srow['autoscore_final']:.3f} | {srow['judge_overall']:.3f} |"
        )

    report = f"""# Qwen WBS Gen 산출 샘플 문제 정리

## 결론

WBS Gen만 Qwen 4B로 교체한 실험은 실행 자체는 가능하지만, 산출물의 구조 안정성이 낮습니다. AutoScore는 세 run 모두 약 0.80으로 안정적으로 보이나, snapshot을 보면 L1/L2/L3 범위와 phase 분포가 크게 흔들리고 모든 run에서 L3 재프롬프트가 필요했습니다. 따라서 `Gemma-4-26B C3`를 선택해야 한다는 결론을 보강하는 반례 실험으로 쓰는 것이 적절합니다.

## Run별 문제 신호

| run | L1/L2/L3 | L3 복구/합성 | 깨진 dependency | 최대 L1 집중도 | AutoScore | LLM Judge |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

참고: Gemma C3 평균은 total task 약 {ref['total_tasks']:.1f}, AutoScore {ref['autoscore_final']:.3f}, LLM Judge {ref['judge_overall']:.3f}입니다.

## 샘플에서 보이는 구체적 문제

1. 구조 완성도가 모델 생성에 강하게 의존합니다.
   - run1은 L3를 54개 재프롬프트로 확보했고, 1개는 템플릿 스텁으로 합성되었습니다.
   - run2는 L3 31개만 생성되어 전체 WBS가 51 tasks로 축소되었습니다.
   - run3은 L1이 3개뿐이고 L3의 62.8%가 `L1-01`에 몰렸습니다.

2. 의존성 표현이 깨지는 샘플이 있습니다.
   - run2에서 `L1-01-01-02` 같은 존재하지 않는 dependency가 4개 발견되었습니다.
   - 이는 WBS ID 체계가 안정적으로 유지되지 않았다는 신호입니다.

3. 높은 점수가 실제 구조 품질을 완전히 설명하지 못합니다.
   - run2/run3은 LLM Judge overall이 각각 0.8075, 0.8245로 높지만, 산출 범위가 축소되거나 특정 phase에 집중되었습니다.
   - AutoScore는 세 run 모두 0.797~0.800 수준이라 구조 변동을 충분히 드러내지 못합니다.

4. 발표 해석 방향
   - 이 결과는 “Qwen 4B를 WBS Gen에 바로 갈아끼울 수 있다”가 아니라 “실행은 되지만 구조 안정성과 재현성이 부족하다”로 해석해야 합니다.
   - 최종 선택 논리는 `Gemma-4-26B + C3`가 점수뿐 아니라 산출 범위 안정성, ID 일관성, 복구 의존도 측면에서도 더 방어 가능하다는 방향이 좋습니다.

## 생성 figure

- `fig1_structure_variability.png`: Qwen run별 WBS 크기와 계층 구조 변동성
- `fig2_repair_and_concentration.png`: L3 재프롬프트/합성 의존도와 특정 phase 집중도
- `fig3_score_gap.png`: AutoScore와 LLM Judge의 괴리
- `fig4_problem_matrix.png`: 샘플별 문제 신호 heatmap
"""
    (EXP_DIR / "QWEN_WBSGEN_OUTPUT_ISSUES.md").write_text(report, encoding="utf-8")


def main():
    set_korean_font()
    FIG_DIR.mkdir(exist_ok=True)
    snapshot_df = load_snapshot_metrics()
    summary_df = load_summary()
    ref = load_gemma_c3_reference()
    draw_structure(snapshot_df, ref)
    draw_repairs(snapshot_df)
    draw_score_gap(summary_df, ref)
    draw_problem_matrix(snapshot_df)
    write_report(snapshot_df, summary_df, ref)
    print("Saved figures to", FIG_DIR)
    print("Saved report to", EXP_DIR / "QWEN_WBSGEN_OUTPUT_ISSUES.md")


if __name__ == "__main__":
    main()
