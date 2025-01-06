"""Robust judge median x3 for context metadata conditions.

This script never writes `summary_judged.csv` with invalid judge scores.
For every snapshot it collects three complete trials, where each trial must
have valid Structure, Assignment, and Debate scores. If a dimension returns
N/A/-1 because of parsing or API failure, only that missing dimension is
retried. A partial CSV is kept for inspection, but the final CSV is replaced
only after every row is complete.
"""
import csv
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/piai/ai_course/agent_test")

from dotenv import load_dotenv

load_dotenv("/home/piai/ai_course/agent_test/.env")

os.environ["GEVAL_JUDGE_BACKEND"] = "gemini"
os.environ["JUDGE_METHOD"] = "scalar"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from eval.llm_judge import evaluate_wbs


EXP = Path("/home/piai/ai_course/agent_test/eval_results/context_metadata_experiment")
EVAL_DIMS = ["structure", "assignment", "debate"]
DIM_TO_COL = {"structure": "S", "assignment": "A", "debate": "D"}
WEIGHTS = {"S": 0.40, "A": 0.35, "D": 0.25}

TRIALS_PER_SNAPSHOT = int(os.getenv("CONTEXT_JUDGE_TRIALS", "3"))
MAX_TRIAL_ATTEMPTS = int(os.getenv("CONTEXT_JUDGE_MAX_ATTEMPTS", "8"))
RETRY_SLEEP_SEC = float(os.getenv("CONTEXT_JUDGE_RETRY_SLEEP_SEC", "10"))
MODEL_SLEEP_SEC = float(os.getenv("CONTEXT_JUDGE_MODEL_SLEEP_SEC", "2"))
JUDGE_MODELS = [
    m.strip()
    for m in os.getenv(
        "CONTEXT_JUDGE_MODELS",
        "gemini-3.1-pro-preview,gemini-3.1-flash-lite-preview",
    ).split(",")
    if m.strip()
]

if not JUDGE_MODELS:
    raise RuntimeError("CONTEXT_JUDGE_MODELS must contain at least one model")

os.environ.setdefault("JUDGE_MODEL_GEMINI", JUDGE_MODELS[0])

TARGETS = [
    (
        "M_resume",
        "/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/snapshots/"
        "wbs_snapshot_C3_3rounds*.json",
    ),
    (
        "M_both",
        "/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/snapshots/"
        "wbs_snapshot_C4_with_disc*.json",
    ),
    (
        "M_disc",
        str(EXP / "snapshots/wbs_snapshot_C4_with_disc_*context_disc_only*.json"),
    ),
]


def load_team():
    from data_pipeline.member_parser import MemberParser

    member_dir = Path("/home/piai/ai_course/agent_test/sample_data/sample_members")
    members = []
    for fp in sorted(member_dir.glob("*.txt")):
        member_id = fp.name.replace("member_", "").replace(".txt", "")
        members.append(MemberParser.from_resume_text(fp.read_text(), member_id))
    return members


def valid_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    return None


def result_score(result, dim):
    return valid_score(result.get(dim, {}).get("score"))


def result_reason(result, dim):
    return str(result.get(dim, {}).get("reason", ""))[:180]


def run_complete_trial(wbs, team, debate, label, run_id, trial_no):
    scores = {}
    reasons = {}
    models_used = {}

    for attempt in range(1, MAX_TRIAL_ATTEMPTS + 1):
        missing = [dim for dim in EVAL_DIMS if dim not in scores]
        if not missing:
            return scores, reasons, models_used

        for model in JUDGE_MODELS:
            print(
                f"   trial {trial_no} attempt {attempt}/{MAX_TRIAL_ATTEMPTS} "
                f"model={model} dims={','.join(missing)}",
                flush=True,
            )
            try:
                res = evaluate_wbs(
                    wbs_tasks=wbs,
                    team_members=team,
                    debate_log=debate,
                    eval_dims=missing,
                    cross_judge=False,
                    judge_model=model,
                )
            except Exception as exc:
                print(f"      call failed: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(MODEL_SLEEP_SEC)
                continue

            still_missing = []
            for dim in missing:
                score = result_score(res, dim)
                col = DIM_TO_COL[dim]
                if score is None:
                    reason = result_reason(res, dim)
                    print(f"      {col}=N/A reason={reason}", flush=True)
                    still_missing.append(dim)
                else:
                    scores[dim] = score
                    reasons[dim] = result_reason(res, dim)
                    models_used[dim] = model
                    print(f"      {col}={score:.4f}", flush=True)

            missing = still_missing
            if not missing:
                return scores, reasons, models_used
            time.sleep(MODEL_SLEEP_SEC)

        if attempt < MAX_TRIAL_ATTEMPTS:
            print(
                f"      incomplete trial; sleeping {RETRY_SLEEP_SEC:g}s before retry",
                flush=True,
            )
            time.sleep(RETRY_SLEEP_SEC)

    missing_cols = ",".join(DIM_TO_COL[d] for d in EVAL_DIMS if d not in scores)
    raise RuntimeError(
        f"{label} r{run_id} trial {trial_no} still has invalid dimensions: {missing_cols}"
    )


def median(values):
    if len(values) != TRIALS_PER_SNAPSHOT:
        raise RuntimeError(f"Expected {TRIALS_PER_SNAPSHOT} values, got {len(values)}")
    if any(valid_score(v) is None for v in values):
        raise RuntimeError(f"Invalid values in median input: {values}")
    return statistics.median(float(v) for v in values)


def overall_score(s_med, a_med, d_med):
    return (
        s_med * WEIGHTS["S"] + a_med * WEIGHTS["A"] + d_med * WEIGHTS["D"]
    ) / sum(WEIGHTS.values())


def validate_rows(rows):
    if not rows:
        return False
    score_cols = [
        "S_t1",
        "S_t2",
        "S_t3",
        "S_med",
        "A_t1",
        "A_t2",
        "A_t3",
        "A_med",
        "D_t1",
        "D_t2",
        "D_t3",
        "D_med",
        "overall_median",
    ]
    for row in rows:
        for col in score_cols:
            if valid_score(row.get(col)) is None:
                return False
    return True


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    team = load_team()
    rows = []
    partial_path = EXP / "summary_judged_partial.csv"
    tmp_path = EXP / "summary_judged.tmp.csv"
    final_path = EXP / "summary_judged.csv"

    for label, pattern in TARGETS:
        fps = sorted(glob.glob(pattern))[:3]
        if len(fps) != 3:
            raise RuntimeError(f"{label} expected 3 snapshots, found {len(fps)}")

        print(f"\n=== {label}: {len(fps)} snapshots ===", flush=True)
        for run_id, fp in enumerate(fps, 1):
            with open(fp, encoding="utf-8") as f:
                snapshot = json.load(f)
            wbs = snapshot.get("wbs_tasks", [])
            debate = snapshot.get("debate_log", [])
            print(
                f"\n[{run_id}/{len(fps)}] {label} r{run_id} "
                f"n_tasks={len(wbs)} snapshot={Path(fp).name}",
                flush=True,
            )

            trials = []
            for trial_no in range(1, TRIALS_PER_SNAPSHOT + 1):
                scores, _, models_used = run_complete_trial(
                    wbs, team, debate, label, run_id, trial_no
                )
                s = scores["structure"]
                a = scores["assignment"]
                d = scores["debate"]
                trials.append({"S": s, "A": a, "D": d})
                model_summary = ",".join(
                    f"{DIM_TO_COL[dim]}={models_used[dim]}" for dim in EVAL_DIMS
                )
                print(
                    f"   trial {trial_no} complete: "
                    f"S={s:.4f} A={a:.4f} D={d:.4f} ({model_summary})",
                    flush=True,
                )

            s_vals = [t["S"] for t in trials]
            a_vals = [t["A"] for t in trials]
            d_vals = [t["D"] for t in trials]
            s_med = median(s_vals)
            a_med = median(a_vals)
            d_med = median(d_vals)
            overall = overall_score(s_med, a_med, d_med)

            row = {
                "mode": label,
                "run_id": run_id,
                "snapshot": Path(fp).name,
                "n_tasks": len(wbs),
                "S_t1": s_vals[0],
                "S_t2": s_vals[1],
                "S_t3": s_vals[2],
                "S_med": s_med,
                "A_t1": a_vals[0],
                "A_t2": a_vals[1],
                "A_t3": a_vals[2],
                "A_med": a_med,
                "D_t1": d_vals[0],
                "D_t2": d_vals[1],
                "D_t3": d_vals[2],
                "D_med": d_med,
                "overall_median": overall,
            }
            rows.append(row)
            write_rows(partial_path, rows)
            print(
                f"   MEDIAN: S={s_med:.4f} A={a_med:.4f} "
                f"D={d_med:.4f} Overall={overall:.4f}",
                flush=True,
            )
            print(f"   partial saved: {partial_path}", flush=True)

    if not validate_rows(rows):
        raise RuntimeError("Refusing to write final summary: invalid score detected")

    write_rows(tmp_path, rows)
    os.replace(tmp_path, final_path)
    print(f"\nSaved complete judged summary: {final_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
