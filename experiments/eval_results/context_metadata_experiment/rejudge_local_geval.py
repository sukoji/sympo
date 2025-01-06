"""Local LLM-Judge G-Eval for context metadata ablation.

Uses the OpenAI-compatible local endpoint configured by QWEN_API_URL/QWEN_API_MODEL
as the judge backend, avoiding external Gemini quota failures.
"""
import csv
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/home/piai/ai_course/agent_test")
from dotenv import load_dotenv

load_dotenv("/home/piai/ai_course/agent_test/.env")
os.environ["GEVAL_JUDGE_BACKEND"] = "qwen-api"
os.environ["JUDGE_METHOD"] = "geval"
os.environ["QWEN_ENABLE_THINKING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from data_pipeline.member_parser import MemberParser
from eval.llm_judge import evaluate_wbs

EXP = "/home/piai/ai_course/agent_test/eval_results/context_metadata_experiment"
EVAL_DIMS = ["structure", "assignment", "debate"]
JUDGE_MODEL = os.getenv("QWEN_API_MODEL", "local-qwen-api")


def load_team():
    md = "/home/piai/ai_course/agent_test/sample_data/sample_members"
    return [
        MemberParser.from_resume_text(
            open(os.path.join(md, f), encoding="utf-8").read(),
            f.replace("member_", "").replace(".txt", ""),
        )
        for f in sorted(os.listdir(md))
        if f.endswith(".txt")
    ]


def median_valid(values):
    valid = [x for x in values if x >= 0]
    return statistics.median(valid) if valid else -1


def main():
    team = load_team()
    targets = [
        ("M_resume", "/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/snapshots/wbs_snapshot_C3_3rounds*.json"),
        ("M_both", "/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/snapshots/wbs_snapshot_C4_with_disc*.json"),
        ("M_disc", f"{EXP}/snapshots/wbs_snapshot_C4_with_disc_*context_disc_only*.json"),
    ]
    rows = []
    for mode, pattern in targets:
        fps = sorted(glob.glob(pattern))[:3]
        print(f"\n=== {mode}: {len(fps)} snapshots ===", flush=True)
        for run_id, fp in enumerate(fps, 1):
            data = json.load(open(fp, encoding="utf-8"))
            wbs = data.get("wbs_tasks", [])
            debate = data.get("debate_log", [])
            print(f"[{run_id}/{len(fps)}] {mode} n_tasks={len(wbs)}", flush=True)
            s_vals, a_vals, d_vals = [], [], []
            for trial in range(3):
                res = evaluate_wbs(
                    wbs_tasks=wbs,
                    team_members=team,
                    debate_log=debate,
                    eval_dims=EVAL_DIMS,
                    cross_judge=False,
                    judge_model=JUDGE_MODEL,
                    judge_method="geval",
                )
                s_vals.append(res["structure"]["score"])
                a_vals.append(res["assignment"]["score"])
                d_vals.append(res["debate"]["score"])
                print(
                    f"  trial {trial + 1}: "
                    f"S={s_vals[-1]} A={a_vals[-1]} D={d_vals[-1]}",
                    flush=True,
                )
                time.sleep(0.5)
            sm, am, dm = median_valid(s_vals), median_valid(a_vals), median_valid(d_vals)
            valid = [(sm, 0.40), (am, 0.35), (dm, 0.25)]
            valid = [(v, w) for v, w in valid if v >= 0]
            overall = sum(v * w for v, w in valid) / sum(w for _, w in valid) if valid else -1
            rows.append(
                {
                    "mode": mode,
                    "run_id": run_id,
                    "snapshot": os.path.basename(fp),
                    "n_tasks": len(wbs),
                    "judge_backend": "qwen-api-local-geval",
                    "judge_model": JUDGE_MODEL,
                    "S_t1": s_vals[0],
                    "S_t2": s_vals[1],
                    "S_t3": s_vals[2],
                    "S_med": sm,
                    "A_t1": a_vals[0],
                    "A_t2": a_vals[1],
                    "A_t3": a_vals[2],
                    "A_med": am,
                    "D_t1": d_vals[0],
                    "D_t2": d_vals[1],
                    "D_t3": d_vals[2],
                    "D_med": dm,
                    "overall_median": round(overall, 4) if overall >= 0 else -1,
                }
            )
            print(f"  -> median S={sm} A={am} D={dm} overall={overall:.4f}", flush=True)

    out = f"{EXP}/summary_judged_local_geval.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
