"""Local OpenAI-compatible scalar LLM-Judge for context metadata ablation."""
import csv
import glob
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, "/home/piai/ai_course/agent_test")
from dotenv import load_dotenv

load_dotenv("/home/piai/ai_course/agent_test/.env")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from data_pipeline.member_parser import MemberParser
from eval.llm_judge import (
    ASSIGNMENT_PROMPT,
    DEBATE_PROMPT,
    STRUCTURE_PROMPT,
    _format_debate,
    _format_team,
    _format_wbs,
    _parse_judge_response,
)

EXP = "/home/piai/ai_course/agent_test/eval_results/context_metadata_experiment"
BASE_URL = os.getenv("QWEN_API_URL", "http://127.0.0.1:8081").rstrip("/")
MODEL = os.getenv("QWEN_API_MODEL", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")


def endpoint():
    if BASE_URL.endswith("/v1"):
        return f"{BASE_URL}/chat/completions"
    if BASE_URL.endswith("/v1/chat/completions"):
        return BASE_URL
    return f"{BASE_URL}/v1/chat/completions"


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


def call_local_judge(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator. Return only one minified JSON object with score and reason.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_judge_response(text)
    if parsed.get("score", -1) >= 0:
        return parsed

    repair = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Convert the evaluation answer to valid JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "Return exactly one JSON object: "
                    "{\"score\": 0.00, \"reason\": \"short reason\"}\n"
                    "Use a score from 0.00 to 1.00.\n\n"
                    f"Invalid answer:\n{text[:1200]}\n\n"
                    f"Original task:\n{prompt[:3000]}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 400,
    }
    req = urllib.request.Request(
        endpoint(),
        data=json.dumps(repair).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    repair_text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    repaired = _parse_judge_response(repair_text)
    if repaired.get("score", -1) >= 0:
        repaired["reason"] = f"[repair_retry] {repaired.get('reason', '')}"[:400]
    return repaired


def median_valid(values):
    valid = [x for x in values if x >= 0]
    return statistics.median(valid) if valid else -1


def score_snapshot(fp: str, team):
    data = json.load(open(fp, encoding="utf-8"))
    wbs = data.get("wbs_tasks", [])
    debate = data.get("debate_log", [])
    wbs_text = _format_wbs(wbs, team_members=team, debate_log=debate)
    l3_lines = [line for line in wbs_text.split("\n") if "(L3," in line]
    prompts = {
        "S": STRUCTURE_PROMPT.format(wbs_text=wbs_text[:3000]),
        "A": ASSIGNMENT_PROMPT.format(
            team_text=_format_team(team)[:1000],
            assignment_text="\n".join(l3_lines[:30]) or "(no L3)",
        ),
        "D": DEBATE_PROMPT.format(debate_text=_format_debate(debate)[:3000]),
    }
    return wbs, {name: call_local_judge(prompt) for name, prompt in prompts.items()}


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
            s_vals, a_vals, d_vals = [], [], []
            n_tasks = 0
            for trial in range(3):
                wbs, scored = score_snapshot(fp, team)
                n_tasks = len(wbs)
                s_vals.append(scored["S"]["score"])
                a_vals.append(scored["A"]["score"])
                d_vals.append(scored["D"]["score"])
                print(
                    f"{mode} r{run_id} trial {trial + 1}: "
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
                    "n_tasks": n_tasks,
                    "judge_backend": "local-openai-compatible-scalar",
                    "judge_model": MODEL,
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
            print(f"-> median S={sm} A={am} D={dm} overall={overall:.4f}", flush=True)

    out = f"{EXP}/summary_judged_local_scalar.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
