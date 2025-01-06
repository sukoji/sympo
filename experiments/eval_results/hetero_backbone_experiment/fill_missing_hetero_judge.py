"""Retry only missing (-1) judge cells in summary_hetero.csv."""
import csv
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, '/home/piai/ai_course/agent_test')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

from eval.llm_judge import evaluate_wbs


ROOT = '/home/piai/ai_course/agent_test'
SUMMARY = f'{ROOT}/eval_results/hetero_backbone_experiment/summary_hetero.csv'
BACKUP = SUMMARY.replace('.csv', '.before_fill_missing.csv')
EVAL_DIMS = ['structure', 'assignment', 'debate']
DIM_WEIGHTS = {"structure": 0.40, "assignment": 0.35, "debate": 0.25}

TARGET_PATTERNS = [
    f'{ROOT}/eval_results/hetero_backbone_experiment/snapshots/wbs_snapshot_C3_3rounds_*hetero_wbsgen*.json',
    f'{ROOT}/eval_results/hetero_backbone_experiment/snapshots/wbs_snapshot_C3_3rounds_*hetero_taskmgr*.json',
    f'{ROOT}/eval_results/hetero_backbone_experiment/snapshots/wbs_snapshot_C3_3rounds_*hetero_both*.json',
    f'{ROOT}/eval_results/gemma26_ablation/snapshots/wbs_snapshot_C3_3rounds*.json',
    f'{ROOT}/eval_results/gemini_ablation/snapshots/wbs_snapshot_C3_3rounds*.json',
]


def load_team():
    from data_pipeline.member_parser import MemberParser
    md = f'{ROOT}/sample_data/sample_members'
    return [
        MemberParser.from_resume_text(
            open(os.path.join(md, f), encoding='utf-8').read(),
            f.replace('member_', '').replace('.txt', ''),
        )
        for f in sorted(os.listdir(md)) if f.endswith('.txt')
    ]


def snapshot_paths():
    paths = {}
    for pattern in TARGET_PATTERNS:
        for fp in glob.glob(pattern):
            paths[os.path.basename(fp)] = fp
    return paths


def recompute_overall(row):
    active = {}
    for dim in EVAL_DIMS:
        score = float(row[f'judge_{dim}'])
        if score >= 0:
            active[dim] = score
    if not active:
        row['judge_overall'] = '-1'
        return
    total_w = sum(DIM_WEIGHTS[d] for d in active)
    overall = sum((DIM_WEIGHTS[d] / total_w) * active[d] for d in active)
    row['judge_overall'] = str(round(overall, 4))


def main():
    with open(SUMMARY, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    paths = snapshot_paths()
    team = load_team()
    updates = 0

    for row in rows:
        missing = [d for d in EVAL_DIMS if float(row[f'judge_{d}']) < 0]
        if not missing:
            continue

        fp = paths.get(row['snapshot'])
        if not fp:
            print(f"skip missing file: {row['snapshot']}")
            continue

        data = json.load(open(fp, encoding='utf-8'))
        print(f"\n=== {row['condition']} r{row['run_idx']} missing={missing} ===")
        for dim in missing:
            for attempt in range(1, 5):
                print(f"  retry {dim} attempt {attempt}")
                res = evaluate_wbs(
                    wbs_tasks=data.get('wbs_tasks', []),
                    team_members=team,
                    debate_log=data.get('debate_log', []),
                    eval_dims=[dim],
                    cross_judge=False,
                )
                score = res[dim]['score']
                if score >= 0:
                    row[f'judge_{dim}'] = str(score)
                    row[f'{dim}_reason'] = res[dim]['reason'][:200]
                    updates += 1
                    print(f"  filled {dim}={score}")
                    break
                print(f"  still missing: {res[dim]['reason'][:120]}")
                time.sleep(15 * attempt)

        recompute_overall(row)

    if updates:
        shutil.copyfile(SUMMARY, BACKUP)
        with open(SUMMARY, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {SUMMARY}; backup={BACKUP}; filled={updates}")
    else:
        print("\nNo missing judge cells were filled.")


if __name__ == '__main__':
    main()
