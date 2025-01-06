"""Hetero 9 snapshots batch re-judge.
Includes baseline (gemma26_ablation/C3) + all_frontier (gemini_ablation/C3) for 5-cell comparison.
JUDGE_MODEL_GEMINI env var determines model.
"""
import json, os, sys, glob, csv, time
sys.path.insert(0, '/home/piai/ai_course/agent_test')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
# JUDGE_MODEL_GEMINI is read by llm_judge module
from eval.llm_judge import evaluate_wbs

EVAL_DIMS = ['structure', 'assignment', 'debate']

def load_team():
    from data_pipeline.member_parser import MemberParser
    md = '/home/piai/ai_course/agent_test/sample_data/sample_members'
    return [MemberParser.from_resume_text(open(os.path.join(md,f)).read(),
                                          f.replace('member_','').replace('.txt',''))
            for f in sorted(os.listdir(md)) if f.endswith('.txt')]

team = load_team()

# Targets: 3 new hetero conditions + 2 reused baselines
TARGETS = [
    ('H_wbsgen',       '/home/piai/ai_course/agent_test/eval_results/hetero_backbone_experiment/snapshots/wbs_snapshot_C3_3rounds_*hetero_wbsgen*.json'),
    ('H_taskmgr',      '/home/piai/ai_course/agent_test/eval_results/hetero_backbone_experiment/snapshots/wbs_snapshot_C3_3rounds_*hetero_taskmgr*.json'),
    ('H_both',         '/home/piai/ai_course/agent_test/eval_results/hetero_backbone_experiment/snapshots/wbs_snapshot_C3_3rounds_*hetero_both*.json'),
    ('H_baseline',     '/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/snapshots/wbs_snapshot_C3_3rounds*.json'),
    ('H_all_frontier', '/home/piai/ai_course/agent_test/eval_results/gemini_ablation/snapshots/wbs_snapshot_C3_3rounds*.json'),
]

rows = []
for label, pattern in TARGETS:
    fps = sorted(glob.glob(pattern))
    print(f"\n=== {label}: {len(fps)} snapshots ===")
    for i, fp in enumerate(fps[:3], 1):  # cap at 3 per condition
        d = json.load(open(fp))
        wbs = d.get('wbs_tasks', [])
        debate = d.get('debate_log', [])
        fname = os.path.basename(fp)
        print(f"  [{i}] {fname[:55]} n_tasks={len(wbs)}")
        try:
            res = evaluate_wbs(wbs_tasks=wbs, team_members=team, debate_log=debate,
                               eval_dims=EVAL_DIMS, cross_judge=False)
            rows.append({
                'condition': label, 'run_idx': i, 'snapshot': fname, 'n_tasks': len(wbs),
                'judge_structure': res['structure']['score'],
                'judge_assignment': res['assignment']['score'],
                'judge_debate': res['debate']['score'],
                'judge_overall': res['overall'],
                'structure_reason': res['structure']['reason'][:200],
                'assignment_reason': res['assignment']['reason'][:200],
                'debate_reason': res['debate']['reason'][:200],
            })
            print(f"     S={res['structure']['score']} A={res['assignment']['score']} D={res['debate']['score']} → {res['overall']}")
        except Exception as e:
            print(f"     ❌ {type(e).__name__}: {str(e)[:100]}")
            rows.append({'condition': label, 'run_idx': i, 'snapshot': fname, 'n_tasks': len(wbs),
                         'judge_structure':-1,'judge_assignment':-1,'judge_debate':-1,'judge_overall':-1,
                         'structure_reason':f'exc:{e}','assignment_reason':'','debate_reason':''})
        time.sleep(2)

out = '/home/piai/ai_course/agent_test/eval_results/hetero_backbone_experiment/summary_hetero.csv'
if not rows:
    raise SystemExit('No snapshots were evaluated; summary file was not changed.')

valid_rows = [r for r in rows if r.get('judge_overall', -1) >= 0]
if not valid_rows:
    failed_out = out.replace('.csv', '.failed_attempt.csv')
    with open(failed_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    raise SystemExit(f'All judge calls failed; kept existing summary unchanged. Failed attempt saved: {failed_out}')

with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\n💾 Saved: {out}  ({len(rows)} rows)")
