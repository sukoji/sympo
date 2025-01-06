"""Compaction 실험 9 snapshots batch re-judge.
**Judge 모델**: gemini-3.1-flash-lite-preview (Pro Preview 503 지속으로 대체 사용).
주의: 이전 ablation 실험들과 Judge 모델 다름 → compaction-내부 비교만 valid.

기존 summary_finegrain.csv와 동일 컬럼 구조의 summary_rejudge.csv 생성.
"""
import json, os, sys, glob, csv, time
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
# Judge 모델 override — Pro Preview 회복 후 사용
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-pro-preview'
from eval.llm_judge import evaluate_wbs

EVAL_DIMS = ['structure', 'assignment', 'debate']  # C3 = full system

def load_team():
    from data_pipeline.member_parser import MemberParser
    md = '/home/piai/ai_course/agent_test/sample_data/sample_members'
    return [MemberParser.from_resume_text(open(os.path.join(md,f)).read(),
                                          f.replace('member_','').replace('.txt',''))
            for f in sorted(os.listdir(md)) if f.endswith('.txt')]

team = load_team()

# Gather all 9 snapshots
patterns = [
    'wbs_snapshot_C3_3rounds_*compact_chrono_w8*.json',  # C_filter
    'wbs_snapshot_C3_3rounds_*compact_off*.json',         # C_off
    'wbs_snapshot_C3_3rounds_*compact_claude*.json',      # C_claude
]
fps = []
for p in patterns:
    fps.extend(sorted(glob.glob(f'/home/piai/ai_course/agent_test/eval_results/{p}')))

print(f"Found {len(fps)} snapshots")

rows = []
for i, fp in enumerate(fps, 1):
    fname = os.path.basename(fp)
    # Determine mode from filename
    if 'chrono_w8' in fname: mode = 'C_filter'
    elif 'compact_off' in fname: mode = 'C_off'
    elif 'compact_claude' in fname: mode = 'C_claude'
    else: mode = 'unknown'
    # Run id from filename
    run_id = fname.split('_r')[1].split('_')[0]

    d = json.load(open(fp))
    wbs = d.get('wbs_tasks', [])
    debate = d.get('debate_log', [])

    print(f"\n[{i}/{len(fps)}] {mode} r{run_id}  n_tasks={len(wbs)}")
    try:
        res = evaluate_wbs(wbs_tasks=wbs, team_members=team, debate_log=debate,
                           eval_dims=EVAL_DIMS, cross_judge=False)
        rows.append({
            'mode': mode, 'run_id': run_id, 'snapshot': fname,
            'n_tasks': len(wbs),
            'judge_structure':  res['structure']['score'],
            'judge_assignment': res['assignment']['score'],
            'judge_debate':     res['debate']['score'],
            'judge_overall':    res['overall'],
            'structure_reason':  res['structure']['reason'][:300],
            'assignment_reason': res['assignment']['reason'][:300],
            'debate_reason':     res['debate']['reason'][:300],
        })
        print(f"   ✅ S={res['structure']['score']} A={res['assignment']['score']} D={res['debate']['score']} → {res['overall']}")
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: {str(e)[:150]}")
        rows.append({
            'mode': mode, 'run_id': run_id, 'snapshot': fname,
            'n_tasks': len(wbs),
            'judge_structure': -1, 'judge_assignment': -1, 'judge_debate': -1, 'judge_overall': -1,
            'structure_reason': f'exc:{e}', 'assignment_reason': '', 'debate_reason': '',
        })
    time.sleep(2)

out_path = '/home/piai/ai_course/agent_test/eval_results/compaction_experiment/summary_rejudge.csv'
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\n💾 Saved: {out_path}")
