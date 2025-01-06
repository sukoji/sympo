"""Gemma26 — 503으로 실패한 3개 snapshot 재심사 + summary CSV in-place 업데이트."""
import json, os, sys, csv, glob, time
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv()
from eval.llm_judge import evaluate_wbs

def load_team():
    from data_pipeline.member_parser import MemberParser
    md = '/home/piai/ai_course/agent_test/sample_data/sample_members'
    return [MemberParser.from_resume_text(open(os.path.join(md,f)).read(),
                                          f.replace('member_','').replace('.txt',''))
            for f in sorted(os.listdir(md)) if f.endswith('.txt')]

EVAL_DIMS = {
    'C0_llm_only':['structure'],
    'C1_with_assign':['structure','assignment'],
    'C2_1round':['structure','assignment','debate'],
    'C3_3rounds':['structure','assignment','debate'],
    'C4_with_disc':['structure','assignment','debate'],
    'C5_5rounds':['structure','assignment','debate'],
}

team = load_team()
TARGETS = [
    ('C5_5rounds', '1', ['assignment']),         # only A failed
    ('C5_5rounds', '2', ['structure','assignment','debate']),
    ('C5_5rounds', '3', ['structure','assignment','debate']),
]

snap_dir = '/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/snapshots'
csv_path = '/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/summary_finegrain.csv'

# Read current CSV
rows = list(csv.DictReader(open(csv_path)))
fields = list(rows[0].keys())

for cond, run_id, dims in TARGETS:
    # Find matching snapshot
    pattern = f'{snap_dir}/wbs_snapshot_{cond}_r{run_id}_*.json'
    fps = glob.glob(pattern)
    if not fps:
        print(f"  ❌ {cond} r{run_id}: snapshot not found")
        continue
    fp = fps[0]
    snap = json.load(open(fp))
    wbs = snap.get('wbs_tasks', [])
    debate = snap.get('debate_log', [])
    print(f"\n  📌 {cond} r{run_id} (n={len(wbs)}) — re-judging dims: {dims}")
    res = evaluate_wbs(wbs_tasks=wbs, team_members=team, debate_log=debate,
                       eval_dims=dims, cross_judge=False)
    new_scores = {}
    for d in dims:
        new_scores[f'judge_{d}'] = res[d]['score']
        print(f"    {d}: {res[d]['score']}  reason={res[d]['reason'][:100]!r}")

    # Update CSV row
    for r in rows:
        if r['condition'] == cond and r['run_id'] == run_id:
            for k, v in new_scores.items():
                r[k] = str(v)
            # Recompute judge_overall using active dims with re-normalization
            S=0.40; A=0.35; D=0.25
            active=[]
            full_dims = EVAL_DIMS[cond]
            for d in full_dims:
                key = f'judge_{d}'
                try:
                    val = float(r.get(key,'-1'))
                    if val >= 0:
                        w = {'structure':S, 'assignment':A, 'debate':D}[d]
                        active.append((d, val, w))
                except: pass
            if active:
                wsum = sum(w for _,_,w in active)
                ov = sum(v*w for _,v,w in active) / wsum
                r['judge_overall'] = f'{ov:.4f}'
                print(f"    → new overall: {r['judge_overall']}")
            break
    time.sleep(2)

# Save updated CSV
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
print(f"\n💾 CSV updated: {csv_path}")
