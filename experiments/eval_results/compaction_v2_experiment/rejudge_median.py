"""각 snapshot을 3회 judge → median 사용 (Pro Preview 비결정성 통제).
출력: summary_judged.csv  (mode, run_id, S/A/D × 3 trials, median, sub_agent_failures)
"""
import json, os, sys, glob, csv, time, statistics
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-pro-preview'
from eval.llm_judge import evaluate_wbs

EVAL_DIMS = ['structure', 'assignment', 'debate']
EXP = '/home/piai/ai_course/agent_test/eval_results/compaction_v2_experiment'

def load_team():
    from data_pipeline.member_parser import MemberParser
    md = '/home/piai/ai_course/agent_test/sample_data/sample_members'
    return [MemberParser.from_resume_text(open(os.path.join(md,f)).read(),
                                          f.replace('member_','').replace('.txt',''))
            for f in sorted(os.listdir(md)) if f.endswith('.txt')]

team = load_team()

# Count sub-agent failures (error strings in debate_log messages)
def count_subagent_failures(debate_log):
    cnt = 0
    for m in debate_log:
        msg = m.message if hasattr(m, 'message') else m.get('message', '')
        if 'QwenAPI 호출 실패' in msg or 'HTTPError' in msg or 'Bad Request' in msg:
            cnt += 1
    return cnt

fps = sorted(glob.glob(f'{EXP}/snapshots/wbs_snapshot_C3_3rounds_*compactv2_*.json'))
print(f"Found {len(fps)} v2 snapshots")
rows = []
for i, fp in enumerate(fps, 1):
    fname = os.path.basename(fp)
    if 'compactv2_minimal' in fname: mode = 'C_minimal'
    elif 'compactv2_filter' in fname: mode = 'C_filter'
    elif 'compactv2_claude' in fname: mode = 'C_claude'
    else: mode = '?'
    run_id = fname.split('_r')[1].split('_')[0]
    d = json.load(open(fp))
    wbs = d.get('wbs_tasks', [])
    debate = d.get('debate_log', [])
    failures = count_subagent_failures(debate)
    print(f"\n[{i}/{len(fps)}] {mode} r{run_id}  n_tasks={len(wbs)}  sub_agent_failures={failures}")
    # Judge 3 times
    s_vals, a_vals, d_vals = [], [], []
    for trial in range(3):
        try:
            res = evaluate_wbs(wbs_tasks=wbs, team_members=team, debate_log=debate,
                               eval_dims=EVAL_DIMS, cross_judge=False)
            s_vals.append(res['structure']['score'])
            a_vals.append(res['assignment']['score'])
            d_vals.append(res['debate']['score'])
            print(f"   trial {trial+1}: S={res['structure']['score']} A={res['assignment']['score']} D={res['debate']['score']}")
        except Exception as e:
            print(f"   trial {trial+1} ❌ {type(e).__name__}: {str(e)[:100]}")
            s_vals.append(-1); a_vals.append(-1); d_vals.append(-1)
        time.sleep(2)
    # Median (drop -1)
    def med(vs):
        valid = [v for v in vs if v >= 0]
        return statistics.median(valid) if valid else -1
    s_med, a_med, d_med = med(s_vals), med(a_vals), med(d_vals)
    # Re-normalized overall
    W = {'S': 0.40, 'A': 0.35, 'D': 0.25}
    valid = []
    if s_med >= 0: valid.append(('S', s_med, W['S']))
    if a_med >= 0: valid.append(('A', a_med, W['A']))
    if d_med >= 0: valid.append(('D', d_med, W['D']))
    if valid:
        wsum = sum(w for _,_,w in valid)
        overall_med = sum(v*w for _,v,w in valid) / wsum
    else:
        overall_med = -1
    rows.append({
        'mode': mode, 'run_id': run_id, 'snapshot': fname, 'n_tasks': len(wbs),
        'sub_agent_failures': failures,
        'S_t1': s_vals[0], 'S_t2': s_vals[1], 'S_t3': s_vals[2], 'S_med': s_med,
        'A_t1': a_vals[0], 'A_t2': a_vals[1], 'A_t3': a_vals[2], 'A_med': a_med,
        'D_t1': d_vals[0], 'D_t2': d_vals[1], 'D_t3': d_vals[2], 'D_med': d_med,
        'overall_median': overall_med,
    })
    print(f"   → MEDIAN: S={s_med} A={a_med} D={d_med} → Overall={overall_med:.4f}")

out = f'{EXP}/summary_judged.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\n💾 Saved: {out}")
