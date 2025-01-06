"""Re-judge v4 snapshots × 3 trials with Flash Lite."""
import json, os, sys, glob, csv, time, statistics
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-flash-lite-preview'
from eval.llm_judge import evaluate_wbs

EXP = '/home/piai/ai_course/agent_test/eval_results/compaction_v4_experiment'
EVAL_DIMS = ['structure','assignment','debate']

def load_team():
    from data_pipeline.member_parser import MemberParser
    md = '/home/piai/ai_course/agent_test/sample_data/sample_members'
    return [MemberParser.from_resume_text(open(os.path.join(md,f)).read(),
            f.replace('member_','').replace('.txt',''))
            for f in sorted(os.listdir(md)) if f.endswith('.txt')]

def count_failures(debate_log):
    cnt=0
    for m in debate_log:
        msg = m.message if hasattr(m,'message') else m.get('message','')
        if 'QwenAPI 호출 실패' in msg or 'HTTPError' in msg or 'Bad Request' in msg:
            cnt += 1
    return cnt

team = load_team()
fps = sorted(glob.glob(f'{EXP}/snapshots/wbs_snapshot_C3_3rounds_*compactv4_*.json'))
print(f"Found {len(fps)} v4 snapshots", flush=True)
rows = []
for i, fp in enumerate(fps, 1):
    fname = os.path.basename(fp)
    if 'compactv4_minimal' in fname: mode = 'C_minimal'
    elif 'compactv4_filter' in fname: mode = 'C_filter'
    elif 'compactv4_claude' in fname: mode = 'C_claude'
    else: mode = '?'
    run_id = fname.split('_r')[1].split('_')[0]
    d = json.load(open(fp))
    wbs = d.get('wbs_tasks', [])
    debate = d.get('debate_log', [])
    failures = count_failures(debate)
    print(f"\n[{i}/{len(fps)}] {mode} r{run_id} n_tasks={len(wbs)} failures={failures}", flush=True)
    s_vals, a_vals, d_vals = [], [], []
    for trial in range(3):
        try:
            res = evaluate_wbs(wbs_tasks=wbs, team_members=team, debate_log=debate,
                               eval_dims=EVAL_DIMS, cross_judge=False)
            s_vals.append(res['structure']['score'])
            a_vals.append(res['assignment']['score'])
            d_vals.append(res['debate']['score'])
            print(f"  trial {trial+1}: S={res['structure']['score']} A={res['assignment']['score']} D={res['debate']['score']}", flush=True)
        except Exception as e:
            print(f"  trial {trial+1} ❌ {e}", flush=True)
            s_vals.append(-1); a_vals.append(-1); d_vals.append(-1)
        time.sleep(2)
    def med(vs):
        v = [x for x in vs if x>=0]
        return statistics.median(v) if v else -1
    sm, am, dm = med(s_vals), med(a_vals), med(d_vals)
    W={'S':0.40,'A':0.35,'D':0.25}; valid=[]
    if sm>=0: valid.append(('S',sm,W['S']))
    if am>=0: valid.append(('A',am,W['A']))
    if dm>=0: valid.append(('D',dm,W['D']))
    overall = sum(v*w for _,v,w in valid)/sum(w for _,_,w in valid) if valid else -1
    rows.append({
        'mode':mode,'run_id':run_id,'snapshot':fname,'n_tasks':len(wbs),
        'sub_agent_failures':failures,
        'S_t1':s_vals[0],'S_t2':s_vals[1],'S_t3':s_vals[2],'S_med':sm,
        'A_t1':a_vals[0],'A_t2':a_vals[1],'A_t3':a_vals[2],'A_med':am,
        'D_t1':d_vals[0],'D_t2':d_vals[1],'D_t3':d_vals[2],'D_med':dm,
        'overall_median':overall,
    })
    print(f"  → MEDIAN: S={sm} A={am} D={dm} → Overall={overall:.4f}", flush=True)

out=f'{EXP}/summary_judged.csv'
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\n💾 Saved: {out}")
