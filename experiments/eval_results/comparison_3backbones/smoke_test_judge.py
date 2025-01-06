"""Judge 패치(rev.4) smoke test — gemma C3/C4/C5에서 A 점수 정상 회복 확인."""
import json, os, sys, glob
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv()
from eval.llm_judge import evaluate_wbs

def load_team():
    from data_pipeline.member_parser import MemberParser
    md = '/home/piai/ai_course/agent_test/sample_data/sample_members'
    return [MemberParser.from_resume_text(open(os.path.join(md,f)).read(),
                                          f.replace('member_','').replace('.txt',''))
            for f in sorted(os.listdir(md)) if f.endswith('.txt')]

team = load_team()
# Test 3 gemma snapshots that previously had A=N/A or A=0
TARGETS = [
    'wbs_snapshot_C3_3rounds_r2_gemma4-api_piai_20260421_020444.json',
    'wbs_snapshot_C4_with_disc_r2_gemma4-api_piai_20260421_031034.json',
    'wbs_snapshot_C5_5rounds_r1_gemma4-api_piai_20260421_040130.json',
]
for fn in TARGETS:
    fp = f'/home/piai/ai_course/agent_test/eval_results/gemma_ablation/snapshots/{fn}'
    d = json.load(open(fp))
    wbs = d.get('wbs_tasks', [])
    debate = d.get('debate_log', [])
    print(f"\n=== {fn[14:50]} (n_tasks={len(wbs)}) ===")
    res = evaluate_wbs(wbs_tasks=wbs, team_members=team, debate_log=debate,
                       eval_dims=['structure','assignment','debate'], cross_judge=False)
    for k in ['structure','assignment','debate']:
        v = res[k]
        print(f"  {k:10}: score={v['score']!s:>5}  reason={v['reason'][:130]!r}")
    print(f"  Overall: {res['overall']}")
