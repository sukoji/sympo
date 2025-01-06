"""저장된 wbs_snapshot 파일들에 새 fine-grained 루브릭 재심사.
사용: python eval_results/rejudge_snapshots.py <pattern>
예:   python eval_results/rejudge_snapshots.py 'wbs_snapshot_*_gemma4-api_*'
"""
import json, os, sys, glob, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv; load_dotenv()
from eval.llm_judge import evaluate_wbs

# 팀원 로드 (snapshot에 없으면 sample_data에서)
def load_team():
    from data_pipeline.member_parser import MemberParser
    member_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_members")
    team = []
    if os.path.isdir(member_dir):
        for fname in sorted(os.listdir(member_dir)):
            if fname.endswith(".txt"):
                content = open(os.path.join(member_dir, fname), encoding="utf-8").read()
                name = fname.replace("member_", "").replace(".txt", "")
                p = MemberParser.from_resume_text(content, name)
                team.append(p)
    return team

# 조건별 eval_dims 매핑 (experiment_runner와 동일)
EVAL_DIMS_BY_COND = {
    "C0_llm_only":     ["structure"],
    "C1_with_assign":  ["structure", "assignment"],
    "C2_1round":       ["structure", "assignment", "debate"],
    "C3_3rounds":      ["structure", "assignment", "debate"],
    "C4_with_disc":    ["structure", "assignment", "debate"],
    "C5_5rounds":      ["structure", "assignment", "debate"],
}


def parse_snapshot(path):
    s = json.load(open(path))
    fname = os.path.basename(path)
    # filename pattern: wbs_snapshot_{cond}_r{N}_{backend}_{runner}_{ts}.json
    m = fname.replace("wbs_snapshot_", "").rsplit(".", 1)[0]
    parts = m.split("_r")
    cond = parts[0]
    rest = parts[1].split("_", 1)
    run_id = int(rest[0])
    return cond, run_id, s


def main(pattern):
    team = load_team()
    files = sorted(glob.glob(f"eval_results/{pattern}.json"))
    if not files:
        print(f"파일 없음: {pattern}")
        return

    out_rows = []
    for i, fp in enumerate(files, 1):
        cond, run_id, snap = parse_snapshot(fp)
        wbs = snap.get("wbs_tasks", [])
        debate_log = snap.get("debate_log", [])
        eval_dims = EVAL_DIMS_BY_COND.get(cond, ["structure", "assignment", "debate"])

        n_tasks = len(wbs)
        if n_tasks == 0:
            print(f"[{i}/{len(files)}] {cond} r{run_id}: tasks=0 SKIP")
            continue

        print(f"[{i}/{len(files)}] {cond} r{run_id}: tasks={n_tasks} dims={eval_dims} ...")
        try:
            result = evaluate_wbs(
                wbs_tasks=wbs,
                team_members=team,
                debate_log=debate_log,
                eval_dims=eval_dims,
                cross_judge=False,
            )
        except Exception as e:
            print(f"  ❌ judge fail: {e}")
            continue

        out_rows.append({
            "condition": cond,
            "run_id": run_id,
            "snapshot": os.path.basename(fp),
            "n_tasks": n_tasks,
            "structure": result["structure"]["score"],
            "assignment": result["assignment"]["score"],
            "debate": result["debate"]["score"],
            "overall": result["overall"],
            "structure_reason": result["structure"]["reason"][:120],
            "assignment_reason": result["assignment"]["reason"][:120],
            "debate_reason": result["debate"]["reason"][:120],
        })
        print(f"  ✅ S={result['structure']['score']} A={result['assignment']['score']} "
              f"D={result['debate']['score']} → Overall={result['overall']}")
        time.sleep(0.5)  # rate-limit 완화

    # 결과 CSV 저장
    if not out_rows:
        print("결과 없음")
        return
    backend = pattern.split("_")[-2] if "_" in pattern else "unknown"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"eval_results/rejudge_finegrain_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader(); w.writerows(out_rows)
    print(f"\n📋 저장: {out_path}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "wbs_snapshot_*_gemma4-api_piai_*"
    main(pattern)
