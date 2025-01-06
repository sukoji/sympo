"""
LangSmith Evaluator — 매 테스트마다 13개 지표 + 종합 점수를 LangSmith에 자동 저장
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
  # Dataset 생성 (1회)
  python eval/langsmith_evaluator.py --create-dataset

  # 평가 실행 (Dataset의 각 조건에 대해 WBS 생성 + 13개 지표 평가)
  python eval/langsmith_evaluator.py --run --backend gemini

  # 특정 조건만
  python eval/langsmith_evaluator.py --run --backend gemini --conditions C0_llm_only C3_3rounds
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Dataset 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_NAME = "symPO-WBS-Eval"

def create_dataset():
    """평가용 Dataset 생성 — 각 조건을 하나의 Example로"""
    from eval.experiment_runner import CONDITIONS

    # 기존 Dataset 삭제 후 재생성
    for ds in client.list_datasets(dataset_name=DATASET_NAME):
        client.delete_dataset(dataset_id=ds.id)

    ds = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="symPO WBS 생성 평가 — Ablation + RAG 조건별 자동 평가",
    )

    inputs = []
    for key, cond in CONDITIONS.items():
        inputs.append({
            "condition": key,
            "label": cond["label"],
            "max_rounds": cond["max_rounds"],
            "use_task_match": cond["use_task_match"],
            "use_disc": cond["use_disc"],
            "rag_strategy": cond.get("rag_strategy", ""),
            "use_meeting": cond.get("use_meeting", False),
        })

    client.create_examples(
        inputs=inputs,
        dataset_id=ds.id,
    )
    print(f"Dataset '{DATASET_NAME}' 생성 완료 — {len(inputs)}개 조건")
    return ds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Target 함수 (WBS 생성 파이프라인)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_prd = None
_team = None
_disc = None

def _load_shared_data():
    global _prd, _team, _disc
    if _prd is None:
        from eval.experiment_runner import _load_sample_prd, _load_sample_team, _load_disc_profiles
        _prd = _load_sample_prd()
        _team = _load_sample_team()
        _disc = _load_disc_profiles()


def wbs_target(inputs: dict) -> dict:
    """LangSmith evaluate()가 Dataset의 각 Example마다 호출하는 함수"""
    _load_shared_data()

    from eval.experiment_runner import CONDITIONS, run_single_experiment

    cond_key = inputs["condition"]
    cond = CONDITIONS.get(cond_key)
    if not cond:
        return {"error": f"Unknown condition: {cond_key}"}

    backend = os.environ.get("EVAL_BACKEND", "gemini")
    start = time.time()

    metrics = run_single_experiment(
        cond_key, cond, _prd, _team, _disc, backend, run_id=1,
    )

    elapsed = round(time.time() - start, 2)

    # LangSmith에 저장될 출력 — 평탄화
    return {
        "total_tasks": metrics.get("total_tasks", 0),
        "elapsed_sec": elapsed,
        "debate_rounds": metrics.get("debate_rounds", 0),
        # 13 metrics (flatten)
        "faithfulness": metrics.get("ragas_faithfulness", {}).get("faithfulness", 0),
        "success_rate": metrics.get("success_rate", {}).get("success_rate", 0),
        "mece_score": metrics.get("mece_score", {}).get("mece_score", 0),
        "granularity": metrics.get("granularity_fitness", {}).get("granularity_fitness", 0),
        "planning_score": metrics.get("planning_score", {}).get("planning_score", 0),
        "workload_gini": metrics.get("workload_gini", {}).get("gini", 0),
        "schedule_feasibility": metrics.get("schedule_feasibility", {}).get("feasibility", 0),
        "buffer_ratio_pct": metrics.get("buffer_ratio", {}).get("buffer_ratio_pct", 0),
        "convergence": 1.0 if metrics.get("convergence", {}).get("is_converging") else 0.0,
        "comm_efficiency": metrics.get("communication_efficiency", {}).get("efficiency", 0),
        "supervisor_ratio": metrics.get("supervisor_intervention", {}).get("intervention_ratio", 0),
        "interaction_turns": metrics.get("interaction_turns", {}).get("total_messages", 0),
        "est_tokens": metrics.get("token_cost", {}).get("est_total_tokens", 0),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Evaluators (각 지표를 LangSmith 점수로 변환)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _score(run, key):
    """run.outputs에서 키 추출"""
    return run.outputs.get(key, 0) if run.outputs else 0


# ── 생성 품질 ──
def eval_faithfulness(run, example):
    return {"key": "faithfulness", "score": _score(run, "faithfulness")}

def eval_success_rate(run, example):
    return {"key": "success_rate", "score": _score(run, "success_rate")}

def eval_mece(run, example):
    return {"key": "mece_score", "score": _score(run, "mece_score")}

def eval_granularity(run, example):
    return {"key": "granularity", "score": _score(run, "granularity")}


# ── 배분 품질 ──
def eval_planning(run, example):
    return {"key": "planning_score", "score": _score(run, "planning_score")}

def eval_gini(run, example):
    gini = _score(run, "workload_gini")
    return {"key": "workload_balance", "score": round(max(0, 1.0 - gini), 4)}

def eval_feasibility(run, example):
    return {"key": "schedule_feasibility", "score": _score(run, "schedule_feasibility")}

def eval_buffer(run, example):
    buf = _score(run, "buffer_ratio_pct")
    if 15 <= buf <= 30:
        score = 1.0
    elif 10 <= buf <= 40:
        score = 0.7
    elif buf > 0:
        score = 0.3
    else:
        score = 0.0
    return {"key": "buffer_adequacy", "score": score}


# ── 오케스트레이션 ──
def eval_comm_eff(run, example):
    return {"key": "comm_efficiency", "score": _score(run, "comm_efficiency")}

def eval_convergence(run, example):
    return {"key": "convergence", "score": _score(run, "convergence")}

def eval_supervisor(run, example):
    ratio = _score(run, "supervisor_ratio")
    return {"key": "supervisor_efficiency", "score": round(max(0, 1.0 - ratio), 4)}


# ── 종합 점수 (AlphaEval 방식 가중 루브릭) ──
def eval_wbs_score(run, example):
    """
    WBS_Score = 0.40 × GenQuality + 0.35 × AssignQuality + 0.25 × OrchEfficiency
    """
    o = run.outputs or {}

    # 생성 품질
    sr = o.get("success_rate", 0)
    mece = o.get("mece_score", 0)
    gran = o.get("granularity", 0)
    faith = o.get("faithfulness", 0)
    gen_q = 0.30 * sr + 0.25 * mece + 0.25 * gran + 0.20 * faith

    # 배분 품질
    plan = o.get("planning_score", 0)
    gini_inv = max(0, 1.0 - o.get("workload_gini", 0))
    feas = o.get("schedule_feasibility", 0)
    buf = o.get("buffer_ratio_pct", 0)
    buf_score = 1.0 if 15 <= buf <= 30 else (0.7 if 10 <= buf <= 40 else (0.3 if buf > 0 else 0.0))
    assign_q = 0.30 * plan + 0.30 * gini_inv + 0.20 * feas + 0.20 * buf_score

    # 오케스트레이션 효율
    comm = o.get("comm_efficiency", 0)
    sup_inv = max(0, 1.0 - o.get("supervisor_ratio", 0))
    conv = o.get("convergence", 0)
    orch_q = 0.40 * comm + 0.30 * sup_inv + 0.30 * conv

    # 종합
    wbs_score = round((0.40 * gen_q + 0.35 * assign_q + 0.25 * orch_q) * 100, 2)
    return {"key": "wbs_score", "score": wbs_score / 100}  # 0~1 정규화


ALL_EVALUATORS = [
    eval_faithfulness, eval_success_rate, eval_mece, eval_granularity,
    eval_planning, eval_gini, eval_feasibility, eval_buffer,
    eval_comm_eff, eval_convergence, eval_supervisor,
    eval_wbs_score,
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_evaluation(backend="gemini", conditions=None):
    """LangSmith evaluate() 실행"""
    os.environ["EVAL_BACKEND"] = backend

    # Dataset 존재 확인
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if not datasets:
        print(f"Dataset '{DATASET_NAME}' 없음 — 생성합니다...")
        create_dataset()

    # 조건 필터링
    if conditions:
        # 필터된 Dataset을 임시로 사용하기 위해 target에서 건너뛰기
        os.environ["EVAL_CONDITIONS"] = ",".join(conditions)

    from datetime import datetime
    prefix = f"sympo_{backend}_{datetime.now().strftime('%m%d_%H%M')}"

    print(f"\n{'='*60}")
    print(f"LangSmith Evaluation: {prefix}")
    print(f"Backend: {backend}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Evaluators: {len(ALL_EVALUATORS)}개")
    print(f"{'='*60}\n")

    results = evaluate(
        wbs_target,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=prefix,
        max_concurrency=1,  # WBS 생성은 순차
    )

    print(f"\n✅ 평가 완료 — LangSmith 대시보드에서 '{prefix}' 확인")
    print(f"   https://smith.langchain.com/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="symPO LangSmith Evaluator")
    parser.add_argument("--create-dataset", action="store_true", help="Dataset 생성")
    parser.add_argument("--run", action="store_true", help="평가 실행")
    parser.add_argument("--backend", default="gemini", choices=["mock", "gemini", "openai"])
    parser.add_argument("--conditions", nargs="*", default=None)
    args = parser.parse_args()

    if args.create_dataset:
        create_dataset()
    elif args.run:
        run_evaluation(backend=args.backend, conditions=args.conditions)
    else:
        print("사용법:")
        print("  python eval/langsmith_evaluator.py --create-dataset")
        print("  python eval/langsmith_evaluator.py --run --backend gemini")
