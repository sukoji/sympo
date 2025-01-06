"""Hetero-backbone experiment — 특정 agent에 다른 백본 할당.
Backbone fixed: Gemma26 (qwen-api at localhost:8081).
Frontier specialization candidate: gemini-3.1-flash-lite-preview (API).

조건:
  H_baseline   : 모든 agent Gemma26 (재사용 — gemma26_ablation/C3 N=3)
  H_wbsgen     : WBS Gen만 frontier (Gemini Flash Lite), 나머지 Gemma26
  H_taskmgr    : Task Mgr (task_match + finalize)만 frontier, 나머지 Gemma26
  H_both       : WBS Gen + Task Mgr 둘 다 frontier
  H_all_frontier : 전부 frontier (재사용 — gemini_ablation/C3 N=3)

코드 무수정 — `agents.llm_config.get_llm`을 caller-aware monkey-patch.
사용:
  python run_hetero.py wbsgen [N=3]
  python run_hetero.py taskmgr [N=3]
  python run_hetero.py both [N=3]
"""
import sys, os, inspect
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')

from agents import llm_config
_orig_get_llm = llm_config.get_llm

# Backend assignment per condition (caller module → backend)
ASSIGNMENTS = {
    'wbsgen':  {'wbs_gen_agent': 'gemini'},
    'taskmgr': {'supervisor_agent': 'gemini'},  # task_match + finalize 둘 다 supervisor_agent.py에 있음
    'both':    {'wbs_gen_agent': 'gemini', 'supervisor_agent': 'gemini'},
}
DEFAULT_BACKEND = 'qwen-api'  # Gemma26 fallback

def make_patched_get_llm(role_map):
    def patched(temperature=0.7, max_tokens=1024, model_id=None):
        # Walk stack to find the agent module that called us
        for frame in inspect.stack()[1:]:
            fn = frame.filename
            for module_key, backend in role_map.items():
                if module_key in fn:
                    old = os.environ.get('LLM_BACKEND')
                    os.environ['LLM_BACKEND'] = backend
                    # Reset cached singletons so the new backend takes effect
                    llm_config._qwen_api_instance = None
                    try:
                        return _orig_get_llm(temperature, max_tokens, model_id)
                    finally:
                        if old: os.environ['LLM_BACKEND'] = old
                        else: os.environ.pop('LLM_BACKEND', None)
        # default
        return _orig_get_llm(temperature, max_tokens, model_id)
    return patched

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ASSIGNMENTS:
        print(f"Usage: {sys.argv[0]} {{wbsgen|taskmgr|both}} [n_runs=3]")
        sys.exit(1)
    cond = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    role_map = ASSIGNMENTS[cond]
    llm_config.get_llm = make_patched_get_llm(role_map)
    print(f"🔧 Hetero patch active for '{cond}': {role_map}")
    os.environ['RUNNER_ID'] = f'hetero_{cond}'

    from eval.experiment_runner import run_all_experiments
    run_all_experiments(
        backend='qwen-api',  # default backend (Gemma26)
        runs_per_condition=n_runs,
        conditions=['C3_3rounds'],
        harness_settings=None,
        cross_judge=False,
    )
