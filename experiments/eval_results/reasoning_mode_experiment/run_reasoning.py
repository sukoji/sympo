"""Reasoning mode ablation (Design A — uniform across all agents).

가설:
  H1: R_high > R_none (CoT 지시가 quality 개선)
  H2: R_max ≥ R_high (더 강한 reasoning은 marginal value)
  H3: token_cost / latency: R_max > R_high > R_none

IV: reasoning_instruction_strength ∈ {none, high, max}
모든 agent (wbs_gen, supervisor, sub_agents)에 동일 reasoning prefix 주입.

통제: Gemma-4-26B, C3_3rounds, PRD/팀, Pro Preview Judge
N=3, 코드 무수정 (LLM wrapper monkey-patch)

사용:
  python run_reasoning.py none [N=3]
  python run_reasoning.py high [N=3]
  python run_reasoning.py max  [N=3]
"""
import sys, os
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-pro-preview'

from agents import llm_config

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reasoning prompt prefix per mode (Korean, prepended to every LLM call)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING_PREFIX = {
    'none': '',
    'high': (
        "[추론 지시] 답변 전에 단계적으로 추론하라. "
        "결정 전 가능한 옵션을 비교하고 trade-off를 명시하라.\n\n"
    ),
    'max': (
        "[추론 지시 — 심층 분석 모드]\n"
        "1) 먼저 문제의 핵심 제약과 가정을 명시\n"
        "2) 최소 2개 대안을 도출하고 각각의 장단점 비교\n"
        "3) trade-off 명시적 평가\n"
        "4) 최적 결정과 그 근거 제시\n"
        "위 단계를 모두 수행한 후 최종 답변을 작성하라.\n\n"
    ),
}

_orig_get_llm = llm_config.get_llm

def prepend_reasoning_prefix(prompt, prefix: str):
    """Prepend reasoning prefix to string prompts and chat-message prompts."""
    if not prefix:
        return prompt
    if isinstance(prompt, str):
        return prefix + prompt
    if isinstance(prompt, list):
        patched = []
        applied = False
        for item in prompt:
            if isinstance(item, dict):
                new_item = dict(item)
                content = new_item.get('content')
                if not applied and isinstance(content, str):
                    new_item['content'] = prefix + content
                    applied = True
                patched.append(new_item)
            else:
                patched.append(item)
        return patched
    return prompt

def make_patched_get_llm(prefix: str):
    """Returns a get_llm replacement that prepends reasoning prefix to invoke prompt."""
    def patched(temperature=0.7, max_tokens=1024, model_id=None):
        llm = _orig_get_llm(temperature=temperature, max_tokens=max_tokens, model_id=model_id)
        if not prefix:
            return llm
        _orig_invoke = llm.invoke
        def patched_invoke(prompt, *args, **kwargs):
            return _orig_invoke(prepend_reasoning_prefix(prompt, prefix), *args, **kwargs)
        llm.invoke = patched_invoke
        return llm
    return patched

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in REASONING_PREFIX:
        print(f"Usage: {sys.argv[0]} {{none|high|max}} [n_runs=3]")
        sys.exit(1)
    mode = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    llm_config.get_llm = make_patched_get_llm(REASONING_PREFIX[mode])
    print(f"🔧 Patched get_llm with reasoning_mode='{mode}' (prefix len={len(REASONING_PREFIX[mode])})")
    os.environ['RUNNER_ID'] = f'reasoning_{mode}'

    from eval.experiment_runner import run_all_experiments
    run_all_experiments(
        backend='qwen-api',  # Gemma-4-26B endpoint
        runs_per_condition=n_runs,
        conditions=['C3_3rounds'],
        harness_settings=None,
        cross_judge=False,
    )
