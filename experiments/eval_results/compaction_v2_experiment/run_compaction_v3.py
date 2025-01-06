"""Compaction v2 — 학술적·안전한 재설계.

가설:
  H1: C_claude > C_filter > C_minimal (압축 정교화 → quality)
  H2: latency C_claude > C_filter > C_minimal (요약 LLM 추가)
  H3: 모든 mode의 sub-agent failure < 10% (시스템 안전 영역)

IV: compaction_mode ∈ {C_minimal, C_filter, C_claude}
통제: Gemma-4-26B, C3_3rounds, PRD/팀, Pro Preview Judge
N=3, 코드 무수정 (monkey-patch)

C_minimal: last 30 raw + PM last 10 — "거의 압축 안 함" 안전 baseline
C_filter:  sliding W=8 + PM 4 — 현 default
C_claude:  Claude Code 패턴 (4K 임계 + Gemma26 self-summary + 캐싱 + last 6 raw)
"""
import sys, os, time
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
# Pro Preview Judge 사용 (회복 후)
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-pro-preview'

from agents import sub_agents

# Re-use compactors from v2 module (claude logic)
sys.path.insert(0, '/home/piai/ai_course/agent_test/eval_results/compaction_experiment')
from run_compaction_v2 import (
    compactor_claude, _CompactState, _is_pm_decision, _format_msg,
)

# Override Claude params for v2
import run_compaction_v2 as _v2
_v2.TOKEN_THRESHOLD = 4000
_v2.KEEP_RECENT = 6
_v2.MIN_TO_SUMMARIZE = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C_minimal: last 30 messages raw + PM last 10 (안 깨지는 약압축)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compactor_minimal(state, max_messages=8):
    logs = state.get("debate_log", [])
    if not logs:
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    pm = [m for m in logs if _is_pm_decision(m)]
    out = []
    if pm:
        out.append("=== PM 확정 결정사항 (마지막 10개) ===")
        for m in pm[-10:]:
            out.append(f"[{m.agent_name}]: {m.message}")
        out.append("=" * 50)
    out.append(f"--- 최근 토론 (last 30) ---")
    for m in logs[-30:]:
        out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C_filter: 현 default (sliding W=8 + PM 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compactor_filter(state, max_messages=8):
    logs = state.get("debate_log", [])
    if not logs:
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    pm = [m for m in logs if _is_pm_decision(m)]
    out = []
    if pm:
        out.append("=== PM 확정 결정사항 (이미 반영됨 — 재요구 불필요) ===")
        for m in pm[-4:]:
            out.append(f"[{m.agent_name}]: {m.message}")
        out.append("=" * 50)
    out.append("--- 최근 토론 내역 ---")
    for m in logs[-8:]:
        out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)

MODES = {
    'minimal': compactor_minimal,
    'filter':  compactor_filter,
    'claude':  compactor_claude,  # from v2 module
}

# Sample capture wrapper
def _wrap_capture(mode_name, fn):
    samples_dir = '/home/piai/ai_course/agent_test/eval_results/compaction_v2_experiment/samples'
    os.makedirs(samples_dir, exist_ok=True)
    sample_path = f'{samples_dir}/{mode_name}.txt'
    seen = set()
    def wrapped(state, max_messages=8):
        result = fn(state, max_messages=max_messages)
        log_len = len(state.get('debate_log', []))
        if log_len not in seen and log_len > 0:
            seen.add(log_len)
            with open(sample_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"# Mode: {mode_name}  | log size: {log_len} | output chars: {len(result)}\n")
                f.write(f"{'='*80}\n")
                f.write(result)
                f.write("\n")
        return result
    return wrapped

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(f"Usage: {sys.argv[0]} {{minimal|filter|claude}} [n_runs=3]")
        sys.exit(1)
    mode = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    sub_agents._format_debate_history = _wrap_capture(mode, MODES[mode])
    print(f"🔧 Patched _format_debate_history with mode='{mode}'")
    os.environ['RUNNER_ID'] = f'compactv2_{mode}'

    from eval.experiment_runner import run_all_experiments
    run_all_experiments(
        backend='qwen-api',  # endpoint label (실제 모델 = Gemma-4-26B GGUF)
        runs_per_condition=n_runs,
        conditions=['C3_3rounds'],
        harness_settings=None,
        cross_judge=False,
    )
