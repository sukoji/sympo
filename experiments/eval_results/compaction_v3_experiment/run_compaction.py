"""Compaction v3 RERUN — 32K context server (failure 회피).

3 modes, all monkey-patch (no project code modification):
  C_minimal:  last 30 + PM 10
  C_filter:   last 8 + PM 4 (현 default)
  C_claude:   v3 fixed (4K threshold + bounded REPLACE summary + last 6 + PM 4)

통제: Gemma-4-26B Q4_K_M (32K ctx now), C3_3rounds, Pro Preview Judge
N=3, 코드 무수정.
"""
import sys, os
sys.path.insert(0, '/home/piai/ai_course/agent_test')
sys.path.insert(0, '/home/piai/ai_course/agent_test/eval_results/compaction_experiment')
sys.path.insert(0, '/home/piai/ai_course/agent_test/eval_results/compaction_v2_experiment')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-pro-preview'

from agents import sub_agents
from compactor_claude_v3_fixed import compactor_claude_v3
from run_compaction_v2 import _is_pm_decision, _format_msg

def compactor_minimal(state, max_messages=8):
    logs = state.get("debate_log", [])
    if not logs: return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    pm = [m for m in logs if _is_pm_decision(m)]
    out = []
    if pm:
        out.append("=== PM 확정 결정사항 (마지막 10개) ===")
        for m in pm[-10:]: out.append(f"[{m.agent_name}]: {m.message}")
        out.append("=" * 50)
    out.append("--- 최근 토론 (last 30) ---")
    for m in logs[-30:]: out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)

def compactor_filter(state, max_messages=8):
    logs = state.get("debate_log", [])
    if not logs: return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    pm = [m for m in logs if _is_pm_decision(m)]
    out = []
    if pm:
        out.append("=== PM 확정 결정사항 (이미 반영됨 — 재요구 불필요) ===")
        for m in pm[-4:]: out.append(f"[{m.agent_name}]: {m.message}")
        out.append("=" * 50)
    out.append("--- 최근 토론 내역 ---")
    for m in logs[-8:]: out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)

MODES = {'minimal': compactor_minimal, 'filter': compactor_filter, 'claude': compactor_claude_v3}

def _wrap_capture(mode_name, fn):
    samples_dir = '/home/piai/ai_course/agent_test/eval_results/compaction_v3_experiment/samples'
    os.makedirs(samples_dir, exist_ok=True)
    sample_path = f'{samples_dir}/{mode_name}.txt'
    seen = set()
    def wrapped(state, max_messages=8):
        result = fn(state, max_messages=max_messages)
        log_len = len(state.get('debate_log', []))
        if log_len not in seen and log_len > 0:
            seen.add(log_len)
            with open(sample_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n# Mode: {mode_name} | log size: {log_len} | output chars: {len(result)}\n{'='*80}\n{result}\n")
        return result
    return wrapped

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(f"Usage: {sys.argv[0]} {{minimal|filter|claude}} [n_runs=3]")
        sys.exit(1)
    mode = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    sub_agents._format_debate_history = _wrap_capture(mode, MODES[mode])
    print(f"🔧 Patched mode='{mode}' (32K context server)")
    os.environ['RUNNER_ID'] = f'compactv3_{mode}'
    from eval.experiment_runner import run_all_experiments
    run_all_experiments(backend='qwen-api', runs_per_condition=n_runs,
                        conditions=['C3_3rounds'], harness_settings=None, cross_judge=False)
