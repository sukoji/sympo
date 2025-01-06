"""Compaction strategy experiment — agents/orchestration code 안 건드리고 monkey-patch.
Backend: Gemma26 (이미 .env에 세팅됨).
조건: C3_3rounds 고정, 4 compaction strategy × N=3.

Strategies:
  - chrono_w8  : 현 default (last 8 msg + PM decisions last 4)
  - chrono_w4  : aggressive (last 4 msg + PM decisions last 2)
  - chrono_w16 : loose (last 16 msg + PM decisions last 4)
  - decision_only : PM decisions만 last 8 + last 2 chatter (Claude Code의 priority retention 흉내)

사용:
  python eval_results/compaction_experiment/run_compaction.py <strategy>
"""
import sys, os
sys.path.insert(0, '/home/piai/ai_course/agent_test')

from dotenv import load_dotenv
load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Monkey-patch BEFORE importing experiment_runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from agents import sub_agents

_orig_format_debate = sub_agents._format_debate_history

def patched_chrono(state, max_messages=8, _W=8, _PM=4):
    """현 default와 동일 — sliding window + PM decisions. _W/_PM로 강도 조절."""
    logs = state.get("debate_log", [])
    if not logs:
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    pm_decisions = [m for m in logs if m.message_type in ("mediation","decision")
                    and ("SUPERVISOR" in str(m.agent_role) or "슈퍼바이저" in str(m.agent_role) or "PM" in str(m.agent_name))]
    lines = []
    if pm_decisions:
        lines.append("=== PM 확정 결정사항 (이미 반영됨 — 재요구 불필요) ===")
        for msg in pm_decisions[-_PM:]:
            lines.append(f"[{msg.agent_name}]: {msg.message}")
        lines.append("=" * 50)
    recent = logs[-_W:]
    lines.append("--- 최근 토론 내역 ---")
    for msg in recent:
        role_v = msg.agent_role.value if hasattr(msg.agent_role, 'value') else msg.agent_role
        lines.append(f"[{msg.agent_name} / {role_v}]: {msg.message}")
    lines.append("--------------------")
    return "\n".join(lines)

def patched_decision_only(state, max_messages=8):
    """Decision priority retention — PM 결정사항만 최대한 보존, 일반 chatter는 마지막 2개만."""
    logs = state.get("debate_log", [])
    if not logs:
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    pm_decisions = [m for m in logs if m.message_type in ("mediation","decision")
                    and ("SUPERVISOR" in str(m.agent_role) or "슈퍼바이저" in str(m.agent_role) or "PM" in str(m.agent_name))]
    chatter = [m for m in logs if m not in pm_decisions]
    lines = []
    if pm_decisions:
        lines.append("=== PM 확정 결정사항 (전체 retain) ===")
        for msg in pm_decisions[-8:]:  # PM decisions 더 많이 보존
            lines.append(f"[{msg.agent_name}]: {msg.message}")
        lines.append("=" * 50)
    if chatter:
        lines.append("--- 최근 일반 발언 (last 2) ---")
        for msg in chatter[-2:]:
            role_v = msg.agent_role.value if hasattr(msg.agent_role, 'value') else msg.agent_role
            lines.append(f"[{msg.agent_name} / {role_v}]: {msg.message}")
        lines.append("--------------------")
    return "\n".join(lines)

STRATEGIES = {
    'chrono_w8':      lambda s, max_messages=8: patched_chrono(s, _W=8, _PM=4),
    'chrono_w4':      lambda s, max_messages=8: patched_chrono(s, _W=4, _PM=2),
    'chrono_w16':     lambda s, max_messages=8: patched_chrono(s, _W=16, _PM=4),
    'decision_only':  patched_decision_only,
}

# Sample capture wrapper — 매 호출마다 결과 저장 (debate_log 길이별로 샘플 1개씩)
def _wrap_with_sample_capture(strategy_name, fn):
    samples_dir = '/home/piai/ai_course/agent_test/eval_results/compaction_experiment/samples'
    os.makedirs(samples_dir, exist_ok=True)
    sample_path = f'{samples_dir}/{strategy_name}.txt'
    seen_lengths = set()
    def wrapped(state, max_messages=8):
        result = fn(state, max_messages=max_messages)
        log_len = len(state.get('debate_log', []))
        # 새로운 debate_log 길이마다 샘플 1개 저장 (총 ~10개)
        if log_len not in seen_lengths and log_len > 0:
            seen_lengths.add(log_len)
            with open(sample_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"# Strategy: {strategy_name}  |  debate_log length at call: {log_len}\n")
                f.write(f"# Returned text length (chars): {len(result)}\n")
                f.write(f"{'='*80}\n")
                f.write(result)
                f.write(f"\n")
        return result
    return wrapped

if __name__ == '__main__':
    strategy = sys.argv[1] if len(sys.argv) > 1 else 'chrono_w8'
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if strategy not in STRATEGIES:
        print(f"Unknown strategy: {strategy}. Available: {list(STRATEGIES.keys())}")
        sys.exit(1)

    # Apply patch + sample capture wrapper
    sub_agents._format_debate_history = _wrap_with_sample_capture(strategy, STRATEGIES[strategy])
    print(f"🔧 Monkey-patched _format_debate_history with strategy='{strategy}' (sample capture ON)")

    # Override RUNNER_ID to identify this batch
    os.environ['RUNNER_ID'] = f'compact_{strategy}'

    # Direct in-process call so monkey-patch remains active
    from eval.experiment_runner import run_all_experiments
    run_all_experiments(
        backend='qwen-api',
        runs_per_condition=n_runs,
        conditions=['C3_3rounds'],
        harness_settings=None,
        cross_judge=False,
    )
