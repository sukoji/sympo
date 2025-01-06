"""Compaction RQ1 experiment — C_off vs C_filter vs C_claude × C3 × N=3.

독립변수: `compaction_mode` ∈ {off, filter, claude}
  - off    : 전체 debate_log 그대로 (no filtering, no compaction)
  - filter : 현 default — sliding window W=8 + PM decisions priority retain (≡ chrono_w8)
  - claude : Claude Code 패턴 — token 임계 8K 초과 시 LLM 요약 + 캐싱, last 10 raw

통제: backend=Gemma26 (qwen-api at localhost:8081), 조건=C3_3rounds, PRD/팀=동일.
요약 LLM = 동일 Gemma26 (Claude Code 디자인).
프로젝트 코드 미수정 (monkey-patch only).

사용:
  python run_compaction_v2.py off    [N]
  python run_compaction_v2.py filter [N]   # already done as chrono_w8 if exists
  python run_compaction_v2.py claude [N]
"""
import sys, os, time, json
sys.path.insert(0, '/home/piai/ai_course/agent_test')

from dotenv import load_dotenv
load_dotenv()

from agents import sub_agents
from schemas.wbs_schema import DebateMessage

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compaction state (resets when new run detected via debate_log shrinkage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _CompactState:
    def __init__(self):
        self.summarized_prefix = ''
        self.summarized_until_idx = 0
        self.last_seen_log_size = 0
        self.summarize_call_count = 0
    def reset(self):
        self.summarized_prefix = ''
        self.summarized_until_idx = 0
        self.summarize_call_count = 0

_C = _CompactState()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper — PM decision detection (mirrors original code)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _is_pm_decision(m):
    if not hasattr(m, 'message_type'):
        return False
    return m.message_type in ("mediation", "decision") and (
        "SUPERVISOR" in str(m.agent_role) or
        "슈퍼바이저" in str(m.agent_role) or
        "PM" in str(m.agent_name)
    )

def _format_msg(m):
    role_v = m.agent_role.value if hasattr(m.agent_role, 'value') else m.agent_role
    return f"[{m.agent_name} / {role_v}]: {m.message}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mode 1: OFF — pass entire debate_log raw (worst-case baseline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compactor_off(state, max_messages=8):
    logs = state.get("debate_log", [])
    if not logs:
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    out = ["--- 전체 토론 내역 (no compaction) ---"]
    for m in logs:
        out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mode 2: FILTER — current default (sliding W=8 + PM priority 4)
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mode 3: CLAUDE — threshold-triggered LLM summarization with caching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOKEN_THRESHOLD = int(os.getenv('CLAUDE_THRESHOLD_TOKENS', '4000'))  # 4K로 낮춤 — sub-agent 16K 한계 도달 전 트리거
KEEP_RECENT = int(os.getenv('CLAUDE_KEEP_RECENT', '6'))               # 6개로 낮춤 — 더 작은 raw 영역
MIN_TO_SUMMARIZE = 3

MAX_SUMMARY_INPUT_CHARS = 6000  # Gemma26 16K token limit 안전 영역 (대략 5K 토큰 입력)

def _gemma26_summarize(messages_to_summarize):
    """Use Gemma26 (qwen-api backend) to summarize older debate messages.
    rev2: 입력 길이 제한 + 실패 감지 (QwenAPI 호출 실패 문자열 패턴)."""
    if not messages_to_summarize:
        return ''
    # 메시지별 본문 truncate (각 800자) + 전체 합 6000자 cap
    formatted = []
    total = 0
    for m in messages_to_summarize:
        msg_text = m.message[:800] if len(m.message) > 800 else m.message
        line = f"[{m.agent_name}]: {msg_text}"
        if total + len(line) > MAX_SUMMARY_INPUT_CHARS:
            formatted.append(f"... (이후 {len(messages_to_summarize) - len(formatted)}개 메시지 truncated)")
            break
        formatted.append(line)
        total += len(line)
    text = "\n".join(formatted)
    prompt = f"""다음 멀티에이전트 토론을 5~10줄로 요약하세요. 보존: 리스크·버퍼 일수·task ID·합의·이견. 생략: 인사·단순 동의.

토론:
{text}

요약:"""
    from agents.llm_config import get_llm
    llm = get_llm(temperature=0.3, max_tokens=400)
    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, 'content') else str(resp)
        # QwenAPI failure 패턴 감지 — error string을 요약으로 저장하지 않도록
        if "QwenAPI 호출 실패" in content or "HTTPError" in content or "Bad Request" in content:
            raise RuntimeError(f"Summarization API call returned error: {content[:200]}")
        return content.strip()
    except Exception as e:
        raise  # 위 호출 측에서 except로 잡혀 fallback 동작

def compactor_claude(state, max_messages=8):
    global _C
    logs = state.get("debate_log", [])
    if not logs:
        # New run started — reset state
        _C.reset()
        _C.last_seen_log_size = 0
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    # Detect new run via log shrinkage
    if len(logs) < _C.last_seen_log_size:
        _C.reset()
    _C.last_seen_log_size = len(logs)

    # Estimate tokens of unsummarized portion
    new_logs = logs[_C.summarized_until_idx:]
    chars = sum(len(m.message) for m in new_logs)
    est_tokens = int(chars * 1.2 / 3)  # rough Korean-mixed estimate

    # Trigger condition
    available_to_summarize = len(logs) - _C.summarized_until_idx - KEEP_RECENT
    if est_tokens > TOKEN_THRESHOLD and available_to_summarize >= MIN_TO_SUMMARIZE:
        target_end = len(logs) - KEEP_RECENT
        to_summarize = logs[_C.summarized_until_idx:target_end]
        try:
            summary = _gemma26_summarize(to_summarize)
            _C.summarize_call_count += 1
            _C.summarized_prefix = (
                (_C.summarized_prefix + "\n\n" + summary).strip()
                if _C.summarized_prefix else summary
            )
            _C.summarized_until_idx = target_end
        except Exception as e:
            print(f"  ⚠️ Summarization failed: {e} — falling back to filter behavior")

    # Build output
    pm = [m for m in logs if _is_pm_decision(m)]
    out = []
    if pm:
        out.append("=== PM 확정 결정사항 (전체 보존) ===")
        for m in pm[-8:]:
            out.append(f"[{m.agent_name}]: {m.message}")
        out.append("=" * 50)
    if _C.summarized_prefix:
        out.append(f"=== 이전 토론 요약 (LLM 압축, summarize calls={_C.summarize_call_count}) ===")
        out.append(_C.summarized_prefix)
        out.append("=" * 50)
    out.append(f"--- 최근 발언 raw (last {len(logs) - _C.summarized_until_idx}) ---")
    for m in logs[_C.summarized_until_idx:]:
        out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)

MODES = {
    'off':    compactor_off,
    'filter': compactor_filter,
    'claude': compactor_claude,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sample capture wrapper — saves outputs by debate_log size
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _wrap_capture(mode_name, fn):
    samples_dir = '/home/piai/ai_course/agent_test/eval_results/compaction_experiment/samples'
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
                f.write(f"# Mode: {mode_name}  |  debate_log size: {log_len}  |  output chars: {len(result)}\n")
                f.write(f"{'='*80}\n")
                f.write(result)
                f.write("\n")
        return result
    return wrapped

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(f"Usage: {sys.argv[0]} {{off|filter|claude}} [n_runs=3]")
        sys.exit(1)
    mode = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    sub_agents._format_debate_history = _wrap_capture(mode, MODES[mode])
    print(f"🔧 Monkey-patched _format_debate_history with mode='{mode}' (sample capture ON)")
    os.environ['RUNNER_ID'] = f'compact_{mode}'

    from eval.experiment_runner import run_all_experiments
    run_all_experiments(
        backend='qwen-api',
        runs_per_condition=n_runs,
        conditions=['C3_3rounds'],
        harness_settings=None,
        cross_judge=False,
    )
