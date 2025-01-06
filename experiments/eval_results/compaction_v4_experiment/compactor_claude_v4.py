"""C_claude_v4 — fixed trigger logic.

이전 v3 버그:
  1. est_tokens = chars × 1.2/3 = chars × 0.4 (영어 기준 underestimate)
     → Korean Gemma tokenizer 실측치 chars × 0.6~0.8 수준
  2. TOKEN_THRESHOLD=4000 + 위 공식 → 실제 chars≥10000 필요
     → 130 msg 토론에서 log≈44 시점에 첫 trigger (이미 1/3 진행 후)
     → 사실상 거의 raw 상태로 진행

Fix:
  - tokens/char ratio: 0.4 → 0.7 (실제 한글 토크나이저)
  - threshold: 4000 → 1500 tokens (≈ 2150 chars)
  - 결과: log≈10-15에서 첫 trigger 예상

타 동작은 v3와 동일 (REPLACE bounded summary, KEEP_RECENT=6, PM_RETAIN=4).
"""
import os, sys
sys.path.insert(0, '/home/piai/ai_course/agent_test/eval_results/compaction_experiment')
from agents import sub_agents
from schemas.wbs_schema import DebateMessage
from run_compaction_v2 import _is_pm_decision, _format_msg, _CompactState

TOKEN_THRESHOLD = 1500
TOKEN_PER_CHAR = 0.7
KEEP_RECENT = 6
MIN_TO_SUMMARIZE = 3
MAX_SUMMARY_CHARS = 1500
PM_RETAIN = 4

_C = _CompactState()

def _gemma_replace_summarize(prev_summary: str, new_messages: list) -> str:
    parts = []
    if prev_summary:
        parts.append(f"[이전 요약]\n{prev_summary}")
    if new_messages:
        formatted, total = [], 0
        for m in new_messages:
            txt = m.message if hasattr(m, 'message') else m.get('message','')
            name = m.agent_name if hasattr(m, 'agent_name') else m.get('agent_name','')
            txt = txt[:500] if len(txt) > 500 else txt
            line = f"[{name}]: {txt}"
            if total + len(line) > 5000:
                formatted.append(f"... (이후 {len(new_messages)-len(formatted)}개 truncated)")
                break
            formatted.append(line); total += len(line)
        parts.append("[새 발언들]\n" + "\n".join(formatted))
    combined = "\n\n".join(parts)
    prompt = f"""다음 multi-agent 토론 정보를 {MAX_SUMMARY_CHARS}자 이내 단일 요약으로 압축하라.

**보존**: 리스크 제안, 버퍼 일수, task ID, 합의·이견
**생략**: 인사·단순 동의·반복

원본:
{combined}

요약 (반드시 {MAX_SUMMARY_CHARS}자 이내):"""
    from agents.llm_config import get_llm
    llm = get_llm(temperature=0.3, max_tokens=600)
    resp = llm.invoke(prompt)
    content = resp.content if hasattr(resp, 'content') else str(resp)
    if "QwenAPI 호출 실패" in content or "HTTPError" in content or "Bad Request" in content:
        raise RuntimeError(f"Summarization failed: {content[:120]}")
    summary = content.strip()
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS] + "...(truncated)"
    return summary

def compactor_claude_v4(state, max_messages=8):
    """Fixed trigger threshold + token estimate."""
    global _C
    logs = state.get("debate_log", [])
    if not logs:
        _C = _CompactState()
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    if len(logs) < _C.last_seen_log_size:
        _C = _CompactState()
    _C.last_seen_log_size = len(logs)

    new_logs = logs[_C.summarized_until_idx:]
    chars = sum(len(m.message if hasattr(m,'message') else m.get('message','')) for m in new_logs)
    est_tokens = int(chars * TOKEN_PER_CHAR)

    available = len(logs) - _C.summarized_until_idx - KEEP_RECENT
    if est_tokens > TOKEN_THRESHOLD and available >= MIN_TO_SUMMARIZE:
        target_end = len(logs) - KEEP_RECENT
        new_old_msgs = logs[_C.summarized_until_idx:target_end]
        try:
            new_summary = _gemma_replace_summarize(_C.summarized_prefix, new_old_msgs)
            _C.summarize_call_count += 1
            _C.summarized_prefix = new_summary
            _C.summarized_until_idx = target_end
        except Exception as e:
            print(f"  ⚠️ Summarization failed: {e}", flush=True)

    pm = [m for m in logs if _is_pm_decision(m)]
    out = []
    if pm:
        out.append("=== PM 확정 결정사항 ===")
        for m in pm[-PM_RETAIN:]:
            out.append(f"[{m.agent_name}]: {m.message}")
        out.append("=" * 50)
    if _C.summarized_prefix:
        out.append(f"=== 누적 요약 (LLM, ≤{MAX_SUMMARY_CHARS}자, REPLACE: calls={_C.summarize_call_count}) ===")
        out.append(_C.summarized_prefix)
        out.append("=" * 50)
    out.append(f"--- 최근 발언 raw (last {len(logs) - _C.summarized_until_idx}) ---")
    for m in logs[_C.summarized_until_idx:]:
        out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)
