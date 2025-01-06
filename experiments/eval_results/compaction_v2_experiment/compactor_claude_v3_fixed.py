"""C_claude_v3 — bounded summary REPLACE (not append).

이전 버그:
- summarized_prefix가 매 trigger마다 append됨 (cap 없음)
- 누적 summary가 unbounded growth → output > 12K chars → 16K context 한계 초과 → sub-agent fail

Fix:
- 매 trigger마다 (old summary + new old messages)를 함께 LLM에 보내고 ONE compressed summary로 REPLACE
- Summary는 ≤2000 chars로 명시적 제약
- PM retention도 4개로 줄임 (C_filter와 동등 → unfair advantage 제거)

monkey-patch로 _format_debate_history 교체. 코드 무수정.
"""
import os
from agents import sub_agents
from schemas.wbs_schema import DebateMessage

# 동일 helpers (run_compaction_v2.py에서 import)
import sys
sys.path.insert(0, '/home/piai/ai_course/agent_test/eval_results/compaction_experiment')
from run_compaction_v2 import _is_pm_decision, _format_msg, _CompactState

TOKEN_THRESHOLD = 4000
KEEP_RECENT = 6
MIN_TO_SUMMARIZE = 3
MAX_SUMMARY_CHARS = 2000  # ★ NEW: bounded summary
PM_RETAIN = 4              # ★ FIX: was 8 — match C_filter

_C_FIXED = _CompactState()

def _gemma26_replace_summarize(prev_summary: str, new_messages: list) -> str:
    """OLD summary + new old messages → fresh ≤2000 char summary.
    핵심: REPLACE (이전 summary 통째로 새것으로 대체)."""
    parts = []
    if prev_summary:
        parts.append(f"[이전 요약]\n{prev_summary}")
    if new_messages:
        # truncate each message text + cap total input
        formatted = []
        total = 0
        for m in new_messages:
            msg_text = m.message[:500] if len(m.message) > 500 else m.message
            line = f"[{m.agent_name}]: {msg_text}"
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
    # Hard cap (LLM이 안 지킬 수 있음)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS] + "...(truncated)"
    return summary

def compactor_claude_v3(state, max_messages=8):
    """Bounded summary REPLACE 버전. C_claude의 unbounded append 버그 수정."""
    global _C_FIXED
    logs = state.get("debate_log", [])
    if not logs:
        _C_FIXED = _CompactState()
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."
    if len(logs) < _C_FIXED.last_seen_log_size:
        _C_FIXED = _CompactState()  # new run reset
    _C_FIXED.last_seen_log_size = len(logs)

    # Estimate tokens of unsummarized portion
    new_logs = logs[_C_FIXED.summarized_until_idx:]
    chars = sum(len(m.message) for m in new_logs)
    est_tokens = int(chars * 1.2 / 3)

    available = len(logs) - _C_FIXED.summarized_until_idx - KEEP_RECENT
    if est_tokens > TOKEN_THRESHOLD and available >= MIN_TO_SUMMARIZE:
        target_end = len(logs) - KEEP_RECENT
        new_old_msgs = logs[_C_FIXED.summarized_until_idx:target_end]
        try:
            # ★ REPLACE: combine prev summary + new messages → ONE fresh summary
            new_summary = _gemma26_replace_summarize(_C_FIXED.summarized_prefix, new_old_msgs)
            _C_FIXED.summarize_call_count += 1
            _C_FIXED.summarized_prefix = new_summary
            _C_FIXED.summarized_until_idx = target_end
        except Exception as e:
            print(f"  ⚠️ Summarization failed: {e}")

    pm = [m for m in logs if _is_pm_decision(m)]
    out = []
    if pm:
        out.append("=== PM 확정 결정사항 ===")
        for m in pm[-PM_RETAIN:]:  # ★ FIX: 4개로 줄임 (C_filter와 동등)
            out.append(f"[{m.agent_name}]: {m.message}")
        out.append("=" * 50)
    if _C_FIXED.summarized_prefix:
        out.append(f"=== 누적 요약 (LLM, ≤{MAX_SUMMARY_CHARS}자, REPLACE: calls={_C_FIXED.summarize_call_count}) ===")
        out.append(_C_FIXED.summarized_prefix)
        out.append("=" * 50)
    out.append(f"--- 최근 발언 raw (last {len(logs) - _C_FIXED.summarized_until_idx}) ---")
    for m in logs[_C_FIXED.summarized_until_idx:]:
        out.append(_format_msg(m))
    out.append("--------------------")
    return "\n".join(out)
