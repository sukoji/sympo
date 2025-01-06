"""
Judge parse 실패 outlier 재평가 — 직접 Gemini 호출로 max_output_tokens 상향.
eval/llm_judge.py는 max_output_tokens=500 고정이라 일부 응답이 잘려서 parse failed 발생.
이 스크립트는 코드 수정 없이 동일 프롬프트를 더 큰 토큰 버짓으로 다시 호출.
"""
from __future__ import annotations
import json
import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# .env 수동 로드
env_path = PROJ_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from eval.llm_judge import (
    STRUCTURE_PROMPT, ASSIGNMENT_PROMPT, DEBATE_PROMPT,
    _format_wbs, _format_team, _format_debate, JUDGE_MODEL_GEMINI,
)
from data_pipeline.member_parser import MemberParser

ROOT = Path(__file__).resolve().parent
MEMBERS_DIR = PROJ_ROOT / "sample_data" / "sample_members"

DIM_WEIGHTS = {"structure": 0.40, "assignment": 0.35, "debate": 0.25}
MAX_TOKENS_JUDGE = 1500  # eval/llm_judge.py 기본 500보다 상향
MAX_RETRY = 3  # 첫 호출 실패 시 토큰을 키워 재시도


def _load_team():
    team = []
    for p in sorted(MEMBERS_DIR.glob("member_*.txt")):
        text = p.read_text(encoding="utf-8")
        name = p.stem.replace("member_", "")
        team.append(MemberParser.from_resume_text(text, name=name))
    return team


def _clamp(v):
    try:
        return round(min(1.0, max(0.0, float(v))), 2)
    except Exception:
        return 0.0


def _parse_response(resp: str) -> Dict[str, Any]:
    """eval/llm_judge.py의 _parse_judge_response와 동일 로직 + 추가 복구"""
    resp = (resp or "").replace("```json", "").replace("```", "").strip()
    # 먼저 완전 JSON 시도
    m = re.search(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', resp, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return {"score": _clamp(data.get("score", 0)),
                    "reason": str(data.get("reason", ""))[:200]}
        except Exception:
            pass
    # truncated JSON 복구
    m = re.search(r'"score"\s*:\s*([\d.]+)', resp)
    if m:
        score = _clamp(m.group(1))
        reason_m = re.search(r'"reason"\s*:\s*"([^"]*)', resp)
        reason = reason_m.group(1)[:200] if reason_m else resp[:100]
        return {"score": score, "reason": reason}
    # 숫자만이라도
    num = re.search(r'(\d+\.\d+)', resp)
    if num:
        return {"score": _clamp(num.group(1)), "reason": resp[:100]}
    return {"score": -1, "reason": f"parse failed even with 1500 tokens. raw: {resp[:150]}"}


def _call_gemini_high_tokens(prompt: str, max_tokens: int = MAX_TOKENS_JUDGE) -> Dict[str, Any]:
    """직접 Gemini 호출 — 라이브러리 기본 토큰 한도 우회. 실패 시 토큰을 키워 재시도."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from agents.llm_config import normalize_content

    last_raw = ""
    token_schedule = [max_tokens, max_tokens * 2, max_tokens * 4]  # 1500, 3000, 6000
    for attempt, tokens in enumerate(token_schedule[:MAX_RETRY], start=1):
        try:
            llm = ChatGoogleGenerativeAI(
                model=JUDGE_MODEL_GEMINI, temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                max_output_tokens=tokens,
            )
            # Judge prompt에 "ONLY JSON" 강제 재주입 (echo 방지)
            strict_suffix = "\n\n반드시 JSON 객체만 출력하세요. 다른 텍스트·마크다운 금지. 예: {\"score\": 0.7, \"reason\": \"A=0.6 B=0.8 C=0.6\"}"
            resp = normalize_content(llm.invoke(prompt + strict_suffix).content)
            last_raw = resp
            parsed = _parse_response(resp)
            if parsed.get("score", -1) >= 0:
                return parsed
        except Exception as e:
            last_raw = f"api error: {type(e).__name__}: {e}"
    return {"score": -1, "reason": f"still failed after {MAX_RETRY} retries. last raw: {last_raw[:200]}"}


def _reconstruct_tasks(snap_tasks):
    from schemas.wbs_schema import WBSTask
    out = []
    for t in snap_tasks:
        try:
            out.append(WBSTask(**t))
        except Exception:
            class _O: pass
            o = _O()
            for k, v in t.items():
                setattr(o, k, v)
            out.append(o)
    return out


def _reconstruct_debate(snap_log):
    from schemas.wbs_schema import DebateMessage
    out = []
    for m in snap_log:
        try:
            out.append(DebateMessage(**m))
        except Exception:
            class _M: pass
            o = _M()
            for k, v in m.items():
                setattr(o, k, v)
            out.append(o)
    return out


def _is_suspicious_reason(reason: str) -> bool:
    """Judge reason이 실제 평가 결과인지 fallback 노이즈인지 판정.
    정상 포맷: `A=X B=X C=X [D=X]; key issues: ...` 혹은 `{"score": ..., "reason": "A=..."}`
    비정상: 입력 텍스트 일부를 그대로 echo한 경우.
    """
    r = (reason or "").strip()
    if not r:
        return True
    # 정상 포맷 검출 (JSON 형태 또는 A=B= 평가 레이블)
    if '"score"' in r:
        return False
    # A=숫자 같은 평가 레이블 하나라도 있으면 정상
    if re.search(r'\b[A-Da-d]\s*=\s*[\d.]+', r):
        return False
    # 그 외 모든 케이스는 의심 (key issues 같은 설명 키워드도 없고 평가 레이블도 없으면 echo로 판단)
    return True


def _needs_rejudge(judge: Dict[str, Any]) -> List[str]:
    bad = []
    for dim in ["structure", "assignment", "debate"]:
        d = judge.get(dim) or {}
        score = d.get("score", -1)
        reason = (d.get("reason") or "").lower()
        # (1) 명시적 파싱 실패
        if (score == 0 or score < 0) and ("parse failed" in reason or "evaluation failed" in reason):
            bad.append(dim); continue
        # (2) 완벽한 1.0인데 reason이 정상 판정 포맷이 아닌 경우 → fallback 정규식 노이즈
        if score == 1.0 and _is_suspicious_reason(d.get("reason") or ""):
            bad.append(dim); continue
        # (3) score 있으나 reason이 입력 echo로 보이는 경우
        if 0 < score < 1.0 and _is_suspicious_reason(d.get("reason") or ""):
            bad.append(dim); continue
    return bad


def rejudge_run(rec_path: Path, snap_path: Path, team) -> bool:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    judge = rec.get("judge") or {}
    bad_dims = _needs_rejudge(judge)
    if not bad_dims:
        return False

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    tasks = _reconstruct_tasks(snap.get("final_wbs") or snap.get("current_wbs_draft") or [])
    debate = _reconstruct_debate(snap.get("debate_log", []))

    print(f"[rejudge] {rec_path.parent.parent.name}/{rec_path.name} — bad={bad_dims}, tasks={len(tasks)}, debate={len(debate)}")

    wbs_text = _format_wbs(tasks)
    team_text = _format_team(team)
    l3_lines = [l for l in wbs_text.split('\n') if '| L3 |' in l]

    updated = dict(judge)
    for dim in bad_dims:
        if dim == "structure":
            prompt = STRUCTURE_PROMPT.format(wbs_text=wbs_text[:3000])
        elif dim == "assignment":
            prompt = ASSIGNMENT_PROMPT.format(
                team_text=team_text[:1000],
                assignment_text='\n'.join(l3_lines[:30]) or '(no L3)',
            )
        else:  # debate
            prompt = DEBATE_PROMPT.format(debate_text=_format_debate(debate)[:3000])

        res = _call_gemini_high_tokens(prompt)
        old_score = updated[dim].get("score", "?")
        if res.get("score", -1) >= 0:
            updated[dim] = res
            print(f"  ✓ {dim}: {old_score} → {res['score']} ({res['reason'][:80]})")
        else:
            print(f"  ✗ {dim}: 여전히 실패 ({res['reason'][:80]})")

    # overall 재계산
    active = {d: DIM_WEIGHTS[d] for d in DIM_WEIGHTS if updated.get(d, {}).get("score", -1) >= 0}
    if active:
        total_w = sum(active.values())
        overall = round(sum((active[d] / total_w) * updated[d]["score"] for d in active), 4)
    else:
        overall = -1
    updated["overall"] = overall
    updated["rejudged"] = True
    updated["rejudged_dims"] = bad_dims

    # 백업 + 저장
    bak = rec_path.with_suffix(".json.bak_rejudge")
    shutil.copy2(rec_path, bak)
    rec["judge"] = updated
    rec_path.write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )
    print(f"  → overall: {overall}")
    return True


def main():
    team = _load_team()
    print(f"team loaded: {len(team)} members")

    fixed = 0
    for backend_dir in sorted(ROOT.glob("backend_*")):
        runs_dir = backend_dir / "runs"
        snap_dir = runs_dir / "snapshots"
        if not runs_dir.exists() or not snap_dir.exists():
            continue
        for rec_path in sorted(runs_dir.glob("*.json")):
            snap_path = snap_dir / rec_path.name
            if not snap_path.exists():
                continue
            try:
                if rejudge_run(rec_path, snap_path, team):
                    fixed += 1
            except Exception as e:
                print(f"[ERROR] {rec_path.name}: {type(e).__name__}: {e}")

    print(f"\n[DONE] {fixed} runs rejudged.")


if __name__ == "__main__":
    main()
