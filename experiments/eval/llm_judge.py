"""
LLM-as-a-Judge 평가 모듈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WBS 생성 결과를 구조/배분/토론 3차원으로 평가합니다.
기본은 scalar rubric judge이며, --judge-method geval 사용 시
G-Eval form-filling + score-token probability weighted scoring을 사용합니다.
experiment_runner에서 매 실행마다 자동 호출됩니다.

단독 실행:
  python eval/llm_judge.py eval_results/wbs_snapshot_*.json
"""
import json
import os
import sys
import re
import math
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

# rev.2: Judge 모델 env 외부화 + 교차심사 지원
# - JUDGE_MODEL_GEMINI: 1차 심사 (생성과 동일 벤더 → self-preference 편향 가능)
# - JUDGE_MODEL_CLAUDE: 교차 심사 (독립 벤더, eval2 §7 스펙)
JUDGE_MODEL_GEMINI = os.getenv("JUDGE_MODEL_GEMINI", "gemini-3.1-pro-preview")
JUDGE_MODEL_CLAUDE = os.getenv("JUDGE_MODEL_CLAUDE", "claude-sonnet-4-6")
JUDGE_MODEL = JUDGE_MODEL_GEMINI  # 하위 호환 (기존 호출부)
JUDGE_METHOD = os.getenv("JUDGE_METHOD", "geval").lower()

GEVAL_STRUCTURE_PROMPT = """Task Introduction:
You will be given one WBS generated for a software project. Your task is to rate the WBS on one metric: Structure.
Please read the WBS carefully and assign a score using the evaluation criteria.

Evaluation Criteria:
Structure (1-5) - the overall quality of the WBS decomposition and estimates.
1 = very poor, empty, single-level, or mostly unusable
2 = poor, major hierarchy, estimate, or task-quality problems
3 = fair, usable but clearly uneven or incomplete
4 = good, mostly complete with minor issues
5 = excellent, complete, realistic, concrete, and well-structured

Evaluation Steps:
1. Check hierarchy completeness: whether L1/L2/L3 decomposition is complete and balanced.
2. Check estimation realism: whether L3 work packages are realistically sized and buffers are present.
3. Check task quality: whether titles are unique, concrete, and domain-specific.
4. Assign a score for Structure on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

Input Target:
WBS:
{wbs_text}

Evaluation Form (scores ONLY):
Final Answer (one digit from 1 to 5, no words, no punctuation):"""

GEVAL_ASSIGNMENT_PROMPT = """Task Introduction:
You will be given a project team profile and task assignments from a generated WBS. Your task is to rate the assignments on one metric: Assignment Quality.
Please read the team profile and assignments carefully and assign a score using the evaluation criteria.

Input Context:
Team profiles:
{team_text}

Evaluation Criteria:
Assignment Quality (1-5) - the quality of task-to-person matching and workload distribution.
1 = very poor, random assignment or all work assigned to one person
2 = poor, many mismatches or severe imbalance
3 = fair, some fit but notable mismatches or imbalance
4 = good, mostly well matched and balanced
5 = excellent, strong skill fit, balanced workload, complete coverage

Evaluation Steps:
1. Compare each assigned task's implied requirements with the assigned members' skills.
2. Check workload balance using estimated days and whether severe concentration exists.
3. Check team coverage and whether idle members are justified by task needs.
4. Assign a score for Assignment Quality on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

Input Target:
WBS Assignments:
{assignment_text}

Evaluation Form (scores ONLY):
Final Answer (one digit from 1 to 5, no words, no punctuation):"""

GEVAL_DEBATE_PROMPT = """Task Introduction:
You will be given a multi-agent debate log produced while refining a WBS. Your task is to rate the debate on one metric: Debate Quality.
Please read the debate log carefully and assign a score using the evaluation criteria.

Evaluation Criteria:
Debate Quality (1-5) - the quality of multi-agent discussion, role behavior, and convergence.
1 = very poor, no meaningful debate
2 = poor, shallow or dominated debate
3 = fair, some useful discussion but limited convergence
4 = good, substantive multi-role discussion with minor gaps
5 = excellent, concrete, role-consistent, convergent debate

Evaluation Steps:
1. Check participation: whether multiple distinct roles contribute substantively.
2. Check substance: whether messages include concrete task, risk, buffer, or assignment analysis rather than empty agreement.
3. Check role consistency: whether agents remain in their assigned persona.
4. Check convergence: whether disagreements or risks are resolved into clearer decisions.
5. Assign a score for Debate Quality on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

Input Target:
Debate log:
{debate_text}

Evaluation Form (scores ONLY):
Final Answer (one digit from 1 to 5, no words, no punctuation):"""

STRUCTURE_PROMPT = """You are a PMP-certified PM evaluating WBS structure on a CONTINUOUS 0.00-1.00 scale.

Evaluate THREE sub-aspects, score each 0.00-1.00 independently, then average them.
The anchor values below are reference points, NOT the only allowed outputs.
You MUST interpolate between anchors and return a score with exactly TWO decimal places.
Examples of valid sub-scores/final scores: 0.71, 0.78, 0.83, 0.91.

A) HIERARCHY_COMPLETENESS
- 1.0: 5+ L1 phases, every L1 has 3+ L2, every L2 has 3+ L3
- 0.8: 5+ L1, most L2 have 3+ L3; 1-2 L2 have only 2 L3
- 0.6: 4-5 L1, L2 coverage uneven (multiple L2 with 1-2 L3)
- 0.4: 3-4 L1, many L2 missing L3 or only 1-2 L3 each
- 0.2: 1-2 L1 or most L2 lack L3 children
- 0.0: Empty or single-level list

B) ESTIMATION_REALISM
- 1.0: All L3 days 1-10, buffer 15-30% present throughout
- 0.8: ~90% L3 in range, buffer 10-15%
- 0.6: Some L3 outside 1-10 OR buffer <10%
- 0.4: Many L3 outside range OR no buffer
- 0.2: Most days unrealistic
- 0.0: Estimates missing entirely

C) TASK_QUALITY
- 1.0: All titles unique, concrete, domain-specific
- 0.8: 1-2 similar titles or slightly generic
- 0.6: 3-5 templated/vague titles
- 0.4: Many generic or repeating pattern titles
- 0.2: Most tasks are stubs/templates
- 0.0: Uninterpretable titles

Compute final_score = round((A + B + C) / 3, 2). DO NOT cap or snap to buckets — use the continuous average.

Return ONLY single-line JSON, NO markdown, NO preamble, NO listing of input tasks.
Reason MUST be under 120 chars and must include the sub-scores with two decimals.
Format: {{"score": 0.XX, "reason": "A=0.XX B=0.XX C=0.XX; <120-char summary>"}}

WBS:
{wbs_text}"""

ASSIGNMENT_PROMPT = """You are an HR expert evaluating task-to-person assignment quality on a CONTINUOUS 0.00-1.00 scale.

Team profiles:
{team_text}

Evaluate THREE sub-aspects, score each 0.00-1.00, then average.
The anchor values below are reference points, NOT the only allowed outputs.
You MUST interpolate between anchors and return a score with exactly TWO decimal places.
Examples of valid sub-scores/final scores: 0.58, 0.67, 0.74, 0.86.

A) SKILL_FIT — does tech_stack / strengths match assigned tasks?
- 1.0: >90% assignments well-matched to member skills
- 0.8: ~75% matched
- 0.6: ~55% matched (some mismatches)
- 0.4: ~35% matched (many mismatches)
- 0.2: Mostly random or all same role
- 0.0: All tasks to one person or skill/task unrelated

B) WORKLOAD_BALANCE — are L3 days distributed fairly?
- 1.0: max/min workload within 2x, no idle members
- 0.8: max/min within 3x
- 0.6: max/min 3-4x, slight imbalance
- 0.4: max/min 4-6x or 1 idle member
- 0.2: One person 50%+ work OR 2+ idle members
- 0.0: All to one person

C) COVERAGE — are all team members utilized?
- 1.0: Every member has ≥1 assignment matching capacity
- 0.8: 1 member idle (if justifiable)
- 0.6: 2 idle members
- 0.4: 3+ idle
- 0.2: Most team idle
- 0.0: Only 1 member used

Compute final_score = round((A + B + C) / 3, 2). Use continuous average.

Return ONLY single-line JSON, NO markdown, NO preamble, NO listing of input tasks.
Reason MUST be under 120 chars and must include the sub-scores with two decimals.
Format: {{"score": 0.XX, "reason": "A=0.XX B=0.XX C=0.XX; <120-char summary>"}}

WBS Assignments:
{assignment_text}"""

DEBATE_PROMPT = """You are evaluating multi-agent debate quality on a CONTINUOUS 0.00-1.00 scale.

Evaluate FOUR sub-aspects, score each 0.00-1.00, then average.
The anchor values below are reference points, NOT the only allowed outputs.
You MUST interpolate between anchors and return a score with exactly TWO decimal places.
Examples of valid sub-scores/final scores: 0.62, 0.69, 0.77, 0.88.

A) PARTICIPATION
- 1.0: 3+ distinct roles contribute substantively each round
- 0.8: 3 roles but uneven contribution
- 0.6: 2 active + 1 minimal
- 0.4: 2 roles only
- 0.2: 1 role dominant, others silent
- 0.0: No debate / system-only messages

B) SUBSTANCE — concrete, task-ID-referencing analysis vs empty agreement
- 1.0: Most messages cite task IDs and provide concrete analysis
- 0.8: Majority concrete, few generic
- 0.6: Mix of concrete and generic
- 0.4: Mostly generic agreements
- 0.2: Empty 'agree'/'good' no new insight
- 0.0: No analytical content

C) ROLE_CONSISTENCY — agents stay in persona
- 1.0: All agents in persona throughout
- 0.8: 1 minor role slip
- 0.6: 2-3 role slips
- 0.4: Multiple role confusions
- 0.2: Agents impersonate others frequently
- 0.0: Total role breakdown

D) CONVERGENCE — disagreements resolved with risk/buffer discussion
- 1.0: Clear convergence with explicit buffer/risk discussion
- 0.8: Convergence with minor open issues
- 0.6: Partial convergence
- 0.4: No clear convergence but discussion present
- 0.2: Unresolved conflict or topic drift
- 0.0: No buffer/risk discussion at all

Compute final_score = round((A + B + C + D) / 4, 2). Use continuous average.

Return ONLY single-line JSON, NO markdown, NO preamble, NO listing of input tasks.
Reason MUST be under 120 chars and must include the sub-scores with two decimals.
Format: {{"score": 0.XX, "reason": "A=0.XX B=0.XX C=0.XX D=0.XX; <120-char summary>"}}

Debate log (last 20 messages):
{debate_text}"""


_ECHO_MARKERS = ("MBR-", "L3-", "L2-", "L1-", "Total: ", "Backend Developer tasks", "tech_stack")

def _parse_judge_response(resp: str) -> dict:
    """Judge 응답 문자열을 {score, reason}으로 파싱. 점수는 [0.0, 1.0]으로 clamp.
    rev.3 (2026-04-23): 프롬프트-에코 실패 감지 추가 — 응답이 입력 단편을 그대로 토해낸 경우
    `score=-1` (N/A)로 표시하여 평균에서 제외 가능하게 함.
    """
    def _clamp(v: float) -> float:
        return round(min(1.0, max(0.0, float(v))), 2)

    resp = resp.replace("```json", "").replace("```", "").strip()

    # 1) 완전한 JSON 시도.
    # Judge가 입력 일부를 에코하거나 앞뒤 설명을 붙이면 첫 {...}가 정답 JSON이 아닐 수
    # 있으므로, 문자열 안의 모든 JSON object 후보를 순서대로 디코딩하고 score 키가
    # 있는 object를 우선 사용한다.
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", resp):
        try:
            data, _ = decoder.raw_decode(resp[m.start():])
            if not isinstance(data, dict) or "score" not in data:
                continue
            score = _clamp(data.get("score"))
            reason = str(data.get("reason", ""))
            return {"score": score, "reason": reason[:400]}
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    # 2) Truncated JSON — closing `}`가 잘렸어도 `{"score": X.XX, "reason": "...` 패턴이면
    #    score만 추출. (max_tokens cap에 걸려 reason이 mid-string에서 끊기는 경우 다수)
    truncated = re.match(r'\s*\{\s*"score"\s*:\s*(-?\d+\.?\d*)', resp)
    if truncated:
        score = _clamp(truncated.group(1))
        # reason도 가능한 만큼 추출
        reason_m = re.search(r'"reason"\s*:\s*"([^"]*)', resp)
        reason = reason_m.group(1) if reason_m else resp[:200]
        return {"score": score, "reason": f"[truncated] {reason[:400]}"}

    # 3) JSON은 아니지만 최종 점수를 명시한 경우만 fallback으로 복구한다.
    # `A=0.50` 같은 sub-score를 final score로 오인하지 않도록 final/overall/score
    # 문맥이 있는 패턴만 허용한다.
    final_patterns = [
        r"(?:final_score|final score|overall_score|overall score)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
        r"\bscore\s+(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
    ]
    for pat in final_patterns:
        m = re.search(pat, resp, re.IGNORECASE)
        if m:
            return {"score": _clamp(m.group(1)), "reason": f"[regex_recovered] {resp[:300]}"}

    # 4) JSON 파싱 실패 — echo 감지 (프롬프트 단편이 응답에 그대로 들어옴)
    if any(marker in resp for marker in _ECHO_MARKERS):
        return {"score": -1.0, "reason": f"judge_echo_failure: {resp[:200]}"}

    # 5) 응답이 짧고 숫자 단독에 가까우면 fallback 허용 (e.g., "0.5"만 출력한 경우)
    if len(resp) <= 20 and re.match(r'^\s*\d+\.?\d*\s*$', resp):
        num = re.search(r'(\d+\.?\d*)', resp)
        return {"score": _clamp(num.group(1)), "reason": resp[:200]}
    return {"score": -1.0, "reason": f"parse_failed_no_json: {resp[:200]}"}


def _call_gemini(prompt: str, model: str) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from agents.llm_config import normalize_content
    try:
        llm = ChatGoogleGenerativeAI(
            model=model, temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=2500,
            retries=2,
            request_timeout=60,
        )
        resp = normalize_content(llm.invoke(prompt).content)
        parsed = _parse_judge_response(resp)
        if parsed.get("score", -1) >= 0:
            return parsed

        # Gemini occasionally ignores the JSON-only instruction and emits partial
        # reasoning. Retry once with a compact repair instruction instead of
        # turning an otherwise evaluable sample into N/A.
        repair_prompt = (
            "Your previous response was invalid for the required parser.\n"
            "Return ONLY one minified JSON object in this exact schema:\n"
            "{\"score\": 0.00, \"reason\": \"A=0.00 B=0.00 C=0.00; short reason\"}\n"
            "No markdown. No preamble. No task listing. Score must be 0.00 to 1.00.\n\n"
            "Previous invalid response:\n"
            f"{resp[:1200]}\n\n"
            "Original evaluation task:\n"
            f"{prompt[:3500]}"
        )
        repair_llm = ChatGoogleGenerativeAI(
            model=model, temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=600,
            retries=1,
            request_timeout=60,
        )
        repair_resp = normalize_content(repair_llm.invoke(repair_prompt).content)
        repaired = _parse_judge_response(repair_resp)
        if repaired.get("score", -1) >= 0:
            repaired["reason"] = f"[repair_retry] {repaired.get('reason', '')}"[:400]
            return repaired
        return parsed
    except Exception as e:
        # -1 = N/A (quota 초과·네트워크 등 실제 평가 실패. 0점 "총체적 실패"와 구분)
        err = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"  [Judge/Gemini] error: {err}")
        return {"score": -1, "reason": f"evaluation failed — {err}"}


def _call_claude(prompt: str, model: str) -> dict:
    from agents.llm_config import normalize_content
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return {"score": -1, "reason": "langchain-anthropic not installed"}
    try:
        llm = ChatAnthropic(
            model=model, temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2500,
        )
        resp = normalize_content(llm.invoke(prompt).content)
        return _parse_judge_response(resp)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"  [Judge/Claude] error: {err}")
        return {"score": -1, "reason": f"evaluation failed — {err}"}


def _call_judge(prompt: str, model: str = None) -> dict:
    """단일 Judge 호출. model 이름에 따라 Gemini/Claude 자동 분기."""
    model = model or JUDGE_MODEL
    m = model.lower()
    if m.startswith("claude"):
        return _call_claude(prompt, model)
    return _call_gemini(prompt, model)


def _score_token_from_text(text: str) -> int | None:
    m = re.search(r"[1-5]", str(text or "").strip())
    return int(m.group()) if m else None


def _geval_from_probs(score_logprobs: dict[int, float]) -> dict:
    """G-Eval probability-weighted score over the discrete score set {1..5}."""
    if not score_logprobs:
        return {"score": -1.0, "reason": "geval_no_score_token_logprobs"}
    max_lp = max(score_logprobs.values())
    weights = {k: math.exp(v - max_lp) for k, v in score_logprobs.items()}
    denom = sum(weights.values())
    if denom <= 0:
        return {"score": -1.0, "reason": "geval_bad_logprob_mass"}
    probs = {k: weights.get(k, 0.0) / denom for k in range(1, 6)}
    expected_1_5 = sum(k * probs[k] for k in range(1, 6))
    score_0_1 = (expected_1_5 - 1.0) / 4.0
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return {
        "score": round(max(0.0, min(1.0, score_0_1)), 4),
        "reason": f"G-Eval expected={expected_1_5:.2f}/5 entropy={entropy:.2f}",
        "geval_expected_1_5": round(expected_1_5, 4),
        "geval_probs": {str(k): round(probs[k], 6) for k in range(1, 6)},
        "geval_entropy": round(entropy, 4),
        "geval_logprobs_available": True,
    }


def _call_gemini_geval(prompt: str, model: str) -> dict:
    """G-Eval style Gemini call using score-token logprobs.

    The model is forced to emit one score token. We then compute
    score = sum_i p(i) * i over i in {1..5}, normalized over visible score tokens.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"score": -1, "reason": "GOOGLE_API_KEY missing", "judge_method": "geval"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1,
            "responseLogprobs": True,
            "logprobs": 20,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        return {"score": -1, "reason": f"geval_http_{e.code}: {body}", "judge_method": "geval"}
    except Exception as e:
        return {"score": -1, "reason": f"geval_error: {type(e).__name__}: {str(e)[:160]}", "judge_method": "geval"}

    cand = (data.get("candidates") or [{}])[0]
    text_parts = cand.get("content", {}).get("parts", [])
    output_text = "".join(p.get("text", "") for p in text_parts)
    logres = cand.get("logprobsResult") or cand.get("logprobs_result") or {}
    top_steps = logres.get("topCandidates") or logres.get("top_candidates") or []

    score_lps: dict[int, float] = {}
    if top_steps:
        for item in top_steps[0].get("candidates", []):
            digit = _score_token_from_text(item.get("token", ""))
            if digit is None:
                continue
            lp = item.get("logProbability", item.get("log_probability"))
            if lp is not None:
                score_lps[digit] = max(float(lp), score_lps.get(digit, float("-inf")))

    result = _geval_from_probs(score_lps)
    result["judge_method"] = "geval"
    result["geval_raw_output"] = output_text.strip()
    if result.get("score", -1) < 0:
        digit = _score_token_from_text(output_text)
        if digit is not None:
            result.update({
                "score": round((digit - 1.0) / 4.0, 4),
                "reason": f"G-Eval fallback chosen={digit}/5; logprobs unavailable",
                "geval_expected_1_5": float(digit),
                "geval_probs": {str(k): (1.0 if k == digit else 0.0) for k in range(1, 6)},
                "geval_entropy": 0.0,
                "geval_logprobs_available": False,
            })
    return result


def _openai_compat_endpoint(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if raw.endswith("/v1/chat/completions"):
        return raw
    if raw.endswith("/v1"):
        return f"{raw}/chat/completions"
    return f"{raw}/v1/chat/completions"


def _call_openai_compat_geval(prompt: str, backend: str) -> dict:
    """G-Eval style call through an OpenAI-compatible chat completions API.

    This covers OpenAI itself and local/vLLM/llama.cpp style endpoints that expose
    `logprobs` and `top_logprobs` on `/v1/chat/completions`.
    """
    if backend == "openai":
        endpoint = "https://api.openai.com/v1/chat/completions"
        model = os.getenv("GEVAL_API_MODEL") or os.getenv("OPENAI_JUDGE_MODEL") or "gpt-4o-mini"
        api_key = os.getenv("GEVAL_API_KEY") or os.getenv("OPENAI_API_KEY")
    elif backend in ("qwen", "qwen-api", "qwen_api"):
        endpoint = _openai_compat_endpoint(os.getenv("GEVAL_API_URL") or os.getenv("QWEN_API_URL"))
        model = os.getenv("GEVAL_API_MODEL") or os.getenv("QWEN_API_MODEL")
        api_key = os.getenv("GEVAL_API_KEY") or os.getenv("QWEN_API_KEY")
    else:
        endpoint = _openai_compat_endpoint(os.getenv("GEVAL_API_URL"))
        model = os.getenv("GEVAL_API_MODEL")
        api_key = os.getenv("GEVAL_API_KEY")

    if not endpoint or not model:
        return {"score": -1, "reason": f"geval_{backend}_endpoint_or_model_missing", "judge_method": "geval"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }
    if backend in ("qwen", "qwen-api", "qwen_api"):
        payload["chat_template_kwargs"] = {
            "enable_thinking": os.getenv("QWEN_ENABLE_THINKING", "false").lower() in ("true", "1", "yes", "on")
        }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        return {"score": -1, "reason": f"geval_{backend}_http_{e.code}: {body}", "judge_method": "geval"}
    except Exception as e:
        return {"score": -1, "reason": f"geval_{backend}_error: {type(e).__name__}: {str(e)[:160]}", "judge_method": "geval"}

    choice = (data.get("choices") or [{}])[0]
    output_text = choice.get("message", {}).get("content") or choice.get("text") or ""
    log_content = ((choice.get("logprobs") or {}).get("content") or [])
    score_lps: dict[int, float] = {}
    if log_content:
        for item in log_content[0].get("top_logprobs", []) or []:
            digit = _score_token_from_text(item.get("token", ""))
            if digit is None:
                continue
            lp = item.get("logprob")
            if lp is not None:
                score_lps[digit] = max(float(lp), score_lps.get(digit, float("-inf")))

    result = _geval_from_probs(score_lps)
    result["judge_method"] = "geval"
    result["geval_backend"] = backend
    result["geval_model"] = model
    result["geval_raw_output"] = str(output_text).strip()
    if result.get("score", -1) < 0:
        digit = _score_token_from_text(output_text)
        if digit is not None:
            result.update({
                "score": round((digit - 1.0) / 4.0, 4),
                "reason": f"G-Eval fallback chosen={digit}/5; logprobs unavailable",
                "geval_expected_1_5": float(digit),
                "geval_probs": {str(k): (1.0 if k == digit else 0.0) for k in range(1, 6)},
                "geval_entropy": 0.0,
                "geval_logprobs_available": False,
            })
    return result


def _call_geval_judge(prompt: str, model: str) -> dict:
    backend = (os.getenv("GEVAL_JUDGE_BACKEND") or "").strip().lower()
    if not backend:
        if os.getenv("OPENAI_API_KEY"):
            backend = "openai"
        elif os.getenv("QWEN_API_URL"):
            backend = "qwen-api"
        else:
            backend = "gemini"
    if backend == "gemini":
        return _call_gemini_geval(prompt, model)
    return _call_openai_compat_geval(prompt, backend)


def _member_get(member, key, default=None):
    if isinstance(member, dict):
        return member.get(key, default)
    return getattr(member, key, default)


def _assignment_name_maps(team_members=None, debate_log=None) -> tuple[dict, dict]:
    """Build member/task assignment display maps for judge input.

    Older snapshots may contain model-generated MBR-* assignment codes that do
    not match actual team member IDs. Task Manager messages usually preserve
    the human-readable assignment lines, so recover those names before judging.
    """
    id_to_name = {}
    for m in team_members or []:
        mid = _member_get(m, "member_id", "")
        name = _member_get(m, "name", "")
        if mid and name:
            id_to_name[str(mid)] = str(name)

    task_to_names = {}
    for msg in debate_log or []:
        body = msg.get("message", "") if isinstance(msg, dict) else getattr(msg, "message", "")
        if not body:
            continue
        for tid, raw_names in re.findall(r"\[(L3-[^\]]+)\][^\n]*?→\s*([^\n]+)", body):
            names = [
                n.strip()
                for n in re.split(r"[,/·]", raw_names)
                if n.strip() and not n.strip().startswith("[")
            ]
            if names:
                task_to_names[tid] = names
    return id_to_name, task_to_names


def _format_wbs(tasks, team_members=None, debate_log=None) -> str:
    """rev.4 (2026-04-23): 핵심 필드만 (id, title, level, days, role, to). 70-task WBS 기준 ~50% 압축.
    Judge 프롬프트가 길수록 echo·truncation 빈도 ↑ → 평가 신뢰도 ↓ 문제 완화."""
    lines = []
    id_to_name, task_to_names = _assignment_name_maps(team_members, debate_log)
    for t in tasks:
        get = lambda k, default=None: t.get(k, default) if isinstance(t, dict) else getattr(t, k, default)
        lv = get('level', '?')
        if hasattr(lv, 'value'): lv = lv.value
        tid = get('task_id', '')
        title = get('title', '')
        est = get('estimated_days', 0)
        buf = get('buffer_days', 0)
        role = get('assigned_role', '')
        asgn = get('assigned_to', [])
        if isinstance(asgn, list):
            assignees = [str(a) for a in asgn]
        elif asgn:
            assignees = [str(asgn)]
        else:
            assignees = []
        resolved = [id_to_name.get(a, a) for a in assignees]
        if assignees and resolved == assignees and tid in task_to_names:
            resolved = task_to_names[tid]
        asgn_s = ','.join(resolved)
        # 압축 포맷: [id] title (lv, Xd+Yd) role→members
        meta = f"({lv},{est}d"
        if buf: meta += f"+{buf}b"
        meta += ")"
        suffix = f" {role}" if role else ""
        if asgn_s: suffix += f"→{asgn_s}"
        lines.append(f"[{tid}] {title} {meta}{suffix}")
    return '\n'.join(lines) or '(empty WBS)'


def _format_team(members) -> str:
    lines = []
    for m in members:
        name = _member_get(m, 'name', str(m))
        tech = (_member_get(m, 'tech_stack', []) or [])[:4]
        strengths = (_member_get(m, 'strengths', []) or [])[:2]
        yoe = _member_get(m, 'years_of_experience', '?')
        mid = _member_get(m, 'member_id', '?')
        lines.append(f"- {name}({mid}): {yoe}yr, tech:{tech}, strengths:{strengths}")
    return '\n'.join(lines) or '(no team)'


def _format_debate(debate_log) -> str:
    lines = []
    for m in (debate_log[-20:] if len(debate_log) > 20 else debate_log):
        if isinstance(m, dict):
            name, role, msg = m.get('agent_name', '?'), m.get('agent_role', '?'), m.get('message', '')
        else:
            name = getattr(m, 'agent_name', '?')
            role = getattr(m, 'agent_role', '?')
            if hasattr(role, 'value'): role = role.value
            msg = getattr(m, 'message', '')
        lines.append(f"[{name}/{role}]: {msg[:200]}")
    return '\n'.join(lines) or '(no debate)'


def evaluate_wbs(wbs_tasks, team_members=None, debate_log=None, eval_dims=None,
                 judge_model: str = None, cross_judge: bool = False,
                 judge_method: str = None) -> dict:
    """
    3차원 LLM Judge 평가.
    eval_dims: 평가할 차원 목록 (["structure", "assignment", "debate"])
               N/A 차원은 건너뛰고, overall은 해당 차원만의 가중 평균.
    judge_model: 사용할 judge 모델 이름. None이면 JUDGE_MODEL_GEMINI.
    cross_judge: True이면 1차(judge_model) + 2차(상대 벤더) 두 번 평가하여
                 result["cross"]에 저장 (eval2 §7 자기 선호 편향 통제).
    judge_method: "scalar"(기존 JSON 점수) 또는 "geval"(G-Eval 확률가중 점수).
    """
    judge_model = judge_model or JUDGE_MODEL_GEMINI
    judge_method = (judge_method or JUDGE_METHOD or "scalar").lower()

    if eval_dims is None:
        eval_dims = ["structure", "assignment", "debate"]

    # 차원별 가중치 (AlphaEval 방식)
    DIM_WEIGHTS = {"structure": 0.40, "assignment": 0.35, "debate": 0.25}

    primary = _evaluate_with_judge(
        wbs_tasks, team_members, debate_log, eval_dims,
        judge_model=judge_model, dim_weights=DIM_WEIGHTS,
        judge_method=judge_method,
    )

    if cross_judge:
        # 상대 벤더 선택 (간단 규칙)
        alt_model = JUDGE_MODEL_CLAUDE if judge_model.lower().startswith("gemini") else JUDGE_MODEL_GEMINI
        print(f"  [Cross Judge] 2차 심사 시작: {alt_model}")
        secondary = _evaluate_with_judge(
            wbs_tasks, team_members, debate_log, eval_dims,
            judge_model=alt_model, dim_weights=DIM_WEIGHTS,
            judge_method="scalar" if judge_method == "geval" else judge_method,
        )
        primary["cross"] = secondary
        primary["cross_agreement"] = _cross_agreement(primary, secondary)

    return primary


def _evaluate_with_judge(wbs_tasks, team_members, debate_log, eval_dims,
                         judge_model: str, dim_weights: dict,
                         judge_method: str = "scalar") -> dict:
    """단일 judge 모델로 3차원 평가. evaluate_wbs 내부에서 사용."""
    print(f"  [LLM Judge/{judge_model}/{judge_method}] 평가 차원: {eval_dims}")
    wbs_text = _format_wbs(wbs_tasks, team_members=team_members, debate_log=debate_log)
    scores = {}
    use_geval = judge_method == "geval" and not judge_model.lower().startswith("claude")

    if "structure" in eval_dims:
        print("    → Structure...")
        if use_geval:
            scores["structure"] = _call_geval_judge(GEVAL_STRUCTURE_PROMPT.format(wbs_text=wbs_text[:3000]), model=judge_model)
        else:
            scores["structure"] = _call_judge(STRUCTURE_PROMPT.format(wbs_text=wbs_text[:3000]), model=judge_model)
    else:
        scores["structure"] = {"score": -1, "reason": "N/A (이 조건에서 미평가)"}

    if "assignment" in eval_dims:
        print("    → Assignment...")
        team_text = _format_team(team_members or [])
        l3_lines = [l for l in wbs_text.split('\n') if '(L3,' in l]
        if use_geval:
            scores["assignment"] = _call_geval_judge(GEVAL_ASSIGNMENT_PROMPT.format(
                team_text=team_text[:1000],
                assignment_text='\n'.join(l3_lines[:30]) or '(no L3)',
            ), model=judge_model)
        else:
            scores["assignment"] = _call_judge(ASSIGNMENT_PROMPT.format(
                team_text=team_text[:1000],
                assignment_text='\n'.join(l3_lines[:30]) or '(no L3)',
            ), model=judge_model)
    else:
        scores["assignment"] = {"score": -1, "reason": "N/A (배정 단계 미포함)"}

    if "debate" in eval_dims:
        if not debate_log or len(debate_log) < 3:
            scores["debate"] = {"score": 0.0, "reason": "토론 메시지 3개 미만"}
        else:
            print("    → Debate...")
            if use_geval:
                scores["debate"] = _call_geval_judge(GEVAL_DEBATE_PROMPT.format(debate_text=_format_debate(debate_log)[:3000]), model=judge_model)
            else:
                scores["debate"] = _call_judge(DEBATE_PROMPT.format(debate_text=_format_debate(debate_log)[:3000]), model=judge_model)
    else:
        scores["debate"] = {"score": -1, "reason": "N/A (토론 단계 미포함)"}

    active_weights = {d: dim_weights[d] for d in eval_dims if scores.get(d, {}).get("score", -1) >= 0}
    if active_weights:
        total_w = sum(active_weights.values())
        overall = round(sum(
            (active_weights[d] / total_w) * scores[d]["score"]
            for d in active_weights
        ), 4)
    else:
        # 모든 차원이 실패(-1)했으면 overall도 N/A로 표시
        overall = -1

    # overall 실패 시 가장 대표적인 에러 사유를 detail로 전달 (UI 표시용)
    overall_detail = None
    if overall < 0:
        first_err = next(
            (scores[d]["reason"] for d in ["structure", "assignment", "debate"]
             if scores.get(d, {}).get("score", 0) < 0),
            "모든 차원 평가 실패"
        )
        overall_detail = first_err

    effective_judge_model = judge_model
    judge_provider = "gemini" if judge_model.lower().startswith("gemini") else "other"
    if use_geval:
        for d in ("structure", "assignment", "debate"):
            if scores.get(d, {}).get("geval_model"):
                effective_judge_model = scores[d]["geval_model"]
                judge_provider = scores[d].get("geval_backend", "geval")
                break

    result = {
        "structure": scores["structure"],
        "assignment": scores["assignment"],
        "debate": scores["debate"],
        "overall": overall,
        "overall_detail": overall_detail,
        "eval_dims": eval_dims,
        "judge_model": effective_judge_model,
        "judge_provider": judge_provider,
        "judge_method": "geval" if use_geval else "scalar",
    }
    parts = [f"{d[0].upper()}={'N/A' if scores[d]['score'] < 0 else scores[d]['score']}"
             for d in ["structure", "assignment", "debate"]]
    print(f"    ✅ {' '.join(parts)} → Overall={overall}")
    return result


def _cross_agreement(a: dict, b: dict) -> dict:
    """1차·2차 judge 점수의 단순 일치도. ICC/κ는 eval/reliability.py 참조."""
    diffs = []
    for d in ["structure", "assignment", "debate"]:
        sa = a.get(d, {}).get("score", -1)
        sb = b.get(d, {}).get("score", -1)
        if sa < 0 or sb < 0:
            continue
        diffs.append(abs(sa - sb))
    mean_abs_diff = round(sum(diffs) / len(diffs), 4) if diffs else -1
    return {
        "mean_abs_diff": mean_abs_diff,
        "overall_diff": round(abs(a.get("overall", 0) - b.get("overall", 0)), 4),
        "n_dimensions": len(diffs),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="WBS 스냅샷 재심사 (단독 Judge)")
    parser.add_argument("snapshot", help="wbs_snapshot_*.json 경로")
    parser.add_argument("--judge", default="gemini",
                        choices=["gemini", "claude", "cross"],
                        help="gemini/claude 단독 또는 cross(교차)")
    parser.add_argument("--judge-method", default=JUDGE_METHOD,
                        choices=["scalar", "geval"],
                        help="scalar=기존 JSON 점수, geval=G-Eval 확률가중 점수(logprobs 지원 judge)")
    args = parser.parse_args()

    data = json.load(open(args.snapshot, encoding='utf-8'))
    tasks = data.get('wbs_tasks', [])
    dl = data.get('debate_log', [])

    if args.judge == "cross":
        r = evaluate_wbs(tasks, [], dl, judge_model=JUDGE_MODEL_GEMINI, cross_judge=True,
                         judge_method=args.judge_method)
    elif args.judge == "claude":
        r = evaluate_wbs(tasks, [], dl, judge_model=JUDGE_MODEL_CLAUDE,
                         judge_method=args.judge_method)
    else:
        r = evaluate_wbs(tasks, [], dl, judge_model=JUDGE_MODEL_GEMINI,
                         judge_method=args.judge_method)
    print(json.dumps(r, ensure_ascii=False, indent=2))
