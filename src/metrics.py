"""
WBS 오케스트레이션 평가 지표 자동 계산 모듈
────────────────────────────────────────────
WBS 생성 완료 후 final_state를 입력받아 7개 정량 지표를 산출하고
generated/metrics_report.json 으로 저장합니다.
매 실행마다 generated/metrics_history.csv 에 누적 이력을 기록합니다.
"""
import csv
import json
import os
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

from eval_results.autoscore_recompute import recompute_autoscore

_EMBEDDING_MODEL = None
_EMBEDDING_LOAD_FAILED = False


def _load_sentence_embedding_model():
    """Load the shared sentence embedding model, or return None in offline envs."""
    global _EMBEDDING_MODEL, _EMBEDDING_LOAD_FAILED
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    if _EMBEDDING_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBEDDING_MODEL
    except Exception as exc:
        _EMBEDDING_LOAD_FAILED = True
        print(f"[WARN] SentenceTransformer unavailable; using metric fallbacks: {exc}")
        return None


def _cosine_similarity(vec_a, vec_b) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    return dot / (norm_a * norm_b + 1e-9)


# ──────────────────────────────────────────────────────────
# 기준값 (Benchmark / Threshold)
# ──────────────────────────────────────────────────────────
METRIC_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "ragas_faithfulness": {
        "label": "RAGAS Faithfulness",
        "unit": "",
        "format": ".2%",
        "threshold_pass": 0.5,
        "threshold_warn": 0.3,
        "direction": "higher",          # 높을수록 좋음
        "description": "RAG 컨텍스트 근거 비율 (≥50% 권장)",
    },
    "success_rate": {
        "label": "Success Rate (SR)",
        "unit": "",
        "format": ".2%",
        "threshold_pass": 1.0,
        "threshold_warn": 0.8,
        "direction": "higher",
        "description": "PRD 기능 커버리지 (100% 필수)",
    },
    "planning_score": {
        "label": "Planning Score",
        "unit": "",
        "format": ".4f",
        "threshold_pass": 0.5,
        "threshold_warn": 0.35,
        "direction": "higher",
        "description": "태스크-팀원 역량 코사인 유사도 (≥0.5 권장)",
    },
    "buffer_ratio_pct": {
        "label": "Buffer Ratio",
        "unit": "%",
        "format": ".1f",
        "threshold_pass_min": 15.0,
        "threshold_pass_max": 30.0,
        "threshold_warn_min": 10.0,
        "threshold_warn_max": 40.0,
        "direction": "range",           # 15~30% 적정 범위
        "description": "리스크 버퍼 비율 (15%~30% 권장)",
    },
    "interaction_turns": {
        "label": "Interaction Turns",
        "unit": "회",
        "format": "d",
        "threshold_pass_min": 5,
        "threshold_pass_max": 60,
        "threshold_warn_min": 3,
        "threshold_warn_max": 80,
        "direction": "range",
        "description": "총 토론 메시지 수 (5~60회 적정)",
    },
    "supervisor_intervention_ratio": {
        "label": "Supervisor 개입율",
        "unit": "",
        "format": ".2%",
        "threshold_pass": 0.40,
        "threshold_warn": 0.55,
        "direction": "lower",           # 낮을수록 좋음
        "description": "슈퍼바이저 메시지 비율 (≤40% 권장)",
    },
    "convergence": {
        "label": "Convergence",
        "unit": "",
        "format": "bool",
        "direction": "bool",            # True가 좋음
        "description": "토론 수렴 여부 (수렴이 정상)",
    },
}


def _judge(key: str, value: Any) -> str:
    """기준값 대비 판정: '✅ 통과' / '⚠️ 경고' / '❌ 미달'"""
    bench = METRIC_BENCHMARKS.get(key)
    if bench is None:
        return "—"

    direction = bench["direction"]

    if direction == "bool":
        return "✅ 통과" if value else "❌ 미달"

    if direction == "higher":
        if value >= bench["threshold_pass"]:
            return "✅ 통과"
        elif value >= bench["threshold_warn"]:
            return "⚠️ 경고"
        return "❌ 미달"

    if direction == "lower":
        if value <= bench["threshold_pass"]:
            return "✅ 통과"
        elif value <= bench["threshold_warn"]:
            return "⚠️ 경고"
        return "❌ 미달"

    if direction == "range":
        p_min = bench.get("threshold_pass_min", 0)
        p_max = bench.get("threshold_pass_max", float("inf"))
        w_min = bench.get("threshold_warn_min", 0)
        w_max = bench.get("threshold_warn_max", float("inf"))
        if p_min <= value <= p_max:
            return "✅ 통과"
        elif w_min <= value <= w_max:
            return "⚠️ 경고"
        return "❌ 미달"

    return "—"


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    중첩 metrics 딕셔너리를 CSV 1행용 평탄 딕셔너리로 변환합니다.
    """
    cfg = metrics.get("experiment_config", {})
    row: Dict[str, Any] = {
        "timestamp":     metrics.get("timestamp", ""),
        "project_name":  metrics.get("project_name", ""),
        "model_backend": metrics.get("model_backend", ""),
        "total_tasks":   metrics.get("total_tasks", 0),
        "debate_rounds": metrics.get("debate_rounds", 0),
        # ── 실험 설정 ──
        "rag_strategy":  cfg.get("rag_strategy", ""),
        "min_rounds":    cfg.get("min_rounds", ""),
        "max_rounds":    cfg.get("max_rounds", ""),
        "team_size":     cfg.get("team_size", ""),
        "budget_weeks":  cfg.get("budget_weeks", ""),
        "note":          cfg.get("note", ""),
        "condition":     cfg.get("condition", cfg.get("condition_key", "")),
        # ── 1. RAGAS Faithfulness ──
        "ragas_faithfulness":    metrics.get("ragas_faithfulness", {}).get("faithfulness", 0.0),
        "ragas_supported_claims": metrics.get("ragas_faithfulness", {}).get("supported_claims", 0),
        "ragas_total_claims":    metrics.get("ragas_faithfulness", {}).get("total_claims", 0),
        # ── 2. Interaction Turns ──
        "interaction_turns":     metrics.get("interaction_turns", {}).get("total_messages", 0),
        "interaction_unique_agents": metrics.get("interaction_turns", {}).get("unique_agents", 0),
        # ── 3. Supervisor Intervention ──
        "supervisor_intervention_ratio": metrics.get("supervisor_intervention", {}).get("intervention_ratio", 0.0),
        "supervisor_messages":   metrics.get("supervisor_intervention", {}).get("supervisor_messages", 0),
        "supervisor_mediation_decisions": metrics.get("supervisor_intervention", {}).get("mediation_decisions", 0),
        # ── 4. Success Rate ──
        "success_rate":          metrics.get("success_rate", {}).get("success_rate", 0.0),
        "success_covered":       metrics.get("success_rate", {}).get("covered", 0),
        "success_total_features": metrics.get("success_rate", {}).get("total_features", 0),
        # ── 5. Planning Score ──
        "planning_score":        metrics.get("planning_score", {}).get("planning_score", 0.0),
        "planning_num_assignments": metrics.get("planning_score", {}).get("num_assignments_evaluated", 0),
        # ── 6. Buffer Ratio ──
        "buffer_ratio_pct":      metrics.get("buffer_ratio", {}).get("buffer_ratio_pct", 0.0),
        "buffer_total_estimated_days": metrics.get("buffer_ratio", {}).get("total_estimated_days", 0),
        "buffer_total_buffer_days": metrics.get("buffer_ratio", {}).get("total_buffer_days", 0),
        "buffer_l1_task_count":  metrics.get("buffer_ratio", {}).get("l1_task_count", 0),
        # ── 7. Convergence ──
        "convergence_is_converging": metrics.get("convergence", {}).get("is_converging", False),
        "convergence_trend":     metrics.get("convergence", {}).get("convergence_trend", 0.0),
        "convergence_proposal_count": metrics.get("convergence", {}).get("proposal_count", 0),
        # ── 8~13. 신규 지표 ──
        "mece_score":            metrics.get("mece_score", {}).get("mece_score", 0.0),
        "mece_overlap_pairs":    metrics.get("mece_score", {}).get("overlap_pairs", 0),
        "mece_gap_l1s":          metrics.get("mece_score", {}).get("gap_l1s", 0),
        "granularity_fitness":   metrics.get("granularity_fitness", {}).get("granularity_fitness", 0.0),
        "granularity_too_small": metrics.get("granularity_fitness", {}).get("too_small", 0),
        "granularity_too_large": metrics.get("granularity_fitness", {}).get("too_large", 0),
        "workload_gini":         metrics.get("workload_gini", {}).get("gini", 0.0),
        "workload_max_load":     metrics.get("workload_gini", {}).get("max_load", 0),
        "workload_min_load":     metrics.get("workload_gini", {}).get("min_load", 0),
        "schedule_feasibility":  metrics.get("schedule_feasibility", {}).get("feasibility", 0.0),
        "schedule_conflicts":    metrics.get("schedule_feasibility", {}).get("conflicts", 0),
        "comm_efficiency":       metrics.get("communication_efficiency", {}).get("efficiency", 0.0),
        "comm_effective_msgs":   metrics.get("communication_efficiency", {}).get("effective_messages", 0),
        "harness_caught_exceptions": metrics.get("harness_observability", {}).get("harness_caught_exceptions", 0),
        "role_drift_detected_count": metrics.get("harness_observability", {}).get("role_drift_detected_count", 0),
        "cumulative_reassignments": metrics.get("cumulative_reassignments", 0),
        "cumulative_buffers_applied": metrics.get("cumulative_buffers_applied", 0),
        "cumulative_new_tasks":  metrics.get("cumulative_new_tasks", 0),
        "est_total_tokens":      metrics.get("token_cost", {}).get("est_total_tokens", 0),
        "est_cost_usd":          metrics.get("token_cost", {}).get("est_cost_usd", 0.0),
        # ── 14. AutoScore (종합 점수) ──
        "autoscore_final":       metrics.get("autoscore", {}).get("autoscore", 0.0),
        "autoscore_quality":     metrics.get("autoscore", {}).get("quality", 0.0),
        "autoscore_allocation":  metrics.get("autoscore", {}).get("allocation", 0.0),
        "autoscore_orchestration": metrics.get("autoscore", {}).get("orchestration", 0.0),
        "autoscore_na_cats":     ",".join(metrics.get("autoscore", {}).get("na_categories", []) or []),
        # ── 판정 결과 ──
        "judge_ragas_faithfulness":         _judge("ragas_faithfulness", metrics.get("ragas_faithfulness", {}).get("faithfulness", 0.0)),
        "judge_success_rate":               _judge("success_rate", metrics.get("success_rate", {}).get("success_rate", 0.0)),
        "judge_planning_score":             _judge("planning_score", metrics.get("planning_score", {}).get("planning_score", 0.0)),
        "judge_buffer_ratio_pct":           _judge("buffer_ratio_pct", metrics.get("buffer_ratio", {}).get("buffer_ratio_pct", 0.0)),
        "judge_interaction_turns":          _judge("interaction_turns", metrics.get("interaction_turns", {}).get("total_messages", 0)),
        "judge_supervisor_intervention_ratio": _judge("supervisor_intervention_ratio", metrics.get("supervisor_intervention", {}).get("intervention_ratio", 0.0)),
        "judge_convergence":                _judge("convergence", metrics.get("convergence", {}).get("is_converging", False)),
    }
    return row


def save_metrics_history(metrics: Dict[str, Any], output_dir: str) -> str:
    """
    매 실행마다 metrics_history.csv 에 1행을 추가합니다.
    파일이 없으면 헤더도 함께 작성합니다.
    기존 파일에 새 컬럼이 추가된 경우, 헤더를 갱신하고 기존 행을 보존합니다.
    Returns: CSV 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics_history.csv")

    row = flatten_metrics(metrics)
    fieldnames = list(row.keys())

    if os.path.isfile(csv_path):
        # 기존 헤더 읽기
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []

        # 새 컬럼이 추가된 경우 → 파일 전체를 새 헤더로 다시 쓰기
        if set(fieldnames) - set(existing_fields):
            merged_fields = list(existing_fields)
            for fn in fieldnames:
                if fn not in merged_fields:
                    # debate_rounds 다음에 실험 설정 컬럼 삽입
                    idx = merged_fields.index("debate_rounds") + 1 if "debate_rounds" in merged_fields else len(merged_fields)
                    merged_fields.insert(idx, fn)
                    idx += 1

            # 기존 데이터 읽기
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

            # 새 헤더로 전체 다시 쓰기
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow(old_row)
                writer.writerow(row)
        else:
            # 헤더 변경 없음 → 단순 append
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=existing_fields, extrasaction="ignore")
                writer.writerow(row)
    else:
        # 새 파일 생성
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    return csv_path


# ──────────────────────────────────────────────────────────
# 1. RAGAS Faithfulness (NLI 기반 — Es et al., 2023)
# ──────────────────────────────────────────────────────────
_nli_model = None

def _get_nli_model():
    """NLI Cross-Encoder 싱글톤 로드"""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", max_length=512)
        except Exception as e:
            print(f"[WARN] NLI 모델 로드 실패: {e}")
            _nli_model = "unavailable"
    return _nli_model if _nli_model != "unavailable" else None


def _chunk_context(contexts: List[str], max_len: int = 400) -> List[str]:
    """RAG 컨텍스트를 NLI 모델 입력에 적합한 크기로 청크 분할"""
    chunks = []
    for ctx in contexts:
        # 문장 단위 분할 (마침표/줄바꿈)
        sentences = [s.strip() for s in ctx.replace("\n", ". ").split(". ") if len(s.strip()) > 10]
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > max_len:
                if current:
                    chunks.append(current.strip())
                current = sent
            else:
                current = current + ". " + sent if current else sent
        if current:
            chunks.append(current.strip())
    return chunks if chunks else [""]


def calc_ragas_faithfulness(
    tasks: list,
    rag_contexts: List[str],
) -> Dict[str, Any]:
    """
    RAGAS Faithfulness (Es et al., 2023) — NLI 모델 기반 구현.

    알고리즘:
      1. WBS 태스크에서 주장(claim) 추출 (title + description)
      2. RAG 컨텍스트를 청크로 분할
      3. 각 claim에 대해 모든 context chunk와 NLI 추론
         → entailment score가 임계값 이상이면 "근거 있음"
      4. Faithfulness = 근거 있는 claim 수 / 전체 claim 수

    NLI 모델: cross-encoder/nli-MiniLM2-L6-H768
    출력: [contradiction, entailment, neutral] logits
    """
    if not tasks or not rag_contexts:
        return {"faithfulness": 0.0, "supported_claims": 0, "total_claims": 0,
                "method": "nli", "detail": "컨텍스트 없음"}

    nli = _get_nli_model()
    if nli is None:
        # NLI 모델 불가 시 키워드 fallback
        return _calc_faithfulness_keyword_fallback(tasks, rag_contexts)

    # 컨텍스트 청크 준비
    context_chunks = _chunk_context(rag_contexts, max_len=400)

    # 주장(claim) 추출
    claims = []
    for t in tasks:
        title = getattr(t, "title", "")
        desc = getattr(t, "description", "")
        buf_rationale = getattr(t, "buffer_rationale", "")
        if title and len(title) > 3:
            claims.append({"task_id": t.task_id, "text": title, "source": "title"})
        if desc and len(desc) > 10:
            claims.append({"task_id": t.task_id, "text": desc, "source": "description"})
        if buf_rationale and len(buf_rationale) > 5:
            claims.append({"task_id": t.task_id, "text": buf_rationale, "source": "buffer_rationale"})

    if not claims:
        return {"faithfulness": 0.0, "supported_claims": 0, "total_claims": 0,
                "method": "nli", "detail": "추출된 claim 없음"}

    # NLI 추론: 각 claim × 각 context chunk
    ENTAILMENT_THRESHOLD = 0.5  # entailment logit 임계값
    supported = 0
    details = []

    # 배치 구성 (효율성)
    pairs = []
    pair_index = []  # (claim_idx, chunk_idx)
    for ci, claim in enumerate(claims):
        for cj, chunk in enumerate(context_chunks):
            # premise=context, hypothesis=claim (RAGAS 방식)
            pairs.append((chunk, claim["text"]))
            pair_index.append((ci, cj))

    # 배치 추론 (메모리 관리를 위해 256개씩)
    BATCH = 256
    all_scores = []
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i:i + BATCH]
        scores = nli.predict(batch)  # shape: (batch, 3)
        all_scores.extend(scores)

    # claim별 최대 entailment score 집계
    claim_max_entailment = [0.0] * len(claims)
    claim_best_chunk = [None] * len(claims)
    for idx, (ci, cj) in enumerate(pair_index):
        entailment_score = float(all_scores[idx][1])  # index 1 = entailment
        if entailment_score > claim_max_entailment[ci]:
            claim_max_entailment[ci] = entailment_score
            claim_best_chunk[ci] = cj

    for ci, claim in enumerate(claims):
        is_supported = claim_max_entailment[ci] >= ENTAILMENT_THRESHOLD
        if is_supported:
            supported += 1
        details.append({
            "task_id": claim["task_id"],
            "claim": claim["text"][:60],
            "source": claim["source"],
            "entailment_score": round(claim_max_entailment[ci], 4),
            "supported": is_supported,
        })

    faithfulness = round(supported / len(claims), 4)
    return {
        "faithfulness": faithfulness,
        "supported_claims": supported,
        "total_claims": len(claims),
        "method": "nli",
        "model": "cross-encoder/nli-MiniLM2-L6-H768",
        "threshold": ENTAILMENT_THRESHOLD,
        "context_chunks": len(context_chunks),
        "top_unsupported": sorted(
            [d for d in details if not d["supported"]],
            key=lambda x: x["entailment_score"]
        )[:5],
    }


def _calc_faithfulness_keyword_fallback(
    tasks: list,
    rag_contexts: List[str],
) -> Dict[str, Any]:
    """NLI 모델 불가 시 키워드 기반 fallback (명시적으로 method='keyword'로 표시)"""
    context_blob = " ".join(rag_contexts).lower()
    supported = 0
    total = 0

    for t in tasks:
        claims = []
        if getattr(t, "title", ""):
            claims.append(t.title)
        if getattr(t, "description", "") and len(t.description) > 5:
            claims.append(t.description)

        for claim in claims:
            total += 1
            keywords = [w for w in claim.lower().replace(",", " ").split() if len(w) >= 3]
            if not keywords:
                continue
            matched = sum(1 for kw in keywords if kw in context_blob)
            if matched / len(keywords) >= 0.3:
                supported += 1

    faithfulness = round(supported / max(total, 1), 4)
    return {
        "faithfulness": faithfulness,
        "supported_claims": supported,
        "total_claims": total,
        "method": "keyword_fallback",
    }


# ──────────────────────────────────────────────────────────
# 2. Interaction Turns  (토론 턴 수)
# ──────────────────────────────────────────────────────────
def calc_interaction_turns(debate_log: list) -> Dict[str, Any]:
    """토론 로그의 총 메시지 수 및 에이전트별 발화 수."""
    total = len(debate_log)
    by_agent = Counter()
    for m in debate_log:
        role = getattr(m.agent_role, "value", str(m.agent_role))
        by_agent[role] += 1
    return {
        "total_messages": total,
        "unique_agents": len(by_agent),
        "messages_by_agent": dict(by_agent),
    }


# ──────────────────────────────────────────────────────────
# 3. Supervisor 개입 빈도
# ──────────────────────────────────────────────────────────
def calc_supervisor_intervention(debate_log: list) -> Dict[str, Any]:
    """Supervisor(PM) 역할의 메시지 비율 계산."""
    total = len(debate_log)
    if total == 0:
        return {"intervention_ratio": 0.0, "supervisor_messages": 0, "total_messages": 0}

    sup_msgs = sum(
        1 for m in debate_log
        if "슈퍼바이저" in str(getattr(m.agent_role, "value", ""))
        or "PM" in str(getattr(m.agent_role, "value", ""))
    )
    mediation_msgs = sum(
        1 for m in debate_log
        if getattr(m, "message_type", "") in ("mediation", "decision")
    )
    return {
        "intervention_ratio": round(sup_msgs / total, 4),
        "supervisor_messages": sup_msgs,
        "mediation_decisions": mediation_msgs,
        "total_messages": total,
    }


# ──────────────────────────────────────────────────────────
# 4. Success Rate (SR) — PRD 기능 커버리지
# ──────────────────────────────────────────────────────────
def calc_success_rate(key_features: List[str], tasks: list,
                      threshold: float = 0.45) -> Dict[str, Any]:
    """
    PRD의 key_features 각각이 WBS 태스크에 의미적으로 커버되는지 판정.
    임베딩 코사인 유사도 기반 — 각 feature와 가장 유사한 태스크의 최대 유사도가
    임계값(기본 0.45) 이상이면 커버됨으로 판정.
    threshold 파라미터로 감도 분석(τ grid) 가능.
    """
    if not key_features:
        return {"success_rate": 0.0, "covered": 0, "total_features": 0, "uncovered": [],
                "method": "embedding"}

    model = _load_sentence_embedding_model()
    if model is None:
        return _calc_sr_keyword_fallback(key_features, tasks)

    # 태스크 텍스트 임베딩
    task_texts = [f"{t.title} {t.description}" for t in tasks]
    if not task_texts:
        return {"success_rate": 0.0, "covered": 0, "total_features": len(key_features),
                "uncovered": key_features, "method": "embedding"}

    try:
        feat_embs = model.encode(key_features)
        task_embs = model.encode(task_texts)
    except Exception as exc:
        print(f"[WARN] Success Rate embedding failed; using keyword fallback: {exc}")
        return _calc_sr_keyword_fallback(key_features, tasks)

    THRESHOLD = threshold
    covered = 0
    uncovered = []
    details = []

    for fi, feat in enumerate(key_features):
        # 이 feature와 모든 태스크 간 코사인 유사도 계산
        sims = []
        for ti in range(len(task_texts)):
            sim = _cosine_similarity(feat_embs[fi], task_embs[ti])
            sims.append((float(sim), tasks[ti].task_id))

        best_sim, best_task = max(sims, key=lambda x: x[0])
        is_covered = best_sim >= THRESHOLD
        if is_covered:
            covered += 1
        else:
            uncovered.append(feat)
        details.append({
            "feature": feat[:50],
            "best_match_task": best_task,
            "similarity": round(best_sim, 4),
            "covered": is_covered,
        })

    sr = round(covered / len(key_features), 4)
    return {
        "success_rate": sr,
        "covered": covered,
        "total_features": len(key_features),
        "uncovered_features": uncovered,
        "method": "embedding",
        "threshold": THRESHOLD,
        "details": details,
    }


def _calc_sr_keyword_fallback(key_features, tasks):
    """임베딩 불가 시 키워드 fallback"""
    import re as _re
    task_blob = " ".join(f"{t.title} {t.description}" for t in tasks).lower()
    covered, uncovered = 0, []
    for feat in key_features:
        # 특수문자로도 분리 (·, /, ·)
        keywords = [w for w in _re.split(r'[\s,·/\-·]+', feat.lower()) if len(w) >= 2]
        if not keywords:
            covered += 1
            continue
        matched = sum(1 for kw in keywords if kw in task_blob)
        if matched / len(keywords) >= 0.4:
            covered += 1
        else:
            uncovered.append(feat)
    return {"success_rate": round(covered / len(key_features), 4), "covered": covered,
            "total_features": len(key_features), "uncovered_features": uncovered, "method": "keyword_fallback"}


# ──────────────────────────────────────────────────────────
# 5. Planning Score (코사인 유사도 — 태스크 ↔ 팀원 적합도)
# ──────────────────────────────────────────────────────────
def calc_planning_score(tasks: list, team_members: list) -> Dict[str, Any]:
    """
    각 태스크의 (title + description) 임베딩과
    배정된 팀원의 (strengths + tech_stack) 임베딩 간의 코사인 유사도 평균.
    sentence-transformers 사용.
    """
    model = _load_sentence_embedding_model()
    if model is None:
        return _calc_planning_keyword_fallback(tasks, team_members)

    member_map = {m.member_id: m for m in team_members}

    # 멤버 임베딩 사전 계산 (성능: O(n) → 1회)
    member_embs = {}
    for m in team_members:
        member_text = (
            " ".join(m.strengths if isinstance(m.strengths, list) else [str(m.strengths)]) + " " +
            " ".join(m.tech_stack) + " " +
            " ".join(m.primary_skills)
        )
        try:
            member_embs[m.member_id] = model.encode(member_text)
        except Exception as exc:
            print(f"[WARN] Planning member embedding failed; using keyword fallback: {exc}")
            return _calc_planning_keyword_fallback(tasks, team_members)

    scores = []
    for t in tasks:
        if not getattr(t, "assigned_to", None):
            continue
        task_text = f"{t.title} {t.description}"
        try:
            emb_task = model.encode(task_text)
        except Exception as exc:
            print(f"[WARN] Planning task embedding failed; using keyword fallback: {exc}")
            return _calc_planning_keyword_fallback(tasks, team_members)
        assigned_ids = t.assigned_to if isinstance(t.assigned_to, list) else [t.assigned_to]

        for mid in assigned_ids:
            if mid not in member_embs:
                continue
            emb_member = member_embs[mid]
            sim = _cosine_similarity(emb_task, emb_member)
            scores.append({"task_id": t.task_id, "member_id": mid, "similarity": round(float(sim), 4)})

    avg_score = round(sum(s["similarity"] for s in scores) / max(len(scores), 1), 4)
    return {
        "planning_score": avg_score,
        "num_assignments_evaluated": len(scores),
        "top_5_lowest": sorted(scores, key=lambda x: x["similarity"])[:5],
    }


def _calc_planning_keyword_fallback(tasks: list, team_members: list) -> Dict[str, Any]:
    """Offline fallback for task-member fit when sentence embeddings are unavailable."""
    import re as _re

    def tokens(text: str) -> set:
        return {w for w in _re.split(r"[\s,./()\[\]{}:;+\-]+", text.lower()) if len(w) >= 2}

    member_map = {m.member_id: m for m in team_members}
    member_tokens = {}
    for m in team_members:
        member_text = (
            " ".join(m.strengths if isinstance(m.strengths, list) else [str(m.strengths)]) + " " +
            " ".join(m.tech_stack) + " " +
            " ".join(m.primary_skills)
        )
        member_tokens[m.member_id] = tokens(member_text)

    scores = []
    for t in tasks:
        if not getattr(t, "assigned_to", None):
            continue
        task_tokens = tokens(f"{t.title} {t.description}")
        assigned_ids = t.assigned_to if isinstance(t.assigned_to, list) else [t.assigned_to]
        for mid in assigned_ids:
            if mid not in member_map:
                continue
            overlap = len(task_tokens & member_tokens.get(mid, set()))
            denom = max(len(task_tokens | member_tokens.get(mid, set())), 1)
            sim = overlap / denom
            scores.append({"task_id": t.task_id, "member_id": mid, "similarity": round(float(sim), 4)})

    avg_score = round(sum(s["similarity"] for s in scores) / max(len(scores), 1), 4)
    return {
        "planning_score": avg_score,
        "num_assignments_evaluated": len(scores),
        "top_5_lowest": sorted(scores, key=lambda x: x["similarity"])[:5],
        "method": "keyword_fallback",
    }


# ──────────────────────────────────────────────────────────
# 6. Buffer Ratio (버퍼 비율)
# ──────────────────────────────────────────────────────────
def calc_buffer_ratio(tasks: list) -> Dict[str, Any]:
    """L1 태스크 기준으로 버퍼 비율 계산."""
    from schemas.wbs_schema import WBSLevel
    l1_tasks = [t for t in tasks if getattr(t, "level", None) == WBSLevel.L1]
    total_est = sum(t.estimated_days for t in l1_tasks)
    total_buf = sum(t.buffer_days for t in l1_tasks)
    ratio = round(total_buf / max(total_est, 0.001) * 100, 2)
    return {
        "buffer_ratio_pct": ratio,
        "total_estimated_days": total_est,
        "total_buffer_days": total_buf,
        "l1_task_count": len(l1_tasks),
    }


# ──────────────────────────────────────────────────────────
# 7. Convergence Rate (라운드별 수렴도)
# ──────────────────────────────────────────────────────────
def calc_convergence(debate_log: list) -> Dict[str, Any]:
    """
    에이전트 토론 메시지 중 buffer_days_proposed가 있는 메시지들을
    시간순으로 정렬하여 변화량 추세를 분석.
    변화량이 감소할수록 수렴 → convergence_trend 값이 음수면 수렴.
    """
    import re as _re
    proposals = []
    for m in debate_log:
        # 1순위: 구조화된 buffer_days_proposed 필드
        bp = getattr(m, "buffer_days_proposed", None)
        if bp is not None:
            proposals.append(bp)
            continue
        # 2순위: 메시지 텍스트에서 버퍼 숫자 추출 (자연어 fallback)
        msg = getattr(m, "message", "")
        buf_match = _re.search(r'(?:버퍼|buffer)\s*[:\s]*(\d+(?:\.\d+)?)\s*일', msg, _re.IGNORECASE)
        if buf_match:
            proposals.append(float(buf_match.group(1)))

    if not proposals:
        # 버퍼 제안이 없어도, 토론 메시지 길이의 변화로 수렴도 추정
        msg_lens = [len(getattr(m, "message", "")) for m in debate_log
                    if getattr(m, "message_type", "") not in ("mediation", "decision", "pass")]
        if len(msg_lens) >= 4:
            mid = len(msg_lens) // 2
            first_avg = sum(msg_lens[:mid]) / mid
            second_avg = sum(msg_lens[mid:]) / max(len(msg_lens) - mid, 1)
            trend = round(second_avg - first_avg, 2)
            return {"convergence_trend": trend, "proposal_count": 0, "is_converging": trend <= 0,
                    "method": "message_length", "note": "버퍼 제안 없음 — 메시지 길이 기반 추정"}
        return {"convergence_trend": 0.0, "proposal_count": 0, "is_converging": False,
                "note": "데이터 부족"}

    if len(proposals) < 2:
        # 단일 제안인 경우, 변화가 없으므로 일단 수렴 가능성으로 간주 (또는 데이터 부족)
        return {"convergence_trend": 0.0, "proposal_count": len(proposals), "is_converging": True}

    # 연속 차이(Delta)의 평균 방향 계산
    deltas = [abs(proposals[i] - proposals[i - 1]) for i in range(1, len(proposals))]
    
    # 모든 델타가 0이면 완벽한 수렴
    if all(d == 0 for d in deltas):
        return {"convergence_trend": 0.0, "is_converging": True, "note": "의견 일치 (수렴 완료)"}

    # 전반부 vs 후반부 평균 비교
    mid = len(deltas) // 2
    first_half_avg = sum(deltas[:mid]) / max(mid, 1)
    second_half_avg = sum(deltas[mid:]) / max(len(deltas) - mid, 1)
    trend = round(second_half_avg - first_half_avg, 4)  # 음수이면 수렴 중

    return {
        "convergence_trend": trend,
        "first_half_avg_delta": round(first_half_avg, 2),
        "second_half_avg_delta": round(second_half_avg, 2),
        "proposal_count": len(proposals),
        "is_converging": trend <= 0,
    }


# ──────────────────────────────────────────────────────────
# 8. MECE Score (상호배타·전체포괄 — Minto, 1987 Pyramid Principle)
# ──────────────────────────────────────────────────────────
def calc_mece_score(tasks: list) -> Dict[str, Any]:
    """
    L2 기능그룹 간 중복(Overlap)과 누락(Gap) 분석.
    - Overlap: L2 title 간 코사인 유사도가 0.7 이상인 쌍의 수
    - Gap: L1 하위 L2 개수가 2 미만인 L1의 비율
    - MECE Score = 1 - (overlap_ratio + gap_ratio) / 2
    """
    from schemas.wbs_schema import WBSLevel
    l1_tasks = [t for t in tasks if getattr(t, "level", None) == WBSLevel.L1]
    l2_tasks = [t for t in tasks if getattr(t, "level", None) == WBSLevel.L2]

    if not l2_tasks:
        return {"mece_score": 0.0, "overlap_pairs": 0, "gap_l1s": 0, "total_l1": len(l1_tasks)}

    # ── Overlap 검사: 같은 L1 내 L2 간 임베딩 코사인 유사도 ──
    _emb_model = _load_sentence_embedding_model()
    try:
        if _emb_model is None:
            raise RuntimeError("embedding model unavailable")
        l2_texts = [f"{t.title} {t.description}" for t in l2_tasks]
        l2_embs = _emb_model.encode(l2_texts)
        use_embedding = True
    except Exception as exc:
        print(f"[WARN] MECE embedding failed; using keyword fallback: {exc}")
        use_embedding = False

    overlap_pairs = 0
    same_parent_pairs = 0  # 분모: 같은 L1 내 L2 쌍 수만

    for i in range(len(l2_tasks)):
        for j in range(i + 1, len(l2_tasks)):
            if l2_tasks[i].parent_id != l2_tasks[j].parent_id:
                continue
            same_parent_pairs += 1
            if use_embedding:
                sim = _cosine_similarity(l2_embs[i], l2_embs[j])
                if sim >= 0.75:  # 코사인 유사도 0.75 이상이면 중복
                    overlap_pairs += 1
            else:
                # fallback: Jaccard
                import re as _re
                kw_a = {w for w in _re.split(r'[\s,·/]+', l2_tasks[i].title.lower()) if len(w) >= 2}
                kw_b = {w for w in _re.split(r'[\s,·/]+', l2_tasks[j].title.lower()) if len(w) >= 2}
                if kw_a and kw_b and len(kw_a & kw_b) / len(kw_a | kw_b) >= 0.5:
                    overlap_pairs += 1

    overlap_ratio = overlap_pairs / max(same_parent_pairs, 1)

    # ── Gap 검사: L1 하위 L2 개수 부족 ──
    l2_per_l1 = Counter(t.parent_id for t in l2_tasks)
    gap_l1s = sum(1 for l1 in l1_tasks if l2_per_l1.get(l1.task_id, 0) < 2)
    gap_ratio = gap_l1s / max(len(l1_tasks), 1)

    mece = round(1.0 - (overlap_ratio + gap_ratio) / 2, 4)
    return {
        "mece_score": max(0.0, mece),
        "overlap_pairs": overlap_pairs,
        "gap_l1s": gap_l1s,
        "total_l1": len(l1_tasks),
        "total_l2": len(l2_tasks),
    }


# ──────────────────────────────────────────────────────────
# 9. Granularity Fitness (운영 가이드라인 — PMBOK 7판, 2021)
# ──────────────────────────────────────────────────────────
def calc_granularity_fitness(tasks: list) -> Dict[str, Any]:
    """
    PMBOK 7판(2021) Work Package 권장 크기: 짧은 리포팅 주기 내 추적 가능한 규모.
    "8-80시간"은 6판 관행으로 엄격 규칙이 아니며 본 지표는 운영 가이드라인으로만 사용.
    Fitness = 범위 내 L3 태스크 수 / 전체 L3 태스크 수 (1일~10일)
    """
    from schemas.wbs_schema import WBSLevel
    l3_tasks = [t for t in tasks if getattr(t, "level", None) == WBSLevel.L3]
    if not l3_tasks:
        return {"granularity_fitness": 0.0, "in_range": 0, "total_l3": 0, "too_small": 0, "too_large": 0}

    MIN_DAYS, MAX_DAYS = 1.0, 10.0
    in_range = sum(1 for t in l3_tasks if MIN_DAYS <= t.estimated_days <= MAX_DAYS)
    too_small = sum(1 for t in l3_tasks if t.estimated_days < MIN_DAYS)
    too_large = sum(1 for t in l3_tasks if t.estimated_days > MAX_DAYS)

    fitness = round(in_range / len(l3_tasks), 4)
    return {
        "granularity_fitness": fitness,
        "in_range": in_range,
        "total_l3": len(l3_tasks),
        "too_small": too_small,
        "too_large": too_large,
        "avg_l3_days": round(sum(t.estimated_days for t in l3_tasks) / len(l3_tasks), 2),
    }


# ──────────────────────────────────────────────────────────
# 10. Workload Balance — Gini Coefficient
# ──────────────────────────────────────────────────────────
def calc_workload_gini(tasks: list, team_members: list = None) -> Dict[str, Any]:
    """
    팀원별 총 배정 일수의 Gini 계수.
    0 = 완벽 균등, 1 = 완벽 불균등.
    미배정 팀원도 workload=0으로 포함하여 정확한 불균등도 측정.
    """
    from schemas.wbs_schema import WBSLevel
    l3_tasks = [t for t in tasks if getattr(t, "level", None) == WBSLevel.L3]

    # 전체 팀원을 workload=0으로 초기화 (미배정 포함)
    workloads: Dict[str, float] = {}
    if team_members:
        for m in team_members:
            workloads[m.member_id] = 0.0

    for t in l3_tasks:
        for mid in (t.assigned_to or []):
            workloads[mid] = workloads.get(mid, 0) + t.total_days

    if len(workloads) < 2:
        return {"gini": 0.0, "workloads": workloads, "num_members": len(workloads)}

    vals = sorted(workloads.values())
    n = len(vals)
    cumulative = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(vals))
    gini = round(cumulative / (n * sum(vals)) if sum(vals) > 0 else 0.0, 4)

    return {
        "gini": max(0.0, gini),
        "workloads": workloads,
        "num_members": n,
        "max_load": max(vals),
        "min_load": min(vals),
        "std_dev": round((sum((v - sum(vals)/n)**2 for v in vals) / n) ** 0.5, 2),
    }


# ──────────────────────────────────────────────────────────
# 11. Schedule Feasibility (인원 충돌 검사)
# ──────────────────────────────────────────────────────────
def calc_schedule_feasibility(tasks: list) -> Dict[str, Any]:
    """
    같은 사람에게 배정된 태스크가 시간적으로 겹치는(동시 수행) 건수.
    Feasibility = 1 - (충돌 태스크 쌍 수 / 전체 쌍 수)
    """
    from schemas.wbs_schema import WBSLevel
    l3_tasks = [t for t in tasks if getattr(t, "level", None) == WBSLevel.L3 and t.start_week and t.end_week]

    person_tasks: Dict[str, list] = {}
    for t in l3_tasks:
        for mid in (t.assigned_to or []):
            person_tasks.setdefault(mid, []).append(t)

    conflicts = 0
    conflict_details = []
    total_pairs = 0

    for mid, ptasks in person_tasks.items():
        ptasks.sort(key=lambda t: t.start_week)
        for i in range(len(ptasks)):
            for j in range(i + 1, len(ptasks)):
                total_pairs += 1
                a, b = ptasks[i], ptasks[j]
                # 의존성 관계(a→b)면 연속 배정이므로 충돌 아님
                is_sequential = b.task_id in (getattr(a, 'dependencies', []) or []) or \
                                a.task_id in (getattr(b, 'dependencies', []) or [])
                if is_sequential:
                    continue
                # 엄격 부등호: 주 경계 핸드오프(a.end_week == b.start_week)는 충돌 아님
                if a.end_week > b.start_week:
                    conflicts += 1
                    conflict_details.append({
                        "member": mid,
                        "task_a": a.task_id,
                        "task_b": b.task_id,
                        "overlap_weeks": a.end_week - b.start_week,
                    })

    feasibility = round(1.0 - (conflicts / max(total_pairs, 1)), 4)
    return {
        "feasibility": feasibility,
        "conflicts": conflicts,
        "total_pairs": total_pairs,
        "conflict_details": conflict_details[:10],
    }


# ──────────────────────────────────────────────────────────
# 12. Communication Efficiency (유효 메시지 비율)
# ──────────────────────────────────────────────────────────
def calc_communication_efficiency(debate_log: list) -> Dict[str, Any]:
    """
    실질적 기여 메시지 비율 = (전체 - PASS - 순수동의 - 시스템 메시지) / 전체
    순수동의: '동의' 키워드 포함 + 새 리스크/버퍼 제안 없음
    """
    if not debate_log:
        return {"efficiency": 0.0, "effective": 0, "total": 0}

    total = len(debate_log)
    agree_kw = ["동의", "맞습니다", "적절합니다", "타당합니다", "공감"]
    risk_kw = ["위험", "리스크", "버퍼", "반대", "이의", "추가 필요", "누락"]

    ineffective = 0
    for m in debate_log:
        msg = m.message
        mtype = getattr(m, "message_type", "")
        role = str(getattr(m, "agent_role", ""))
        # PASS 메시지
        if mtype == "pass" or "[PASS]" in msg:
            ineffective += 1
            continue
        # 시스템 메시지
        if "SYSTEM" in role or "시스템" in str(getattr(m, "agent_name", "")):
            ineffective += 1
            continue
        # 순수 동의 (새 리스크 없이 동의만)
        is_agree = any(k in msg for k in agree_kw)
        has_risk = any(k in msg for k in risk_kw) or getattr(m, "buffer_days_proposed", None)
        if is_agree and not has_risk:
            ineffective += 1

    effective = total - ineffective
    efficiency = round(effective / max(total, 1), 4)
    return {
        "efficiency": efficiency,
        "effective_messages": effective,
        "ineffective_messages": ineffective,
        "total_messages": total,
    }


# ──────────────────────────────────────────────────────────
# 13. Token Cost (API 토큰 비용 추적)
# ──────────────────────────────────────────────────────────
def calc_token_cost(debate_log: list, tasks: list) -> Dict[str, Any]:
    """
    토큰 비용 근사치 계산.
    실제 API 토큰 추적이 불가하므로 문자 수 기반 추정 사용.
    한국어 1자 ≈ 1.5 토큰, 영어 1단어 ≈ 1.3 토큰 (근사)
    """
    total_chars = 0
    for m in debate_log:
        total_chars += len(m.message)

    # WBS 태스크 텍스트 길이 (생성 비용 근사)
    wbs_chars = sum(len(t.title) + len(t.description) for t in tasks)

    # 한국어 평균 토큰 추정
    est_debate_tokens = int(total_chars * 1.2)  # 한국어 혼합 근사
    est_wbs_tokens = int(wbs_chars * 1.2)
    est_total = est_debate_tokens + est_wbs_tokens

    # Gemini Flash Lite 가격 근사: $0.075 / 1M tokens
    est_cost_usd = round(est_total * 0.075 / 1_000_000, 6)

    return {
        "est_total_tokens": est_total,
        "est_debate_tokens": est_debate_tokens,
        "est_wbs_tokens": est_wbs_tokens,
        "est_cost_usd": est_cost_usd,
        "debate_messages": len(debate_log),
        "total_chars": total_chars + wbs_chars,
    }


# ──────────────────────────────────────────────────────────
# 14. AutoScore — 카테고리 가중 평균 종합 점수 (eval2 §4)
# ──────────────────────────────────────────────────────────
DEFAULT_AUTOSCORE_WEIGHTS: Dict[str, float] = {
    "quality": 0.40,
    "allocation": 0.35,
    "orchestration": 0.25,
}

# 각 범주 대표 지표 (direction: "higher"=높을수록 좋음, "lower"=1-x 변환)
AUTOSCORE_CATEGORIES: Dict[str, List] = {
    "quality": [
        ("mece_score", "mece_score", "higher"),
        ("success_rate", "success_rate", "higher"),
    ],
    "allocation": [
        ("planning_score", "planning_score", "higher"),
        ("workload_gini", "gini", "lower"),
        ("schedule_feasibility", "feasibility", "higher"),
    ],
    "orchestration": [
        ("communication_efficiency", "efficiency", "higher"),
    ],
}

NA_SENTINEL = -1


def _autoscore_pick(metrics: Dict[str, Any], block: str, field: str) -> Optional[float]:
    """중첩 metrics에서 값 추출 — N/A(-1)/None은 재정규화 제외용으로 None 반환."""
    sub = metrics.get(block)
    if sub is None:
        return None
    v = sub.get(field) if isinstance(sub, dict) else None
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if fv == NA_SENTINEL:
        return None
    return fv


def _category_score(metrics: Dict[str, Any], category: str) -> Optional[float]:
    vals = []
    for block, field, direction in AUTOSCORE_CATEGORIES[category]:
        v = _autoscore_pick(metrics, block, field)
        if v is None:
            continue
        if direction == "lower":
            v = 1.0 - min(max(v, 0.0), 1.0)
        vals.append(min(max(v, 0.0), 1.0))
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_autoscore(metrics: Dict[str, Any],
                      weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    카테고리(품질/배분/오케스트레이션)별 가중 평균으로 AutoScore 산출.
    Canonical 구현은 eval_results.autoscore_recompute.recompute_autoscore().
    런타임 평가와 과거 실험 재산출이 같은 산식을 쓰도록 이 함수는 위임만 한다.
    반환: {"autoscore": float, "quality": float, "allocation": float,
           "orchestration": float, "weights_used": {...}, "na_categories": [...]}
    """
    return recompute_autoscore(metrics, top_level_weights=weights)


# ──────────────────────────────────────────────────────────
# 통합 메트릭 계산 및 저장
# ──────────────────────────────────────────────────────────
def compute_all_metrics(
    final_state: dict,
    prd,
    team_members: list,
    output_dir: str = None,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    모든 지표를 계산하고 JSON으로 저장합니다.
    experiment_config: 실험 설정 (LLM 백엔드, RAG 전략, 라운드 등)이 함께 기록됩니다.
    Returns: 전체 지표 딕셔너리
    """
    tasks = final_state.get("final_wbs") or final_state.get("current_wbs_draft", [])
    debate_log = final_state.get("debate_log", [])
    rag_contexts = (
        final_state.get("rag_reference_wbs", []) +
        final_state.get("rag_meeting_logs", [])
    )
    key_features = getattr(prd, "key_features", [])

    # RAG 미사용 조건 판정: Faithfulness N/A 처리 (-1)
    # 근거 컨텍스트 자체가 없는 조건에 0점 부과는 부당 → AutoScore에서 재정규화로 제외
    _rag_na = not rag_contexts
    if _rag_na:
        faith = {
            "faithfulness": -1,
            "supported_claims": 0,
            "total_claims": 0,
            "method": "na_no_rag_context",
            "detail": "RAG 컨텍스트 없음 → N/A 처리(종합점수 제외)",
        }
    else:
        faith = calc_ragas_faithfulness(tasks, rag_contexts)

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": getattr(prd, "project_name", "Unknown"),
        "model_backend": os.environ.get("LLM_BACKEND", "unknown"),
        "total_tasks": len(tasks),
        "debate_rounds": final_state.get("current_round", 0),

        # 7 핵심 지표 (기존)
        "ragas_faithfulness": faith,
        "interaction_turns": calc_interaction_turns(debate_log),
        "supervisor_intervention": calc_supervisor_intervention(debate_log),
        "success_rate": calc_success_rate(key_features, tasks),
        "planning_score": calc_planning_score(tasks, team_members),
        "buffer_ratio": calc_buffer_ratio(tasks),
        "convergence": calc_convergence(debate_log),

        # 6 신규 지표 (학술적 보강)
        "mece_score": calc_mece_score(tasks),
        "granularity_fitness": calc_granularity_fitness(tasks),
        "workload_gini": calc_workload_gini(tasks, team_members),
        "schedule_feasibility": calc_schedule_feasibility(tasks),
        "communication_efficiency": calc_communication_efficiency(debate_log),
        "token_cost": calc_token_cost(debate_log, tasks),
        "cumulative_reassignments": int(final_state.get("cumulative_reassignments", 0) or 0),
        "cumulative_buffers_applied": int(final_state.get("cumulative_buffers_applied", 0) or 0),
        "cumulative_new_tasks": int(final_state.get("cumulative_new_tasks", 0) or 0),
        "harness_observability": {
            "harness_enabled": bool(final_state.get("harness_enabled", False)),
            "harness_caught_exceptions": sum(
                1 for m in debate_log
                if "하네스 포착 예외" in str(getattr(m, "message", ""))
            ),
            "role_drift_detected_count": int(final_state.get("role_drift_detected_count", 0) or 0),
        },
    }

    # 실험 설정 기록
    if experiment_config:
        metrics["experiment_config"] = experiment_config

    # AutoScore
    metrics["autoscore"] = recompute_autoscore(metrics)

    # JSON 저장 (최신 결과)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    # CSV 이력 누적 저장
    save_metrics_history(metrics, output_dir)

    return metrics
