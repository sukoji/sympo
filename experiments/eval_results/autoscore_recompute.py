"""Post-hoc autoscore recomputation from archived experiment records.

This module intentionally works with both:
- flat CSV rows (e.g. summary_finegrain.csv)
- nested metadata JSON entries (e.g. experiment_metadata.json)

It recomputes autoscore from the raw metric fields that are already stored.
If richer mediation signals are present, orchestration scoring automatically
uses them; otherwise it falls back to the archived fields only.

Design rule: new scoring revisions must remain backfill-safe. A metric that is
not present in historical CSV/metadata is treated as N/A and reweighted out,
not as zero. This keeps old and new runs comparable under the same formula.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


AUTOSCORE_VERSION = "v2_backfill_safe"

QUALITY_WEIGHTS: Dict[str, float] = {
    "success_rate": 0.35,
    "mece_score": 0.35,
    "granularity_fitness": 0.30,
}

ALLOCATION_WEIGHTS: Dict[str, float] = {
    "planning_score": 0.45,
    "schedule_feasibility": 0.25,
    "buffer_adequacy": 0.15,
    "workload_balance": 0.15,
}

ORCHESTRATION_WEIGHTS: Dict[str, float] = {
    "communication_efficiency": 0.25,
    "convergence": 0.30,
    "revision_yield": 0.25,
    "failure_resilience": 0.20,
}

TOP_LEVEL_WEIGHTS: Dict[str, float] = {
    "quality": 0.45,
    "allocation": 0.35,
    "orchestration": 0.20,
}

NA_SENTINELS = {"", None, -1, -1.0, "NA", "N/A", "nan", "None"}
FAITHFULNESS_GATE_THRESHOLD = 0.60
FAITHFULNESS_GATE_FLOOR = 0.25


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_float(value: Any) -> Optional[float]:
    # Handle nested dict (some metrics fields stored as dict in metadata.json)
    if isinstance(value, dict):
        for key in ("success_rate", "score", "value", "fitness", "ratio",
                    "feasibility", "gini", "convergence_score", "intervention_ratio"):
            if key in value:
                return _to_float(value[key])
        return None
    if isinstance(value, list):
        return None
    if value in NA_SENTINELS:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if fv == -1:
        return None
    return fv


def _to_bool(value: Any) -> Optional[bool]:
    if value in ("True", "true", "1", 1, 1.0, True):
        return True
    if value in ("False", "false", "0", 0, 0.0, False):
        return False
    return None


def _pick(record: Dict[str, Any], *paths: Iterable[str]) -> Any:
    for path in paths:
        cur: Any = record
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def _mean_weighted(parts: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    active = {k: weights[k] for k, v in parts.items() if v is not None}
    if not active:
        return None
    total = sum(active.values())
    return sum((active[k] / total) * parts[k] for k in active)


def _mean_full_scope(parts: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    active_weights = {k: v for k, v in weights.items() if k in parts}
    total = sum(active_weights.values())
    if total <= 0:
        return None
    return sum((active_weights[k] / total) * (parts.get(k) or 0.0) for k in active_weights)


def _active_weights(parts: Dict[str, Optional[float]], weights: Dict[str, float]) -> Dict[str, float]:
    active = {k: weights[k] for k, v in parts.items() if v is not None}
    total = sum(active.values())
    if total <= 0:
        return {}
    return {k: active[k] / total for k in active}


def _condition(record: Dict[str, Any]) -> str:
    return str(_pick(record, ("condition",), ("mode",), ("experiment_config", "condition"), ("experiment_config", "condition_key")) or "")


def _debate_rounds(record: Dict[str, Any]) -> int:
    value = _pick(record, ("debate_rounds",), ("experiment_config", "max_rounds"))
    fv = _to_float(value)
    return int(fv) if fv is not None else 0


def _score_buffer_adequacy(buffer_ratio_pct: Optional[float]) -> Optional[float]:
    if buffer_ratio_pct is None:
        return None
    r = max(0.0, float(buffer_ratio_pct))
    if 15.0 <= r <= 30.0:
        return 1.0
    if 10.0 <= r < 15.0:
        return 0.5 + 0.5 * ((r - 10.0) / 5.0)
    if 30.0 < r <= 40.0:
        return 0.5 + 0.5 * ((40.0 - r) / 10.0)
    if 0.0 <= r < 10.0:
        return 0.5 * (r / 10.0)
    if 40.0 < r <= 60.0:
        return 0.5 * ((60.0 - r) / 20.0)
    return 0.0


def _score_workload_balance(gini: Optional[float]) -> Optional[float]:
    if gini is None:
        return None
    g = max(0.0, float(gini))
    if g <= 0.20:
        return 1.0
    return _clamp01(1.0 - ((g - 0.20) / 0.60))


def _score_convergence(record: Dict[str, Any]) -> Optional[float]:
    is_conv = _to_bool(_pick(record, ("convergence", "is_converging"), ("convergence_is_converging",), ("convergence",)))
    trend = _to_float(_pick(record, ("convergence", "convergence_trend"), ("convergence_trend",)))
    proposal_count = _to_float(_pick(record, ("convergence", "proposal_count"), ("convergence_proposal_count",)))

    if is_conv is None and trend is None and proposal_count is None:
        return None

    base = 1.0 if is_conv else 0.55
    if trend is None:
        trend_score = base
    else:
        # trend <= -1 is strongly converging, trend >= 3 means divergence/noisy churn
        trend_score = _clamp01(1.0 - ((trend + 1.0) / 4.0))

    proposal_bonus = None
    if proposal_count is not None:
        # some proposal activity is healthy, but diminishing returns past ~8
        proposal_bonus = _clamp01(float(proposal_count) / 8.0)

    parts = [0.6 * base + 0.4 * trend_score]
    if proposal_bonus is not None:
        parts.append(0.85 * parts[0] + 0.15 * proposal_bonus)
    return _clamp01(parts[-1])


def _score_failure_resilience(record: Dict[str, Any]) -> Optional[float]:
    exc = _to_float(_pick(
        record,
        ("harness_observability", "harness_caught_exceptions"),
        ("harness_caught_exceptions",),
    ))
    drift = _to_float(_pick(
        record,
        ("harness_observability", "role_drift_detected_count"),
        ("role_drift_detected_count",),
    ))
    if exc is None and drift is None:
        return None
    exc = exc or 0.0
    drift = drift or 0.0
    penalty = 0.35 * exc + 0.15 * drift
    return _clamp01(1.0 - penalty)


def _score_revision_yield(record: Dict[str, Any]) -> Optional[float]:
    reassign = _to_float(_pick(record, ("cumulative_reassignments",)))
    buffers = _to_float(_pick(record, ("cumulative_buffers_applied",)))
    new_tasks = _to_float(_pick(record, ("cumulative_new_tasks",)))
    mediate = _to_float(_pick(
        record,
        ("supervisor_intervention", "mediation_decisions"),
        ("supervisor_mediation_decisions",),
    ))
    total_tasks = _to_float(_pick(record, ("total_tasks",)))

    if reassign is None and buffers is None and new_tasks is None and mediate is None:
        return None

    total_tasks = max(total_tasks or 1.0, 1.0)
    reassign = reassign or 0.0
    buffers = buffers or 0.0
    new_tasks = new_tasks or 0.0

    # Normalized impact score with conservative saturation.
    reassignment_score = _clamp01((reassign / total_tasks) / 0.25)
    buffer_score = _clamp01((buffers / total_tasks) / 0.35)
    new_task_score = _clamp01((new_tasks / total_tasks) / 0.15)

    score = 0.35 * reassignment_score + 0.45 * buffer_score + 0.20 * new_task_score
    if mediate is not None:
        decision_score = _clamp01(float(mediate) / max(total_tasks * 0.40, 1.0))
        score = 0.8 * score + 0.2 * decision_score
    return _clamp01(score)


def _score_faithfulness_gate(faithfulness: Optional[float]) -> Optional[float]:
    """Evidence gate: unsupported claims should cap WBS quality, not just average in.

    Faithfulness is N/A for non-RAG conditions in older experiments. In that
    case the gate is not applied, preserving cross-condition backfillability.
    """
    if faithfulness is None:
        return None
    f = _clamp01(faithfulness)
    if f >= FAITHFULNESS_GATE_THRESHOLD:
        return 1.0
    return max(FAITHFULNESS_GATE_FLOOR, f / FAITHFULNESS_GATE_THRESHOLD)


def recompute_autoscore(
    record: Dict[str, Any],
    top_level_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    cond = _condition(record)
    rounds = _debate_rounds(record)
    top_weights = dict(top_level_weights or TOP_LEVEL_WEIGHTS)

    faithfulness = _to_float(_pick(
        record,
        ("ragas_faithfulness", "faithfulness"),
        ("faithfulness",),
        ("ragas_faithfulness",),
    ))
    quality_parts = {
        "success_rate": _to_float(_pick(record, ("success_rate", "success_rate"), ("success_rate",))),
        "mece_score": _to_float(_pick(record, ("mece_score", "mece_score"), ("mece_score",))),
        "granularity_fitness": _to_float(_pick(record, ("granularity_fitness", "granularity_fitness"), ("granularity_fitness",))),
    }
    quality_raw = _mean_weighted(
        {k: (_clamp01(v) if v is not None else None) for k, v in quality_parts.items()},
        QUALITY_WEIGHTS,
    )
    faithfulness_gate = _score_faithfulness_gate(faithfulness)
    quality = quality_raw
    if quality is not None and faithfulness_gate is not None:
        quality *= faithfulness_gate

    allocation_active = cond != "C0_llm_only"
    allocation = None
    allocation_parts = {}
    if allocation_active:
        allocation_parts = {
            "planning_score": _to_float(_pick(record, ("planning_score", "planning_score"), ("planning_score",))),
            "schedule_feasibility": _to_float(_pick(record, ("schedule_feasibility", "feasibility"), ("schedule_feasibility",))),
            "buffer_adequacy": _score_buffer_adequacy(_to_float(_pick(record, ("buffer_ratio", "buffer_ratio_pct"), ("buffer_ratio_pct",)))),
            "workload_balance": _score_workload_balance(_to_float(_pick(record, ("workload_gini", "gini"), ("workload_gini",)))),
        }
        allocation = _mean_weighted(
            {k: (_clamp01(v) if v is not None else None) for k, v in allocation_parts.items()},
            ALLOCATION_WEIGHTS,
        )

    orchestration_active = cond not in {"C0_llm_only", "C1_with_assign"} and rounds > 0
    orchestration = None
    orchestration_parts = {}
    if orchestration_active:
        orchestration_parts = {
            "communication_efficiency": _to_float(_pick(record, ("communication_efficiency", "efficiency"), ("comm_efficiency",))),
            "convergence": _score_convergence(record),
            "revision_yield": _score_revision_yield(record),
            "failure_resilience": _score_failure_resilience(record),
        }
        orchestration = _mean_weighted(
            {k: (_clamp01(v) if v is not None else None) for k, v in orchestration_parts.items()},
            ORCHESTRATION_WEIGHTS,
        )

    category_scores = {
        "quality": quality,
        "allocation": allocation,
        "orchestration": orchestration,
    }
    overall_applicable = _mean_weighted(category_scores, top_weights)
    overall = _mean_full_scope(category_scores, top_weights)
    category_weights = _active_weights(category_scores, top_weights)
    na_categories = [k for k, v in category_scores.items() if v is None]
    return {
        "version": AUTOSCORE_VERSION,
        "autoscore": round(overall or 0.0, 4),
        "autoscore_applicable": round(overall_applicable or 0.0, 4),
        "quality": round(quality or 0.0, 4),
        "allocation": round(allocation or 0.0, 4),
        "orchestration": round(orchestration or 0.0, 4),
        "weights_used": {k: round(top_weights[k], 4) for k in top_weights},
        "applicable_weights_used": {k: round(v, 4) for k, v in category_weights.items()},
        "na_categories": na_categories,
        "faithfulness_gate": round(faithfulness_gate, 4) if faithfulness_gate is not None else None,
        "components": {
            "quality": {
                **{k: (round(_clamp01(v), 4) if v is not None else None) for k, v in quality_parts.items()},
                "faithfulness": round(_clamp01(faithfulness), 4) if faithfulness is not None else None,
                "quality_raw": round(quality_raw or 0.0, 4),
                "weights_used": {k: round(v, 4) for k, v in _active_weights(quality_parts, QUALITY_WEIGHTS).items()},
            },
            "allocation": {
                **{k: (round(_clamp01(v), 4) if v is not None else None) for k, v in allocation_parts.items()},
                "weights_used": {k: round(v, 4) for k, v in _active_weights(allocation_parts, ALLOCATION_WEIGHTS).items()},
            },
            "orchestration": {
                **{k: (round(_clamp01(v), 4) if v is not None else None) for k, v in orchestration_parts.items()},
                "weights_used": {k: round(v, 4) for k, v in _active_weights(orchestration_parts, ORCHESTRATION_WEIGHTS).items()},
            },
        },
    }


def enrich_rows_with_autoscore(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        scored = recompute_autoscore(row)
        item = dict(row)
        item["autoscore_final"] = scored["autoscore"]
        item["autoscore_applicable"] = scored["autoscore_applicable"]
        item["autoscore_quality"] = scored["quality"]
        item["autoscore_allocation"] = scored["allocation"]
        item["autoscore_orchestration"] = scored["orchestration"]
        item["autoscore_version"] = scored["version"]
        item["autoscore_faithfulness_gate"] = scored["faithfulness_gate"]
        item["autoscore_na_cats"] = ",".join(scored.get("na_categories", []) or [])
        enriched.append(item)
    return enriched
