from __future__ import annotations

from typing import Any


def _weighted_sentiment(rows: list[dict[str, Any]], *, context_only: bool = False) -> float:
    selected = [
        row for row in rows
        if not context_only or float(row.get("M", 0.0)) >= 0.20
    ]
    if not selected:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for row in selected:
        weight = max(0.05, float(row.get("priority", 0.0)))
        numerator += int(row.get("sentiment", 0)) * weight
        denominator += weight

    return max(-1.0, min(1.0, numerator / denominator if denominator else 0.0))


def score_decision(
    visible_rows: list[dict[str, Any]],
    eda: dict[str, Any],
    rfm_summary: dict[str, Any],
    rca: dict[str, Any],
    wald: dict[str, Any],
) -> dict[str, Any]:
    n = len(visible_rows)
    source_count = len(eda.get("source_counts", {}))

    count_quality = min(1.0, n / 35.0)
    source_quality = min(1.0, source_count / 3.0)
    recency_quality = float(eda.get("avg_recency", 0.0))
    context_quality = min(1.0, float(rfm_summary.get("avg_M", 0.0)) * 2.0)

    evidence_quality = (
        0.35 * count_quality
        + 0.20 * source_quality
        + 0.25 * recency_quality
        + 0.20 * context_quality
    )

    conflict_count = len(rca.get("conflicts", []))
    explained_aspects = {
        c.get("aspect") for c in rca.get("cause_candidates", [])
    }
    unresolved = sum(
        1 for conflict in rca.get("conflicts", [])
        if conflict.get("aspect") not in explained_aspects
    )
    conflict_penalty = min(0.25, unresolved * 0.05)
    confidence = max(0.15, min(0.95, evidence_quality - conflict_penalty))

    sentiment = _weighted_sentiment(visible_rows)
    context_sentiment = _weighted_sentiment(visible_rows, context_only=True)
    rca_risk = float(rca.get("aligned_risk", 0.0))
    wald_risk = (
        0.65 * float(wald.get("signal_score", 0.0))
        + 0.35 * float(wald.get("severe_signal", 0.0))
    )

    raw = (
        50.0
        + 22.0 * sentiment
        + 15.0 * context_sentiment
        - 14.0 * rca_risk
        - 14.0 * wald_risk
    )
    raw = max(0.0, min(100.0, raw))

    shrink = 0.30 + 0.70 * confidence
    score = 50.0 + (raw - 50.0) * shrink

    caps: list[str] = []
    if n < 8:
        score = min(score, 62.0)
        caps.append("visible evidence < 8")
    if n < 15:
        score = min(score, 68.0)
        caps.append("visible evidence < 15")
    if confidence < 0.55:
        score = min(score, 67.0)
        caps.append("low confidence")
    if rca_risk >= 0.65 or float(wald.get("severe_signal", 0.0)) >= 0.75:
        score = min(score, 55.0)
        caps.append("high aligned risk")

    score = round(max(0.0, min(100.0, score)), 1)
    confidence_pct = round(confidence * 100.0, 1)

    if (
        score >= 72
        and confidence >= 0.65
        and rca_risk < 0.50
        and float(wald.get("severe_signal", 0.0)) < 0.60
        and n >= 15
    ):
        verdict = "GO"
    elif score <= 42 and confidence >= 0.55:
        verdict = "AVOID"
    elif rca_risk >= 0.80 or float(wald.get("severe_signal", 0.0)) >= 0.90:
        verdict = "AVOID"
    else:
        verdict = "CONDITIONAL"

    return {
        "verdict": verdict,
        "fit_score": score,
        "confidence": confidence_pct,
        "components": {
            "weighted_sentiment": round(sentiment, 4),
            "context_sentiment": round(context_sentiment, 4),
            "rca_risk": round(rca_risk, 4),
            "wald_risk": round(wald_risk, 4),
            "evidence_quality": round(evidence_quality, 4),
            "shrink_to_neutral": round(shrink, 4),
            "conflict_count": conflict_count,
            "unresolved_conflicts": unresolved,
        },
        "score_caps": caps,
        "policy": {
            "go_threshold": "score>=72 AND confidence>=65 AND n>=15 AND no high aligned risk",
            "principle": "low evidence shrinks toward neutral 50; low-confidence evidence cannot receive a strong recommendation",
        },
    }
