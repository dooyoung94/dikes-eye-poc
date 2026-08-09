from __future__ import annotations

from typing import Any


def _weighted_sentiment(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = max(0.05, float(row.get("priority", 0.0)))
        numerator += int(row.get("sentiment", 0)) * weight
        denominator += weight
    return max(-1.0, min(1.0, numerator / denominator if denominator else 0.0))


def _positive_strength(eda: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    strengths: list[dict[str, Any]] = []
    for aspect, counts in eda.get("aspect_sentiment", {}).items():
        positive = int(counts.get("1", 0))
        negative = int(counts.get("-1", 0))
        opinion_count = positive + negative
        if opinion_count < 3 or positive <= negative:
            continue
        rate = positive / opinion_count
        support = min(1.0, opinion_count / 8.0)
        strengths.append({
            "aspect": aspect,
            "positive_count": positive,
            "negative_count": negative,
            "opinion_count": opinion_count,
            "positive_rate": round(rate, 4),
            "strength": round(rate * support, 4),
        })
    strengths.sort(key=lambda x: (x["strength"], x["opinion_count"]), reverse=True)
    top = strengths[:3]
    score = sum(float(x["strength"]) for x in top) / len(top) if top else 0.0
    return min(1.0, score), top


def _condition_score(rca: dict[str, Any]) -> tuple[float, float, int]:
    results = [x for x in rca.get("condition_results", []) if x.get("enough_evidence")]
    numerator = 0.0
    denominator = 0.0
    for item in results:
        importance = max(0.1, min(1.0, float(item.get("importance", 0.8))))
        evidence_confidence = max(0.1, min(1.0, float(item.get("evidence_confidence", 0.0))))
        if str(item.get("direction")) == "tolerate":
            importance *= 0.55
        weight = importance * evidence_confidence
        signed_fit = 2.0 * float(item.get("fit", 0.5)) - 1.0
        numerator += signed_fit * weight
        denominator += weight
    signed = numerator / denominator if denominator else 0.0
    coverage = min(1.0, len(results) / max(1, len(rca.get("condition_results", [])))) if rca.get("condition_results") else 0.0
    return max(-1.0, min(1.0, signed)), coverage, len(results)


def score_decision(
    visible_rows: list[dict[str, Any]],
    eda: dict[str, Any],
    rfm_summary: dict[str, Any],
    rca: dict[str, Any],
    wald: dict[str, Any],
    consensus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = len(visible_rows)
    source_count = len(eda.get("source_counts", {}))
    overall_sentiment = _weighted_sentiment(visible_rows)
    positive_strength, positive_aspects = _positive_strength(eda)
    condition_signed, condition_coverage, condition_count = _condition_score(rca)

    consensus = consensus or {}
    consensus_n = int(consensus.get("sample_count", 0))
    if consensus_n >= 5:
        consensus_signed = max(-1.0, min(1.0, float(consensus.get("weighted_sentiment", 0.0))))
        consensus_quality = min(1.0, consensus_n / 20.0)
    else:
        consensus_signed = overall_sentiment
        consensus_quality = min(0.45, n / 20.0)

    count_quality = min(1.0, n / 35.0)
    source_quality = min(1.0, source_count / 3.0)
    recency_quality = float(eda.get("avg_recency", 0.0))
    match_quality = min(1.0, float(rfm_summary.get("avg_M", 0.0)) * 2.0)
    evidence_quality = (
        0.25 * count_quality
        + 0.17 * source_quality
        + 0.20 * recency_quality
        + 0.15 * match_quality
        + 0.15 * condition_coverage
        + 0.08 * consensus_quality
    )

    unresolved = sum(
        1 for c in rca.get("conflicts", [])
        if c.get("aspect") not in {x.get("aspect") for x in rca.get("condition_results", [])}
    )
    confidence = max(0.20, min(0.95, evidence_quality - min(0.10, unresolved * 0.015)))

    rca_risk = float(rca.get("aligned_risk", 0.0))
    wald_risk = (
        0.60 * float(wald.get("signal_score", 0.0))
        + 0.40 * float(wald.get("severe_signal", 0.0))
    )

    # conditional-v5: 전체 여론은 배경 판단, 사용자 조건은 핵심 판단으로 사용한다.
    raw = (
        50.0
        + 13.0 * overall_sentiment
        + 12.0 * consensus_signed
        + 29.0 * condition_signed
        + 8.0 * positive_strength
        - 10.0 * rca_risk
        - 7.0 * wald_risk
    )
    raw = max(0.0, min(100.0, raw))

    shrink = 0.64 + 0.36 * confidence
    score = 50.0 + (raw - 50.0) * shrink

    caps: list[str] = []
    if n < 6:
        score = min(score, 65.0)
        caps.append("visible evidence < 6")
    elif n < 12:
        score = min(score, 72.0)
        caps.append("visible evidence < 12")
    if confidence < 0.42:
        score = min(score, 70.0)
        caps.append("very low confidence")
    if rca_risk >= 0.78 or float(wald.get("severe_signal", 0.0)) >= 0.88:
        score = min(score, 63.0)
        caps.append("high aligned risk")

    score = round(max(0.0, min(100.0, score)), 1)
    confidence_pct = round(confidence * 100.0, 1)

    if (
        score >= 70
        and confidence >= 0.56
        and rca_risk < 0.62
        and float(wald.get("severe_signal", 0.0)) < 0.72
        and n >= 12
    ):
        verdict = "GO"
    elif score <= 40 and confidence >= 0.50:
        verdict = "AVOID"
    elif rca_risk >= 0.92 or float(wald.get("severe_signal", 0.0)) >= 0.96:
        verdict = "AVOID"
    else:
        verdict = "CONDITIONAL"

    return {
        "verdict": verdict,
        "fit_score": score,
        "confidence": confidence_pct,
        "components": {
            "weighted_sentiment": round(overall_sentiment, 4),
            "consensus_sentiment": round(consensus_signed, 4),
            "consensus_score": round(50.0 + 50.0 * consensus_signed, 1),
            "consensus_sample_count": consensus_n,
            "condition_fit": round(condition_signed, 4),
            "condition_score": round(50.0 + 50.0 * condition_signed, 1),
            "condition_coverage": round(condition_coverage, 4),
            "condition_count": condition_count,
            "positive_strength": round(positive_strength, 4),
            "positive_aspects": positive_aspects,
            "rca_risk": round(rca_risk, 4),
            "wald_risk": round(wald_risk, 4),
            "evidence_quality": round(evidence_quality, 4),
            "shrink_to_neutral": round(shrink, 4),
        },
        "score_caps": caps,
        "policy": {
            "version": "conditional-v5-consensus-balance",
            "go_threshold": "score>=70 AND confidence>=56 AND n>=12 AND no high aligned risk",
            "principle": (
                "general-review consensus is treated as background truth; user-stated conditions remain the dominant axis; "
                "situational RCA and Wald risks adjust the final balance"
            ),
        },
    }
