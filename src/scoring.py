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


def _positive_strength(eda: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """반복적으로 긍정 평가가 우세한 aspect를 0~1 strength로 계산한다.

    단순 긍정 건수만 보지 않고 의견 수와 긍정 비율을 함께 본다.
    표본이 3건 미만인 aspect는 strength에 사용하지 않는다.
    """
    strengths: list[dict[str, Any]] = []
    for aspect, counts in eda.get("aspect_sentiment", {}).items():
        positive = int(counts.get("1", 0))
        negative = int(counts.get("-1", 0))
        opinion_count = positive + negative
        if opinion_count < 3 or positive <= negative:
            continue

        positive_rate = positive / opinion_count
        support = min(1.0, opinion_count / 8.0)
        strength = positive_rate * support
        strengths.append({
            "aspect": aspect,
            "positive_count": positive,
            "negative_count": negative,
            "opinion_count": opinion_count,
            "positive_rate": round(positive_rate, 4),
            "strength": round(strength, 4),
        })

    strengths.sort(key=lambda x: (x["strength"], x["opinion_count"]), reverse=True)
    top = strengths[:3]
    if not top:
        return 0.0, []

    score = sum(float(x["strength"]) for x in top) / len(top)
    return min(1.0, score), top


def score_decision(
    visible_rows: list[dict[str, Any]],
    eda: dict[str, Any],
    rfm_summary: dict[str, Any],
    rca: dict[str, Any],
    wald: dict[str, Any],
) -> dict[str, Any]:
    n = len(visible_rows)
    source_count = len(eda.get("source_counts", {}))

    count_quality = min(1.0, n / 30.0)
    source_quality = min(1.0, source_count / 3.0)
    recency_quality = float(eda.get("avg_recency", 0.0))
    context_quality = min(1.0, float(rfm_summary.get("avg_M", 0.0)) * 2.0)

    evidence_quality = (
        0.30 * count_quality
        + 0.20 * source_quality
        + 0.25 * recency_quality
        + 0.25 * context_quality
    )

    conflict_count = len(rca.get("conflicts", []))
    explained_aspects = {
        c.get("aspect") for c in rca.get("cause_candidates", [])
    }
    unresolved = sum(
        1 for conflict in rca.get("conflicts", [])
        if conflict.get("aspect") not in explained_aspects
    )

    # 충돌 자체는 나쁜 것이 아니라 '불확실성'이므로 confidence에만 작게 반영한다.
    conflict_penalty = min(0.15, unresolved * 0.03)
    confidence = max(0.20, min(0.95, evidence_quality - conflict_penalty))

    sentiment = _weighted_sentiment(visible_rows)
    context_sentiment = _weighted_sentiment(visible_rows, context_only=True)
    positive_strength, positive_aspects = _positive_strength(eda)

    rca_risk = float(rca.get("aligned_risk", 0.0))
    wald_risk = (
        0.60 * float(wald.get("signal_score", 0.0))
        + 0.40 * float(wald.get("severe_signal", 0.0))
    )

    # v2: 긍정 강점은 명시적으로 보상하고, 위험은 사용자 조건에 맞물릴 때만 보수적으로 감점.
    raw = (
        50.0
        + 25.0 * sentiment
        + 18.0 * context_sentiment
        + 10.0 * positive_strength
        - 10.0 * rca_risk
        - 8.0 * wald_risk
    )
    raw = max(0.0, min(100.0, raw))

    # 저신뢰도에서 50점으로 지나치게 수축되던 문제 완화.
    shrink = 0.55 + 0.45 * confidence
    score = 50.0 + (raw - 50.0) * shrink

    caps: list[str] = []
    if n < 6:
        score = min(score, 65.0)
        caps.append("visible evidence < 6")
    elif n < 12:
        score = min(score, 72.0)
        caps.append("visible evidence < 12")

    if confidence < 0.45:
        score = min(score, 70.0)
        caps.append("very low confidence")

    # 높은 위험에서도 긍정 Evidence를 완전히 무시하지 않되 강한 추천은 제한한다.
    if rca_risk >= 0.75 or float(wald.get("severe_signal", 0.0)) >= 0.85:
        score = min(score, 62.0)
        caps.append("high aligned risk")

    score = round(max(0.0, min(100.0, score)), 1)
    confidence_pct = round(confidence * 100.0, 1)

    if (
        score >= 70
        and confidence >= 0.58
        and rca_risk < 0.60
        and float(wald.get("severe_signal", 0.0)) < 0.70
        and n >= 12
    ):
        verdict = "GO"
    elif score <= 40 and confidence >= 0.50:
        verdict = "AVOID"
    elif rca_risk >= 0.90 or float(wald.get("severe_signal", 0.0)) >= 0.95:
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
            "positive_strength": round(positive_strength, 4),
            "positive_aspects": positive_aspects,
            "rca_risk": round(rca_risk, 4),
            "wald_risk": round(wald_risk, 4),
            "evidence_quality": round(evidence_quality, 4),
            "shrink_to_neutral": round(shrink, 4),
            "conflict_count": conflict_count,
            "unresolved_conflicts": unresolved,
        },
        "score_caps": caps,
        "policy": {
            "version": "balanced-v2",
            "go_threshold": "score>=70 AND confidence>=58 AND n>=12 AND no high aligned risk",
            "principle": (
                "positive repeated strengths are rewarded; conflicts mainly reduce certainty; "
                "RCA/Wald only reduce fit when risk is materially aligned with the user's conditions"
            ),
        },
    }
