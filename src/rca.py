from __future__ import annotations

from collections import defaultdict
from typing import Any


PURPOSE_FLAGS = {
    "출퇴근": {"commute"},
    "업무": {"office"},
    "업무 미팅": {"office"},
    "게임": {"gaming"},
    "여행": {"travel"},
    "운동": {"exercise"},
}


def _user_flags(context: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    day = str(context.get("date_or_day", ""))
    time = str(context.get("time", ""))
    purpose = str(context.get("purpose", ""))

    if any(x in day for x in ["토요일", "일요일", "주말"]):
        flags.add("weekend")
    if any(x in day for x in ["월요일", "화요일", "수요일", "목요일", "금요일", "평일"]):
        flags.add("weekday")

    try:
        digits = "".join(ch for ch in time[:3] if ch.isdigit())
        hour = int(digits) if digits else -1
        if 11 <= hour <= 14:
            flags.add("lunch")
        if 17 <= hour <= 22:
            flags.add("dinner")
    except (ValueError, TypeError):
        pass

    flags.update(PURPOSE_FLAGS.get(purpose, set()))
    return flags


def build_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(lambda: {"positive": [], "negative": [], "neutral": []})

    for row in rows:
        sentiment = int(row.get("sentiment", 0))
        bucket = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        for aspect in row.get("aspects", ["other"]):
            grouped[aspect][bucket].append(row.get("evidence_id", ""))

    conflicts: list[dict[str, Any]] = []
    for aspect, buckets in grouped.items():
        pos = len(buckets["positive"])
        neg = len(buckets["negative"])
        neutral = len(buckets["neutral"])
        total_opinion = pos + neg

        if pos < 2 or neg < 2 or total_opinion < 5:
            continue

        balance = min(pos, neg) / max(1, total_opinion)
        conflicts.append({
            "aspect": aspect,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neutral,
            "opinion_count": total_opinion,
            "total_count": total_opinion + neutral,
            "positive_rate": round(pos / total_opinion, 4),
            "negative_rate": round(neg / total_opinion, 4),
            "conflict_strength": round(balance, 4),
            "positive_evidence": buckets["positive"][:8],
            "negative_evidence": buckets["negative"][:8],
        })

    conflicts.sort(key=lambda x: x["conflict_strength"], reverse=True)
    return conflicts


def derive_rca(rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    conflicts = build_conflicts(rows)
    flags = _user_flags(context)
    candidates: list[dict[str, Any]] = []

    for conflict in conflicts:
        aspect = conflict["aspect"]
        relevant = [
            row for row in rows
            if aspect in row.get("aspects", []) and int(row.get("sentiment", 0)) != 0
        ]
        if not relevant:
            continue

        baseline_negative = [row for row in relevant if int(row.get("sentiment", 0)) < 0]
        baseline_positive = [row for row in relevant if int(row.get("sentiment", 0)) > 0]
        baseline_negative_rate = len(baseline_negative) / len(relevant)

        explicit_contexts = sorted({
            ctx
            for row in relevant
            for ctx in row.get("contexts", [])
        })

        for ctx in explicit_contexts:
            subset = [row for row in relevant if ctx in row.get("contexts", [])]
            if len(subset) < 3:
                continue

            negative = [row for row in subset if int(row.get("sentiment", 0)) < 0]
            positive = [row for row in subset if int(row.get("sentiment", 0)) > 0]
            negative_rate = len(negative) / len(subset)
            lift = negative_rate - baseline_negative_rate

            if abs(lift) < 0.12:
                continue

            support_factor = min(1.0, len(subset) / 8.0)
            confidence = min(
                0.90,
                0.30
                + 0.35 * min(1.0, abs(lift) * 2)
                + 0.20 * support_factor
                + 0.15 * conflict["conflict_strength"],
            )

            effect = "worsens" if lift > 0 else "improves"
            supporting = negative if lift > 0 else positive
            counter = positive if lift > 0 else negative

            candidates.append({
                "aspect": aspect,
                "context": ctx,
                "effect": effect,
                "baseline_total_count": len(relevant),
                "baseline_positive_count": len(baseline_positive),
                "baseline_negative_count": len(baseline_negative),
                "baseline_negative_rate": round(baseline_negative_rate, 4),
                "context_total_count": len(subset),
                "context_positive_count": len(positive),
                "context_negative_count": len(negative),
                "context_negative_rate": round(negative_rate, 4),
                "lift": round(lift, 4),
                "support_count": len(subset),
                "supporting_evidence": [x.get("evidence_id", "") for x in supporting[:8]],
                "counter_evidence": [x.get("evidence_id", "") for x in counter[:8]],
                "confidence": round(confidence, 4),
                "user_aligned": ctx in flags,
                "claim_level": "observed_association",
            })

    candidates.sort(
        key=lambda x: (
            x["user_aligned"],
            abs(x["lift"]) * x["confidence"],
            x["support_count"],
        ),
        reverse=True,
    )

    aligned_worsening = [
        c for c in candidates
        if c["user_aligned"] and c["effect"] == "worsens"
    ]
    aligned_risk = max(
        (
            min(1.0, abs(c["lift"]) * c["confidence"] * 1.8)
            for c in aligned_worsening
        ),
        default=0.0,
    )

    return {
        "conflicts": conflicts,
        "cause_candidates": candidates[:20],
        "user_context_flags": sorted(flags),
        "aligned_risk": round(aligned_risk, 4),
        "interpretation": "조건별 비율 차이를 비교한 관찰 결과이며, 인과관계를 확정하지 않습니다.",
    }
