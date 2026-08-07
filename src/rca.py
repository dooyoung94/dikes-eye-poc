from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.condition_analysis import analyze_conditions


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
    digits = "".join(ch for ch in time[:3] if ch.isdigit())
    try:
        hour = int(digits) if digits else -1
    except ValueError:
        hour = -1
    if 11 <= hour <= 14:
        flags.add("lunch")
    if 17 <= hour <= 22:
        flags.add("dinner")
    flags.update(PURPOSE_FLAGS.get(purpose, set()))
    return flags


def _context_label(context: dict[str, Any]) -> str:
    parts = [
        str(context.get("date_or_day") or "").strip(),
        str(context.get("time") or "").strip(),
        str(context.get("purpose") or "").strip(),
    ]
    return " · ".join(x for x in parts if x) or "사용 상황"


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
        opinion_count = pos + neg
        if pos < 2 or neg < 2 or opinion_count < 5:
            continue
        conflicts.append({
            "aspect": aspect,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neutral,
            "opinion_count": opinion_count,
            "total_count": opinion_count + neutral,
            "positive_rate": round(pos / opinion_count, 4),
            "negative_rate": round(neg / opinion_count, 4),
            "conflict_strength": round(min(pos, neg) / opinion_count, 4),
            "positive_evidence": buckets["positive"][:8],
            "negative_evidence": buckets["negative"][:8],
        })
    conflicts.sort(key=lambda x: x["conflict_strength"], reverse=True)
    return conflicts


def _situational_candidates(
    condition_results: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    label = _context_label(context)
    out: list[dict[str, Any]] = []
    for item in condition_results:
        if int(item.get("situational_count", 0)) < 3:
            continue
        lift = float(item.get("situational_lift", 0.0))
        effect = "worsens" if lift > 0.03 else "improves" if lift < -0.03 else "similar"
        confidence = min(
            0.92,
            0.35
            + 0.30 * min(1.0, abs(lift) * 2)
            + 0.25 * min(1.0, int(item.get("situational_count", 0)) / 8.0),
        )
        out.append({
            "aspect": item.get("aspect"),
            "context": label,
            "effect": effect,
            "baseline_total_count": int(item.get("total_count", 0)),
            "baseline_positive_count": int(item.get("positive_count", 0)),
            "baseline_negative_count": int(item.get("negative_count", 0)),
            "baseline_negative_rate": float(item.get("negative_rate", 0.0)),
            "context_total_count": int(item.get("situational_count", 0)),
            "context_positive_count": int(item.get("situational_positive_count", 0)),
            "context_negative_count": int(item.get("situational_negative_count", 0)),
            "context_negative_rate": float(item.get("situational_negative_rate", 0.0)),
            "lift": round(lift, 4),
            "support_count": int(item.get("situational_count", 0)),
            "confidence": round(confidence, 4),
            "user_aligned": True,
            "claim_level": "observed_association",
            "analysis_scope": "situational_condition_shift",
        })
    out.sort(
        key=lambda x: (
            abs(float(x["lift"])) * float(x["confidence"]),
            x["support_count"],
        ),
        reverse=True,
    )
    return out


def derive_rca(rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    conflicts = build_conflicts(rows)
    condition_results = analyze_conditions(rows, context)
    situational = _situational_candidates(condition_results, context)
    worsening = [x for x in situational if x.get("effect") == "worsens"]
    aligned_risk = max(
        (
            min(1.0, abs(float(x.get("lift", 0))) * float(x.get("confidence", 0)) * 1.6)
            for x in worsening
        ),
        default=0.0,
    )

    return {
        "conflicts": conflicts,
        "condition_results": condition_results,
        "cause_candidates": situational[:12],
        "main_candidates": situational[:12],
        "diagnostic_candidates": situational[:20],
        "user_context_flags": sorted(_user_flags(context)),
        "user_context_label": _context_label(context),
        "aligned_risk": round(aligned_risk, 4),
        "interpretation": (
            "중요조건 자체 평가는 해당 Aspect의 전체 Evidence로 계산하고, "
            "요일·시간·목적에 따른 변화는 별도 상황 subset으로 비교합니다."
        ),
    }
