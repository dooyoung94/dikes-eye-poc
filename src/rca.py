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


def _context_label(context: dict[str, Any], flags: set[str]) -> str:
    parts: list[str] = []
    day = str(context.get("date_or_day") or "").strip()
    time = str(context.get("time") or "").strip()
    purpose = str(context.get("purpose") or "").strip()
    preference = str(context.get("preference") or "").strip()
    if day:
        parts.append(day)
    if time:
        parts.append(time)
    if purpose:
        parts.append(purpose)
    if preference:
        parts.append(preference)
    if parts:
        return " · ".join(parts)
    return ", ".join(sorted(flags)) or "사용자 조건"


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


def _candidate_from_subset(aspect: str, relevant: list[dict[str, Any]], subset: list[dict[str, Any]], label: str, *, user_aligned: bool, conflict_strength: float) -> dict[str, Any] | None:
    if len(subset) < 3:
        return None

    baseline_negative = [r for r in relevant if int(r.get("sentiment", 0)) < 0]
    baseline_positive = [r for r in relevant if int(r.get("sentiment", 0)) > 0]
    negative = [r for r in subset if int(r.get("sentiment", 0)) < 0]
    positive = [r for r in subset if int(r.get("sentiment", 0)) > 0]

    baseline_rate = len(baseline_negative) / len(relevant)
    context_rate = len(negative) / len(subset)
    lift = context_rate - baseline_rate

    # 사용자 조건 subset은 실제로 달라졌는지 보여주기 위해 작은 차이도 반환하되 confidence를 낮춘다.
    support_factor = min(1.0, len(subset) / 8.0)
    confidence = min(
        0.92,
        0.28
        + 0.32 * min(1.0, abs(lift) * 2)
        + 0.25 * support_factor
        + 0.15 * conflict_strength,
    )

    effect = "worsens" if lift > 0.03 else "improves" if lift < -0.03 else "similar"
    supporting = negative if effect == "worsens" else positive if effect == "improves" else subset
    counter = positive if effect == "worsens" else negative if effect == "improves" else []

    return {
        "aspect": aspect,
        "context": label,
        "effect": effect,
        "baseline_total_count": len(relevant),
        "baseline_positive_count": len(baseline_positive),
        "baseline_negative_count": len(baseline_negative),
        "baseline_negative_rate": round(baseline_rate, 4),
        "context_total_count": len(subset),
        "context_positive_count": len(positive),
        "context_negative_count": len(negative),
        "context_negative_rate": round(context_rate, 4),
        "lift": round(lift, 4),
        "support_count": len(subset),
        "supporting_evidence": [x.get("evidence_id", "") for x in supporting[:8]],
        "counter_evidence": [x.get("evidence_id", "") for x in counter[:8]],
        "confidence": round(confidence, 4),
        "user_aligned": user_aligned,
        "claim_level": "observed_association",
    }


def derive_rca(rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    conflicts = build_conflicts(rows)
    flags = _user_flags(context)
    context_label = _context_label(context, flags)
    candidates: list[dict[str, Any]] = []

    # 1) 가장 먼저 사용자 조건에 직접 맞는 subset을 전체 baseline과 비교한다.
    aspects = sorted({aspect for row in rows for aspect in row.get("aspects", ["other"])})
    conflict_map = {c["aspect"]: c for c in conflicts}
    for aspect in aspects:
        relevant = [r for r in rows if aspect in r.get("aspects", []) and int(r.get("sentiment", 0)) != 0]
        if len(relevant) < 4:
            continue

        subset = [r for r in relevant if bool(r.get("context_aligned"))]
        candidate = _candidate_from_subset(
            aspect,
            relevant,
            subset,
            context_label,
            user_aligned=True,
            conflict_strength=float(conflict_map.get(aspect, {}).get("conflict_strength", 0.0)),
        )
        if candidate:
            candidate["analysis_scope"] = "user_context_subset"
            candidates.append(candidate)

    # 2) 보조 설명용으로 명시적 context 패턴도 유지하되 사용자 조건과 일치하는 것만 우선한다.
    for conflict in conflicts:
        aspect = conflict["aspect"]
        relevant = [r for r in rows if aspect in r.get("aspects", []) and int(r.get("sentiment", 0)) != 0]
        explicit_contexts = sorted({ctx for row in relevant for ctx in row.get("contexts", [])})
        for ctx in explicit_contexts:
            subset = [r for r in relevant if ctx in row.get("contexts", [])] if False else [r for r in relevant if ctx in r.get("contexts", [])]
            candidate = _candidate_from_subset(
                aspect,
                relevant,
                subset,
                ctx,
                user_aligned=ctx in flags,
                conflict_strength=float(conflict.get("conflict_strength", 0.0)),
            )
            if candidate:
                candidate["analysis_scope"] = "explicit_context"
                candidates.append(candidate)

    candidates.sort(
        key=lambda x: (
            x.get("analysis_scope") == "user_context_subset",
            x["user_aligned"],
            abs(float(x["lift"])) * float(x["confidence"]),
            x["support_count"],
        ),
        reverse=True,
    )

    aligned_worsening = [c for c in candidates if c["user_aligned"] and c["effect"] == "worsens"]
    aligned_risk = max((min(1.0, abs(float(c["lift"])) * float(c["confidence"]) * 1.6) for c in aligned_worsening), default=0.0)

    return {
        "conflicts": conflicts,
        "cause_candidates": candidates[:24],
        "user_context_flags": sorted(flags),
        "user_context_label": context_label,
        "aligned_risk": round(aligned_risk, 4),
        "interpretation": "사용자 조건으로 회수·매칭된 Evidence subset을 전체 Evidence와 우선 비교합니다. 결과는 관찰된 차이이며 인과관계를 확정하지 않습니다.",
    }
