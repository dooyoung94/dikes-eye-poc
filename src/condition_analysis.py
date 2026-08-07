from __future__ import annotations

from typing import Any

from src.condition_taxonomy import aspect_label, normalized_preferences


def _safe_rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def normalize_context_conditions(context: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = context.get("conditions")
    if isinstance(raw_items, list) and raw_items:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_items[:6]:
            if not isinstance(item, dict):
                continue
            aspect = str(item.get("aspect") or "").strip()
            if not aspect or aspect in seen:
                continue
            seen.add(aspect)
            direction = str(item.get("direction") or "prefer").strip().lower()
            if direction not in {"prefer", "avoid", "tolerate"}:
                direction = "prefer"
            try:
                importance = float(item.get("importance", 0.8))
            except (TypeError, ValueError):
                importance = 0.8
            out.append({
                "raw": str(item.get("raw") or aspect_label(aspect)).strip(),
                "aspect": aspect,
                "label": str(item.get("label") or aspect_label(aspect)),
                "direction": direction,
                "importance": max(0.1, min(1.0, importance)),
                "search_terms": [str(x) for x in item.get("search_terms", []) if str(x).strip()][:5],
            })
        if out:
            return out

    fallback = []
    for item in normalized_preferences(str(context.get("preference") or ""), limit=6):
        fallback.append({
            **item,
            "direction": "prefer",
            "importance": 0.8,
            "search_terms": [item["term"]],
        })
    return fallback


def analyze_conditions(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = normalize_context_conditions(context)
    results: list[dict[str, Any]] = []

    for cond in conditions:
        aspect = str(cond.get("aspect") or "")
        aspect_rows = [
            row for row in rows
            if aspect in row.get("aspects", []) and int(row.get("sentiment", 0)) != 0
        ]
        positive = [row for row in aspect_rows if int(row.get("sentiment", 0)) > 0]
        negative = [row for row in aspect_rows if int(row.get("sentiment", 0)) < 0]
        total = len(aspect_rows)

        direct_rows = [
            row for row in aspect_rows
            if bool(row.get("condition_direct"))
            or str(row.get("retrieval_condition_aspect") or "") == aspect
        ]
        direct_positive = [row for row in direct_rows if int(row.get("sentiment", 0)) > 0]
        direct_negative = [row for row in direct_rows if int(row.get("sentiment", 0)) < 0]

        situational_rows = [
            row for row in aspect_rows
            if bool(row.get("situational_aligned"))
        ]
        situational_positive = [row for row in situational_rows if int(row.get("sentiment", 0)) > 0]
        situational_negative = [row for row in situational_rows if int(row.get("sentiment", 0)) < 0]

        positive_rate = _safe_rate(len(positive), total)
        negative_rate = _safe_rate(len(negative), total)
        situational_negative_rate = _safe_rate(len(situational_negative), len(situational_rows))
        situational_lift = situational_negative_rate - negative_rate if situational_rows else 0.0

        direction = str(cond.get("direction") or "prefer")
        if direction == "avoid":
            desirability = negative_rate
            # avoid 조건은 부정 신호가 많을수록 사용자의 회피 요구와 충돌하므로 낮은 적합도
            fit = 1.0 - negative_rate
        elif direction == "tolerate":
            desirability = positive_rate
            fit = 0.5 + 0.5 * (positive_rate - negative_rate)
        else:
            desirability = positive_rate
            fit = positive_rate

        support = min(1.0, total / 10.0)
        direct_support = min(1.0, len(direct_rows) / 6.0)
        evidence_confidence = min(1.0, 0.65 * support + 0.35 * direct_support)

        results.append({
            **cond,
            "total_count": total,
            "positive_count": len(positive),
            "negative_count": len(negative),
            "positive_rate": round(positive_rate, 4),
            "negative_rate": round(negative_rate, 4),
            "direct_count": len(direct_rows),
            "direct_positive_count": len(direct_positive),
            "direct_negative_count": len(direct_negative),
            "situational_count": len(situational_rows),
            "situational_positive_count": len(situational_positive),
            "situational_negative_count": len(situational_negative),
            "situational_negative_rate": round(situational_negative_rate, 4),
            "situational_lift": round(situational_lift, 4),
            "fit": round(max(0.0, min(1.0, fit)), 4),
            "desirability": round(max(0.0, min(1.0, desirability)), 4),
            "evidence_confidence": round(evidence_confidence, 4),
            "enough_evidence": total >= 3,
            "interpretation": (
                "조건 자체 Evidence는 해당 aspect로 분류된 전체 Evidence를 사용하고, "
                "조건 전용 검색은 coverage 보강용으로만 사용합니다."
            ),
        })

    return results
