from __future__ import annotations

from collections import Counter
from typing import Any


HIDDEN_SIGNAL_RULES = {
    "wait_abandonment": {
        "keywords": ["웨이팅 포기", "대기 포기", "기다리다 포기", "다른 식당", "다른 곳으로"],
        "weight": 1.0,
    },
    "reservation_failure": {
        "keywords": ["예약 실패", "예약 마감", "예약이 꽉", "예약 못", "예약 불가"],
        "weight": 1.0,
    },
    "parking_failure": {
        "keywords": ["주차 못", "주차 실패", "주차장을 못", "주차 포기"],
        "weight": 0.8,
    },
    "regret_churn": {
        "keywords": ["후회", "다시 안", "재방문 안", "다시는 안"],
        "weight": 0.7,
    },
}


def analyze_wald(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    evidence_by_category: dict[str, list[str]] = {}
    weighted_hits = 0.0
    sources: set[str] = set()

    for row in rows:
        text = str(row.get("text") or f"{row.get('title', '')} {row.get('snippet', '')}")
        sources.add(str(row.get("source") or "unknown"))

        for category, rule in HIDDEN_SIGNAL_RULES.items():
            if any(keyword in text for keyword in rule["keywords"]):
                counts[category] += 1
                weighted_hits += float(rule["weight"])
                evidence_by_category.setdefault(category, []).append(
                    str(row.get("evidence_id") or "")
                )

    # 신호가 몇 건 검색됐는지를 위험률로 해석하지 않는다.
    diversity = min(1.0, len(sources) / 3.0)
    volume_signal = min(1.0, weighted_hits / 8.0)
    signal_score = min(1.0, 0.80 * volume_signal + 0.20 * diversity) if rows else 0.0

    severe_hits = counts["wait_abandonment"] + counts["reservation_failure"]
    severe_signal = min(1.0, severe_hits / 5.0)

    return {
        "hidden_evidence_count": len(rows),
        "source_diversity": len(sources),
        "signal_counts": dict(counts),
        "evidence_by_category": evidence_by_category,
        "signal_score": round(signal_score, 4),
        "severe_signal": round(severe_signal, 4),
        "coverage": "limited" if len(rows) < 8 else "usable",
        "interpretation": (
            "Wald 값은 예약 실패·웨이팅 포기 등 선택편향/생존자편향의 위험 신호입니다. "
            "전체 방문자 대비 실제 실패율이나 발생확률을 뜻하지 않습니다."
        ),
    }
