from __future__ import annotations

from collections import Counter
from typing import Any


RULES = {
    "restaurant": {
        "wait_abandonment": {
            "keywords": ["웨이팅 포기", "대기 포기", "기다리다 포기", "다른 식당", "다른 곳으로"],
            "weight": 1.0,
            "severe": True,
        },
        "reservation_failure": {
            "keywords": ["예약 실패", "예약 마감", "예약이 꽉", "예약 못", "예약 불가"],
            "weight": 1.0,
            "severe": True,
        },
        "parking_failure": {
            "keywords": ["주차 못", "주차 실패", "주차장을 못", "주차 포기"],
            "weight": 0.8,
            "severe": False,
        },
        "regret_churn": {
            "keywords": ["후회", "다시 안", "재방문 안", "다시는 안"],
            "weight": 0.7,
            "severe": False,
        },
    },
    "product": {
        "return_refund": {
            "keywords": ["반품", "환불", "교환", "반품했", "환불했"],
            "weight": 1.0,
            "severe": True,
        },
        "defect_failure": {
            "keywords": ["초기 불량", "불량", "고장", "먹통", "작동 안", "연결 안"],
            "weight": 1.0,
            "severe": True,
        },
        "resale_exit": {
            "keywords": ["중고 판매", "중고로 판매", "중고로 팔", "처분", "당근에"],
            "weight": 0.8,
            "severe": False,
        },
        "regret_churn": {
            "keywords": ["후회", "다시 안", "재구매 안", "다시는 안", "괜히 샀"],
            "weight": 0.7,
            "severe": False,
        },
    },
}


def analyze_wald(rows: list[dict[str, Any]], kind: str = "restaurant") -> dict[str, Any]:
    rules = RULES.get(kind, RULES["restaurant"])
    counts: Counter[str] = Counter()
    evidence_by_category: dict[str, list[str]] = {}
    weighted_hits = 0.0
    severe_hits = 0
    sources: set[str] = set()

    for row in rows:
        text = str(row.get("text") or f"{row.get('title', '')} {row.get('snippet', '')}")
        sources.add(str(row.get("source") or "unknown"))

        for category, rule in rules.items():
            if any(keyword in text for keyword in rule["keywords"]):
                counts[category] += 1
                weighted_hits += float(rule["weight"])
                severe_hits += 1 if rule.get("severe") else 0
                evidence_by_category.setdefault(category, []).append(
                    str(row.get("evidence_id") or "")
                )

    diversity = min(1.0, len(sources) / 3.0)
    volume_signal = min(1.0, weighted_hits / 8.0)
    signal_score = min(1.0, 0.80 * volume_signal + 0.20 * diversity) if rows else 0.0
    severe_signal = min(1.0, severe_hits / 5.0)

    if kind == "product":
        interpretation = (
            "Wald 값은 반품·환불·고장·재판매처럼 만족한 사용자의 일반 리뷰에 덜 남을 수 있는 "
            "이탈 신호를 탐지한 결과입니다. 실제 불량률이나 반품률을 뜻하지 않습니다."
        )
    else:
        interpretation = (
            "Wald 값은 예약 실패·웨이팅 포기처럼 실제 방문 리뷰에 남기 어려운 선택 이전 이탈 신호를 "
            "탐지한 결과입니다. 전체 방문자 대비 실제 실패율을 뜻하지 않습니다."
        )

    return {
        "kind": kind,
        "hidden_evidence_count": len(rows),
        "source_diversity": len(sources),
        "signal_counts": dict(counts),
        "evidence_by_category": evidence_by_category,
        "signal_score": round(signal_score, 4),
        "severe_signal": round(severe_signal, 4),
        "coverage": "limited" if len(rows) < 6 else "usable",
        "interpretation": interpretation,
    }
