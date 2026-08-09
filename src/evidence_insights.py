from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.condition_taxonomy import aspect_label


STOPWORDS = {
    "후기", "리뷰", "정말", "진짜", "너무", "조금", "약간", "그냥", "그리고", "하지만", "그래서",
    "있다", "있는", "없다", "없는", "했다", "하는", "되어", "같다", "같은", "좋다", "좋은", "좋았",
    "별로", "추천", "만족", "사용", "방문", "제품", "식당", "매장", "곳이", "곳은", "이번", "정도",
    "해서", "하고", "하는데", "에서", "으로", "에게", "까지", "보다", "관련", "느낌", "생각",
}


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", str(text or "").lower())
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def _top_keywords(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(set(_tokens(str(row.get("text") or ""))))
    return [
        {"keyword": keyword, "count": count}
        for keyword, count in counts.most_common(limit)
    ]


def _sample_rows(rows: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda x: float(x.get("priority", 0.0)), reverse=True)
    samples: list[dict[str, Any]] = []
    for row in ordered[:limit]:
        snippet = str(row.get("snippet") or row.get("text") or "").strip()
        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."
        samples.append({
            "evidence_id": str(row.get("evidence_id") or ""),
            "source": str(row.get("source") or "unknown"),
            "title": str(row.get("title") or "").strip(),
            "snippet": snippet,
            "link": str(row.get("link") or "").strip(),
        })
    return samples


def build_consensus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # 전체 여론은 긍정/부정 전용 검색어가 개입하지 않은 일반 후기 중심으로 계산한다.
    base_rows = [
        row for row in rows
        if str(row.get("retrieval_scope") or "base") == "base"
        and int(row.get("sentiment", 0)) != 0
    ]
    if len(base_rows) < 5:
        base_rows = [
            row for row in rows
            if str(row.get("retrieval_scope") or "") not in {"positive", "negative", "hidden_base"}
            and int(row.get("sentiment", 0)) != 0
        ]

    positive = [row for row in base_rows if int(row.get("sentiment", 0)) > 0]
    negative = [row for row in base_rows if int(row.get("sentiment", 0)) < 0]
    total = len(base_rows)
    positive_rate = len(positive) / total if total else 0.0
    negative_rate = len(negative) / total if total else 0.0

    weighted_num = 0.0
    weighted_den = 0.0
    for row in base_rows:
        weight = max(0.05, float(row.get("priority", 0.0)))
        weighted_num += int(row.get("sentiment", 0)) * weight
        weighted_den += weight
    weighted_sentiment = weighted_num / weighted_den if weighted_den else 0.0
    opinion_score = max(0.0, min(100.0, 50.0 + 50.0 * weighted_sentiment))

    return {
        "sample_scope": "general_review",
        "sample_count": total,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_rate": round(positive_rate, 4),
        "negative_rate": round(negative_rate, 4),
        "weighted_sentiment": round(weighted_sentiment, 4),
        "opinion_score": round(opinion_score, 1),
        "positive_keywords": _top_keywords(positive),
        "negative_keywords": _top_keywords(negative),
        "positive_samples": _sample_rows(positive),
        "negative_samples": _sample_rows(negative),
        "interpretation": "긍정/부정 전용 검색어를 제외한 일반 후기 중심의 수집 표본입니다. 전체 이용자 모집단의 여론조사 결과를 의미하지 않습니다.",
    }


def build_conflict_insights(
    rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    for conflict in conflicts[:limit]:
        aspect = str(conflict.get("aspect") or "other")
        aspect_rows = [
            row for row in rows
            if aspect in row.get("aspects", []) and int(row.get("sentiment", 0)) != 0
        ]
        positive = [row for row in aspect_rows if int(row.get("sentiment", 0)) > 0]
        negative = [row for row in aspect_rows if int(row.get("sentiment", 0)) < 0]
        insights.append({
            "aspect": aspect,
            "label": aspect_label(aspect),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "positive_keywords": _top_keywords(positive),
            "negative_keywords": _top_keywords(negative),
            "positive_samples": _sample_rows(positive),
            "negative_samples": _sample_rows(negative),
        })
    return insights


def build_condition_evidence(
    rows: list[dict[str, Any]],
    condition_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for item in condition_results:
        aspect = str(item.get("aspect") or "")
        aspect_rows = [
            row for row in rows
            if aspect in row.get("aspects", []) and int(row.get("sentiment", 0)) != 0
        ]
        positive = [row for row in aspect_rows if int(row.get("sentiment", 0)) > 0]
        negative = [row for row in aspect_rows if int(row.get("sentiment", 0)) < 0]
        details.append({
            "aspect": aspect,
            "label": str(item.get("label") or aspect_label(aspect)),
            "positive_keywords": _top_keywords(positive, 4),
            "negative_keywords": _top_keywords(negative, 4),
            "positive_samples": _sample_rows(positive, 2),
            "negative_samples": _sample_rows(negative, 2),
        })
    return details
