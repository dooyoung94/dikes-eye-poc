from collections import Counter, defaultdict
from typing import Any


def build_eda(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter()
    aspect_counts = Counter()
    context_counts = Counter()
    sentiment_counts = Counter()
    aspect_sentiment = defaultdict(lambda: Counter())

    for row in rows:
        source_counts[str(row.get("source") or "unknown")] += 1
        sentiment = int(row.get("sentiment", 0))
        sentiment_counts[str(sentiment)] += 1
        for aspect in row.get("aspects", ["other"]):
            aspect_counts[aspect] += 1
            aspect_sentiment[aspect][str(sentiment)] += 1
        for context in row.get("contexts", []):
            context_counts[context] += 1

    total = len(rows)
    avg_r = sum(float(r.get("R", 0)) for r in rows) / total if total else 0.0
    avg_m = sum(float(r.get("M", 0)) for r in rows) / total if total else 0.0

    return {
        "total": total,
        "source_counts": dict(source_counts),
        "aspect_counts": dict(aspect_counts),
        "context_counts": dict(context_counts),
        "sentiment_counts": dict(sentiment_counts),
        "aspect_sentiment": {k: dict(v) for k, v in aspect_sentiment.items()},
        "avg_recency": round(avg_r, 4),
        "avg_context_match": round(avg_m, 4),
    }
