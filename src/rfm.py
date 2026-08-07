from collections import Counter, defaultdict
from math import log1p
from typing import Any


def build_rfm(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = Counter()
    sources = defaultdict(set)
    for row in rows:
        source = str(row.get("source") or "unknown")
        for aspect in row.get("aspects", ["other"]):
            counts[aspect] += 1
            sources[aspect].add(source)

    max_count = max(counts.values(), default=1)
    denom = log1p(max_count)
    enriched = []
    for row in rows:
        f_score = 0.0
        for aspect in row.get("aspects", ["other"]):
            freq = log1p(counts[aspect]) / denom if denom else 0.0
            diversity = min(1.0, len(sources[aspect]) / 3.0)
            f_score = max(f_score, 0.75 * freq + 0.25 * diversity)
        priority = 0.35 * float(row.get("R", 0)) + 0.25 * f_score + 0.40 * float(row.get("M", 0))
        enriched.append({**row, "F": round(f_score, 4), "priority": round(priority, 4)})

    enriched.sort(key=lambda x: x["priority"], reverse=True)
    n = len(enriched)
    summary = {
        "formula": "0.35R + 0.25F + 0.40M",
        "avg_R": round(sum(x.get("R", 0) for x in enriched) / n, 4) if n else 0.0,
        "avg_F": round(sum(x.get("F", 0) for x in enriched) / n, 4) if n else 0.0,
        "avg_M": round(sum(x.get("M", 0) for x in enriched) / n, 4) if n else 0.0,
        "top_evidence_ids": [x["evidence_id"] for x in enriched[:10]],
    }
    return enriched, summary
