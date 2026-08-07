from __future__ import annotations

from typing import Any


def build_rashomon(rca: dict[str, Any]) -> dict[str, Any]:
    conflicts = []
    candidates = rca.get("cause_candidates", [])

    for conflict in rca.get("conflicts", []):
        aspect = conflict.get("aspect", "other")
        split_causes = [
            c for c in candidates
            if c.get("aspect") == aspect
        ][:3]
        conflicts.append({
            "aspect": aspect,
            "positive_count": conflict.get("positive_count", 0),
            "negative_count": conflict.get("negative_count", 0),
            "neutral_count": conflict.get("neutral_count", 0),
            "conflict_strength": conflict.get("conflict_strength", 0),
            "split_causes": split_causes,
            "evidence_level": "observed_association" if split_causes else "insufficient_evidence",
        })

    return {
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "summary": f"상반된 평가가 확인된 핵심 aspect {len(conflicts)}개",
    }
