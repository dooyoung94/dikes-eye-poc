from __future__ import annotations

from typing import Any


ASPECT_LABELS = {
    "noise_atmosphere": "분위기·소음",
    "wait_reservation": "웨이팅·예약",
    "service": "서비스·응대",
    "price_value": "가격·가성비",
    "quality_performance": "품질·성능",
    "convenience_fit": "편의성·적합성",
    "design_experience": "디자인·사용경험",
    "other": "기타 평가",
}

WALD_LABELS = {
    "wait_abandonment": "웨이팅 포기",
    "reservation_failure": "예약 실패",
    "parking_failure": "주차 포기",
    "return_refund": "반품·환불",
    "defect_failure": "불량·고장",
    "resale_exit": "재판매·처분",
    "regret_churn": "후회·이탈",
}


def _verdict_label(verdict: str) -> str:
    return {"GO": "추천", "CONDITIONAL": "조건부 추천", "AVOID": "비추천"}.get(verdict, verdict)


def _context_text(context: dict[str, Any]) -> str:
    values = [
        str(context.get("date_or_day") or "").strip(),
        str(context.get("time") or "").strip(),
        str(context.get("purpose") or "").strip(),
        str(context.get("preference") or "").strip(),
    ]
    return " · ".join(x for x in values if x)


def _top_conflict(analysis: dict[str, Any]) -> dict[str, Any] | None:
    conflicts = analysis.get("rashomon", {}).get("conflicts", [])
    return conflicts[0] if conflicts else None


def _top_rca(analysis: dict[str, Any]) -> dict[str, Any] | None:
    candidates = analysis.get("rca", {}).get("cause_candidates", [])
    aligned = [x for x in candidates if x.get("user_aligned")]
    return (aligned or candidates or [None])[0]


def _top_wald(analysis: dict[str, Any]) -> tuple[str, int] | None:
    counts = analysis.get("wald", {}).get("signal_counts", {})
    if not counts:
        return None
    category, count = max(counts.items(), key=lambda x: x[1])
    return category, int(count)


def build_user_report(analysis: dict[str, Any], target: str, kind: str) -> dict[str, Any]:
    decision = analysis.get("decision", {})
    context = analysis.get("context", {})
    verdict = str(decision.get("verdict", "CONDITIONAL"))
    label = _verdict_label(verdict)
    score = decision.get("fit_score", 50)
    confidence = decision.get("confidence", 0)
    context_text = _context_text(context)

    conflict = _top_conflict(analysis)
    rca = _top_rca(analysis)
    wald = _top_wald(analysis)

    why: list[str] = []
    risks: list[str] = []
    actions: list[str] = []

    if conflict:
        aspect = ASPECT_LABELS.get(str(conflict.get("aspect")), str(conflict.get("aspect")))
        why.append(
            f"{aspect}에 대한 평가가 한쪽으로 모이지 않고 갈립니다. 평균 리뷰만 보면 이 차이가 가려질 수 있습니다."
        )
    else:
        why.append("현재 수집된 Evidence에서는 강한 의견 충돌이 크게 나타나지 않았습니다.")

    if rca:
        aspect = ASPECT_LABELS.get(str(rca.get("aspect")), str(rca.get("aspect")))
        ctx = str(rca.get("context") or "특정 조건")
        effect = str(rca.get("effect") or "")
        if effect == "worsens":
            risks.append(f"{ctx} 조건에서 {aspect} 관련 부정 평가가 기준보다 더 자주 관찰됩니다.")
        else:
            why.append(f"{ctx} 조건에서는 {aspect} 평가가 상대적으로 나아지는 패턴이 관찰됩니다.")

    if wald:
        category, count = wald
        risks.append(
            f"일반 만족 리뷰 밖에서 '{WALD_LABELS.get(category, category)}' 신호가 {count}건 확인됩니다. "
            "이 수치는 실제 발생률이 아니라 놓치기 쉬운 이탈 신호입니다."
        )

    if confidence < 55:
        risks.append("현재 Evidence의 양·최신성·조건 일치도가 충분하지 않아 판단 신뢰도가 낮습니다.")

    if kind == "restaurant":
        if verdict == "GO":
            actions.append("현재 조건과 크게 충돌하는 신호가 적습니다. 예약 가능 여부만 확인하고 선택해도 됩니다.")
        elif verdict == "AVOID":
            actions.append("같은 목적이라면 대체 식당을 먼저 비교하는 편이 안전합니다.")
        else:
            actions.append("예약 가능 여부와 예상 웨이팅을 먼저 확인한 뒤 결정하는 것이 좋습니다.")
            if "소개팅" in str(context.get("purpose", "")) or "데이트" in str(context.get("purpose", "")):
                actions.append("대화가 중요한 일정이라면 혼잡 시간대와 좌석 간격 관련 후기부터 확인하세요.")
    else:
        if verdict == "GO":
            actions.append("현재 용도와 크게 충돌하는 반복 리스크가 적습니다. 가격 조건이 맞으면 구매 후보로 볼 수 있습니다.")
        elif verdict == "AVOID":
            actions.append("같은 예산대의 대체 제품을 비교한 뒤 구매하는 편이 안전합니다.")
        else:
            actions.append("구매 전에 본인의 핵심 조건과 관련된 단점 후기와 반품·불량 신호를 한 번 더 확인하세요.")
            if str(context.get("purpose") or ""):
                actions.append(f"특히 '{context.get('purpose')}' 용도에 직접 관련된 성능·편의성 Evidence를 우선 보세요.")

    headline_context = f" — {context_text} 기준" if context_text else ""
    headline = f"{label} · {score}/100{headline_context}"

    if verdict == "GO":
        summary = f"{target}은 현재 조건에서 선택 가능한 쪽에 가깝습니다."
    elif verdict == "AVOID":
        summary = f"{target}은 현재 조건에서는 리스크가 이점보다 더 크게 보입니다."
    else:
        summary = f"{target}은 무조건 추천하기보다 조건을 확인하고 선택해야 하는 대상입니다."

    return {
        "headline": headline,
        "summary": summary,
        "why": why[:3],
        "risks": risks[:3],
        "actions": actions[:3],
        "confidence_note": f"판단 신뢰도 {confidence}% · 적합도는 성공확률이 아니라 현재 Evidence 기준의 조건 적합도입니다.",
        "method_note": (
            "Dike's Eye는 보이는 리뷰의 의견 충돌(Rashomon)과 리뷰에 덜 남는 이탈 신호(Wald)를 함께 보고, "
            "사용자 조건과 맞는 Evidence에 더 높은 가중치를 줍니다."
        ),
    }
