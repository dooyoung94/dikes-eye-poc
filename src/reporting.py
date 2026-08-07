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
    return {"GO": "추천", "CONDITIONAL": "조건부 추천", "AVOID": "신중 추천"}.get(verdict, verdict)


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
    positive_aspects = decision.get("components", {}).get("positive_aspects", [])

    strengths: list[str] = []
    findings: list[str] = []
    risks: list[str] = []
    missing_side: list[str] = []
    limitations: list[str] = []
    recommendations: list[str] = []

    for item in positive_aspects[:3]:
        aspect = ASPECT_LABELS.get(str(item.get("aspect")), str(item.get("aspect")))
        pos = int(item.get("positive_count", 0))
        neg = int(item.get("negative_count", 0))
        rate = float(item.get("positive_rate", 0.0)) * 100
        strengths.append(
            f"{aspect}은 긍정 {pos}건 / 부정 {neg}건으로, 의견 Evidence 중 {rate:.0f}%가 긍정적이었습니다."
        )

    if not strengths:
        strengths.append("현재 Evidence에서는 반복적으로 확인되는 뚜렷한 긍정 강점이 아직 충분하지 않습니다.")

    if conflict:
        aspect = ASPECT_LABELS.get(str(conflict.get("aspect")), str(conflict.get("aspect")))
        positive_count = int(conflict.get("positive_count", 0))
        negative_count = int(conflict.get("negative_count", 0))
        opinion_count = int(conflict.get("opinion_count", positive_count + negative_count))
        positive_rate = float(conflict.get("positive_rate", 0.0)) * 100
        negative_rate = float(conflict.get("negative_rate", 0.0)) * 100
        findings.append(
            f"{aspect}은 총 {opinion_count}건 중 긍정 {positive_count}건({positive_rate:.0f}%), 부정 {negative_count}건({negative_rate:.0f}%)으로 의견이 갈렸습니다."
        )

    if rca:
        aspect = ASPECT_LABELS.get(str(rca.get("aspect")), str(rca.get("aspect")))
        ctx = str(rca.get("context") or "특정 조건")
        base_total = int(rca.get("baseline_total_count", 0))
        base_neg = int(rca.get("baseline_negative_count", 0))
        ctx_total = int(rca.get("context_total_count", 0))
        ctx_neg = int(rca.get("context_negative_count", 0))
        base_rate = float(rca.get("baseline_negative_rate", 0.0)) * 100
        ctx_rate = float(rca.get("context_negative_rate", 0.0)) * 100
        lift = float(rca.get("lift", 0.0)) * 100

        if str(rca.get("effect")) == "worsens":
            risks.append(
                f"'{ctx}'에서는 {aspect} 부정 의견이 {ctx_neg}/{ctx_total}건({ctx_rate:.0f}%)으로, 전체 {base_neg}/{base_total}건({base_rate:.0f}%)보다 {abs(lift):.1f}%p 높았습니다."
            )
        else:
            strengths.append(
                f"'{ctx}'에서는 {aspect} 부정 의견이 {ctx_neg}/{ctx_total}건({ctx_rate:.0f}%)으로, 전체 {base_neg}/{base_total}건({base_rate:.0f}%)보다 {abs(lift):.1f}%p 낮았습니다."
            )

    if wald:
        category, count = wald
        label_wald = WALD_LABELS.get(category, category)
        missing_side.append(f"일반 리뷰 밖에서 '{label_wald}' 관련 신호가 {count}건 확인됐습니다.")
        limitations.append(
            f"'{label_wald}' {count}건은 실제 발생률이 아니라 검색된 이탈 신호의 건수입니다. 전체 사용자 대비 비율로 해석하면 안 됩니다."
        )
    else:
        missing_side.append("현재 검색 범위에서는 강한 이탈 신호가 두드러지지 않았습니다.")

    if confidence < 45:
        limitations.append("Evidence의 양·최신성·조건 일치도가 낮아 결론은 참고 수준으로 보는 것이 좋습니다.")
    elif confidence < 60:
        limitations.append("판단 가능한 Evidence는 확보됐지만, 추가 후기나 최신 정보가 들어오면 결론이 달라질 수 있습니다.")

    if kind == "restaurant":
        if verdict == "GO":
            recommendations.append("현재 조건에서는 긍정 강점이 위험보다 우세합니다. 예약 가능 여부만 확인하고 선택해도 괜찮습니다.")
        elif verdict == "AVOID":
            recommendations.append("현재 조건에서는 불리한 신호가 꽤 강합니다. 같은 목적의 대체 식당을 한 곳 더 비교해보는 편이 좋습니다.")
        else:
            recommendations.append("좋은 점은 분명하지만 조건에 따라 체감이 달라질 수 있습니다. 예약과 혼잡도를 확인한 뒤 결정하는 것이 좋습니다.")
            if "소개팅" in str(context.get("purpose", "")) or "데이트" in str(context.get("purpose", "")):
                recommendations.append("대화가 중요하다면 소음·좌석 간격 관련 최신 후기를 우선 확인하세요.")
    else:
        if verdict == "GO":
            recommendations.append("현재 용도에서는 반복적으로 확인되는 장점이 더 큽니다. 가격과 보증 조건이 맞으면 구매 후보로 충분합니다.")
        elif verdict == "AVOID":
            recommendations.append("현재 사용 목적과 핵심 조건에서 불리한 신호가 강합니다. 같은 예산대의 대체 제품을 함께 비교해보세요.")
        else:
            recommendations.append("장점은 분명하지만 일부 단점이 용도와 맞물릴 수 있습니다. 핵심 조건에 해당하는 단점 후기만 한 번 더 확인하고 구매를 결정하세요.")

    headline_context = f" · {context_text} 기준" if context_text else ""
    headline = f"{label} · {score}/100{headline_context}"

    if verdict == "GO":
        conclusion = f"{target}은 지금 조건에서는 장점이 단점보다 더 크게 보입니다."
    elif verdict == "AVOID":
        conclusion = f"{target}은 장점도 있지만, 지금 조건에서는 핵심 위험이 더 크게 보입니다."
    else:
        conclusion = f"{target}은 좋은 점이 분명하지만, 몇 가지 조건을 확인한 뒤 선택하는 편이 좋습니다."

    return {
        "document_title": "Dike's View",
        "matter": f"{target} 선택 분석",
        "scope": context_text or "추가 조건 없음",
        "headline": headline,
        "summary": conclusion,
        "strengths": strengths[:4],
        "findings": findings[:3],
        "conflicting_evidence": risks[:3],
        "missing_side": missing_side[:2],
        "limitations": limitations[:3],
        "recommendations": recommendations[:3],
        "why": (strengths + findings)[:4],
        "risks": (risks + limitations)[:3],
        "actions": recommendations[:3],
        "confidence_note": f"판단 신뢰도 {confidence}% · 적합도 {score}/100",
        "method_note": "긍정 강점, 의견 충돌, 사용자 조건별 차이, 리뷰 밖 이탈 신호를 함께 평가했습니다.",
        "closing_note": "점수는 성공확률이나 절대 품질점수가 아니라 현재 Evidence와 사용자 조건 사이의 적합도입니다.",
    }
