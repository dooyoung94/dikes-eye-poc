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
    return {"GO": "권고", "CONDITIONAL": "조건부 권고", "AVOID": "권고 유보"}.get(verdict, verdict)


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

    findings: list[str] = []
    conflicting_evidence: list[str] = []
    missing_side: list[str] = []
    limitations: list[str] = []
    recommendations: list[str] = []

    if conflict:
        aspect = ASPECT_LABELS.get(str(conflict.get("aspect")), str(conflict.get("aspect")))
        positive_count = int(conflict.get("positive_count", 0))
        negative_count = int(conflict.get("negative_count", 0))
        findings.append(
            f"{aspect}에 관한 평가는 일관된 한 방향으로 수렴하지 않습니다. 확보된 의견 Evidence 중 "
            f"긍정 {positive_count}건과 부정 {negative_count}건이 함께 존재합니다."
        )
        conflicting_evidence.append(
            f"동일한 {aspect} 항목에 상반된 평가가 공존하므로 단순 평균만으로는 사용자의 실제 경험을 충분히 설명하기 어렵습니다."
        )
    else:
        findings.append(
            "현재 확보된 Evidence 범위에서는 특정 평가 항목에 대한 강한 의견 충돌이 뚜렷하게 확인되지 않았습니다."
        )

    if rca:
        aspect = ASPECT_LABELS.get(str(rca.get("aspect")), str(rca.get("aspect")))
        ctx = str(rca.get("context") or "특정 조건")
        effect = str(rca.get("effect") or "")
        lift = abs(float(rca.get("lift", 0.0)))
        if effect == "worsens":
            findings.append(
                f"'{ctx}' 조건에서는 {aspect} 관련 부정 평가 비중이 전체 기준보다 상대적으로 높게 관찰됩니다."
            )
            conflicting_evidence.append(
                f"해당 조건의 차이는 약 {lift * 100:.1f}%p 수준의 관찰 차이로 나타났습니다. 다만 이는 인과관계를 확정하는 통계적 추론이 아니라 현재 Evidence 내 연관성입니다."
            )
        else:
            findings.append(
                f"'{ctx}' 조건에서는 {aspect} 평가가 전체 기준보다 상대적으로 우호적인 방향으로 관찰됩니다."
            )

    if wald:
        category, count = wald
        label_wald = WALD_LABELS.get(category, category)
        missing_side.append(
            f"일반 만족 후기 바깥에서 '{label_wald}' 관련 신호가 {count}건 확인되었습니다. "
            "이는 선택 이전 또는 이탈 이후의 경험이 일반 리뷰에 충분히 포함되지 않을 가능성을 시사합니다."
        )
        limitations.append(
            f"'{label_wald}' {count}건은 전체 사용자 대비 실제 발생률이나 확률을 의미하지 않습니다. 분모가 없는 선택편향 위험 신호로만 해석해야 합니다."
        )
    else:
        missing_side.append(
            "현재 검색 범위에서는 일반 리뷰 바깥의 강한 이탈 신호가 두드러지게 확인되지 않았습니다. 다만 검색되지 않았다는 사실이 그러한 경험의 부재를 의미하지는 않습니다."
        )

    if confidence < 55:
        limitations.append(
            "Evidence의 양·최신성·출처 다양성 또는 사용자 조건 일치도가 충분하지 않아 본 판단의 신뢰도는 제한적입니다."
        )
    else:
        limitations.append(
            "본 판단은 현재 시점에 수집 가능한 공개 Evidence를 대상으로 하며, 이후 운영상태·가격·혼잡도·제품 품질 변화에 따라 달라질 수 있습니다."
        )

    if kind == "restaurant":
        if verdict == "GO":
            recommendations.append(
                "현재 조건과 직접 충돌하는 반복 위험은 제한적입니다. 예약 가능 여부와 당일 운영상태를 확인한 뒤 선택하는 것이 합리적입니다."
            )
        elif verdict == "AVOID":
            recommendations.append(
                "현재 목적과 사용자 조건을 기준으로 보면 불리한 Evidence가 우세하므로 동일 목적의 대체 식당을 우선 비교하는 편이 타당합니다."
            )
        else:
            recommendations.append(
                "전면적인 추천보다는 조건부 선택이 타당합니다. 우선 예약 가능 여부와 예상 웨이팅을 확인한 뒤 최종 결정하십시오."
            )
            if "소개팅" in str(context.get("purpose", "")) or "데이트" in str(context.get("purpose", "")):
                recommendations.append(
                    "대화의 안정성이 중요한 일정이라면 혼잡 시간대, 좌석 간격, 소음 관련 최신 후기의 비중을 우선 확인하는 것이 좋습니다."
                )
    else:
        if verdict == "GO":
            recommendations.append(
                "현재 용도와 직접 충돌하는 반복 위험이 제한적이므로 가격·보증 조건이 합리적이라면 구매 후보로 유지할 수 있습니다."
            )
        elif verdict == "AVOID":
            recommendations.append(
                "현재 사용 목적과 핵심 조건에서 불리한 Evidence가 상대적으로 커 보이므로 동일 예산대의 대체 제품을 함께 비교하는 것이 타당합니다."
            )
        else:
            recommendations.append(
                "구매를 즉시 확정하기보다는 핵심 조건과 직접 관련된 단점 후기, 반품·불량·장기 사용 Evidence를 추가 확인한 뒤 결정하는 것이 좋습니다."
            )
            if str(context.get("purpose") or ""):
                recommendations.append(
                    f"특히 '{context.get('purpose')}' 용도에 직접 연결되는 성능·편의성·내구성 Evidence에 우선순위를 두십시오."
                )

    headline_context = f" · {context_text} 기준" if context_text else ""
    headline = f"{label} 의견 · 적합도 {score}/100{headline_context}"

    if verdict == "GO":
        conclusion = (
            f"현재 확보된 Evidence와 사용 조건을 종합하면, {target}은 선택 가능한 쪽에 가깝다는 의견입니다. "
            "다만 이는 절대적 우수성이나 성공확률을 의미하지 않으며, 현재 조건과의 상대적 적합성 판단입니다."
        )
    elif verdict == "AVOID":
        conclusion = (
            f"현재 확보된 Evidence와 사용 조건을 종합하면, {target}을 우선 선택하는 데에는 신중함이 필요하다는 의견입니다. "
            "불리한 조건이 사용자의 핵심 목적과 직접 맞물릴 가능성이 상대적으로 높게 관찰됩니다."
        )
    else:
        conclusion = (
            f"현재 확보된 Evidence만으로 {target}을 전면적으로 권고하기는 어렵습니다. "
            "다만 특정 조건을 사전에 확인하거나 회피할 수 있다면 선택 가능한 범위에 있다는 조건부 의견입니다."
        )

    matter = f"{target} 선택 적합성 검토"
    scope = context_text or "사용자가 별도로 지정한 세부 조건 없음"

    return {
        "document_title": "DIKE 검토 의견서",
        "matter": matter,
        "scope": scope,
        "headline": headline,
        "summary": conclusion,
        "findings": findings[:4],
        "conflicting_evidence": conflicting_evidence[:3],
        "missing_side": missing_side[:2],
        "limitations": limitations[:3],
        "recommendations": recommendations[:3],
        # 기존 UI 호환 필드
        "why": findings[:3],
        "risks": (conflicting_evidence + limitations)[:3],
        "actions": recommendations[:3],
        "confidence_note": (
            f"판단 신뢰도 {confidence}% · 적합도 {score}/100. 적합도는 성공확률이나 객관적 품질점수가 아니라 "
            "현재 확보된 Evidence와 사용자 조건 사이의 상대적 적합성 지표입니다."
        ),
        "method_note": (
            "검토 방식: 보이는 후기의 상반된 의견(Rashomon), 조건별 관찰 차이(RCA), 리뷰에 잘 남지 않는 이탈 신호(Wald), "
            "최신성·반복성·사용자 조건 일치도(RFM)를 함께 고려했습니다."
        ),
        "closing_note": (
            "본 의견은 공개 Evidence를 구조화하여 선택을 보조하기 위한 참고 의견이며, 사실의 완전성이나 미래 결과를 보장하지 않습니다."
        ),
    }
