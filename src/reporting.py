from __future__ import annotations

from typing import Any

from src.condition_taxonomy import CONDITION_RULES, aspect_label


ASPECT_LABELS = {
    aspect: str(rule.get("label") or aspect)
    for aspect, rule in CONDITION_RULES.items()
}
ASPECT_LABELS["other"] = "기타 평가"

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
    return {
        "GO": "추천",
        "CONDITIONAL": "조건부 추천",
        "AVOID": "신중 추천",
    }.get(verdict, verdict)


def _context_text(context: dict[str, Any]) -> str:
    values = [
        str(context.get("date_or_day") or "").strip(),
        str(context.get("time") or "").strip(),
        str(context.get("purpose") or "").strip(),
    ]
    return " · ".join(x for x in values if x)


def _top_wald(analysis: dict[str, Any]) -> tuple[str, int] | None:
    counts = analysis.get("wald", {}).get("signal_counts", {})
    if not counts:
        return None
    category, count = max(counts.items(), key=lambda x: x[1])
    return category, int(count)


def build_user_report(
    analysis: dict[str, Any],
    target: str,
    kind: str,
) -> dict[str, Any]:
    decision = analysis.get("decision", {})
    context = analysis.get("context", {})
    verdict = str(decision.get("verdict", "CONDITIONAL"))
    label = _verdict_label(verdict)
    score = float(decision.get("fit_score", 50))
    confidence = float(decision.get("confidence", 0))
    context_text = _context_text(context)
    condition_results = analysis.get("rca", {}).get("condition_results", [])
    conflicts = analysis.get("rca", {}).get("conflicts", [])
    wald = _top_wald(analysis)

    strengths: list[str] = []
    findings: list[str] = []
    risks: list[str] = []
    limitations: list[str] = []
    recommendations: list[str] = []

    for item in condition_results[:6]:
        label_cond = str(item.get("label") or aspect_label(str(item.get("aspect"))))
        total = int(item.get("total_count", 0))
        pos = int(item.get("positive_count", 0))
        neg = int(item.get("negative_count", 0))
        pos_rate = float(item.get("positive_rate", 0)) * 100
        fit = float(item.get("fit", 0.5)) * 100
        importance = float(item.get("importance", 0.8)) * 100
        direction = str(item.get("direction") or "prefer")

        if total < 3:
            limitations.append(
                f"{label_cond}은 관련 의견 Evidence가 {total}건이라 안정적인 조건 판단에는 부족합니다."
            )
            continue

        if direction == "avoid":
            risks.append(
                f"{label_cond}은 회피 조건으로 설정됐습니다. 관련 의견 {total}건 중 부정 {neg}건이며, 조건 적합도는 {fit:.0f}/100입니다."
            )
        elif direction == "tolerate":
            findings.append(
                f"{label_cond}은 어느 정도 허용 가능한 조건으로 해석했습니다. 관련 의견 {total}건 중 긍정 {pos}건({pos_rate:.0f}%)이고 중요도는 {importance:.0f}%입니다."
            )
        elif pos >= neg:
            strengths.append(
                f"{label_cond}은 관련 의견 {total}건 중 긍정 {pos}건({pos_rate:.0f}%)으로, 사용자가 중요하게 본 조건에서 비교적 우호적입니다."
            )
        else:
            risks.append(
                f"{label_cond}은 관련 의견 {total}건 중 부정 {neg}건이 긍정 {pos}건보다 많아, 사용자가 중요하게 본 조건에서 주의가 필요합니다."
            )

        situational_count = int(item.get("situational_count", 0))
        if situational_count >= 3:
            lift = float(item.get("situational_lift", 0)) * 100
            if lift > 3:
                risks.append(
                    f"{label_cond}은 현재 사용 상황과 맞는 {situational_count}건에서 부정 비율이 전체보다 {lift:.1f}%p 높았습니다."
                )
            elif lift < -3:
                strengths.append(
                    f"{label_cond}은 현재 사용 상황과 맞는 {situational_count}건에서 부정 비율이 전체보다 {abs(lift):.1f}%p 낮았습니다."
                )

    if conflicts:
        top = conflicts[0]
        label_conflict = aspect_label(str(top.get("aspect")))
        findings.append(
            f"{label_conflict}은 긍정 {int(top.get('positive_count', 0))}건 / 부정 {int(top.get('negative_count', 0))}건으로 의견이 갈렸습니다."
        )

    if wald:
        category, count = wald
        limitations.append(
            f"일반 리뷰 밖에서 {WALD_LABELS.get(category, category)} 신호가 {count}건 확인됐습니다. 이는 실제 발생률이 아니라 이탈 가능성을 보여주는 보조 신호입니다."
        )

    if confidence < 45:
        limitations.append("Evidence의 양·최신성·조건 coverage가 낮아 결론은 참고 수준으로 보는 것이 좋습니다.")
    elif confidence < 60:
        limitations.append("판단 가능한 Evidence는 확보됐지만, 최신 후기나 추가 데이터에 따라 결론이 달라질 수 있습니다.")

    if verdict == "GO":
        summary = f"{target}은 지금 입력한 조건에서는 선택해도 되는 쪽에 가깝습니다."
        recommendations.append("현재 조건에서 반복적으로 불리한 신호가 강하지 않습니다. 마지막으로 가격·예약·보증처럼 실시간 확인이 필요한 항목만 점검하세요.")
    elif verdict == "AVOID":
        summary = f"{target}은 장점도 있지만, 지금 입력한 조건에서는 우선순위를 낮추는 편이 좋습니다."
        recommendations.append("사용자가 중요하게 본 조건 중 불리한 축이 커서, 같은 목적의 대안을 하나 더 비교하는 편이 좋습니다.")
    else:
        summary = f"{target}은 선택 가능하지만, 지금 조건에서는 확인해야 할 지점이 남아 있습니다."
        recommendations.append("조건부로 선택할 수 있습니다. 점수가 낮아진 핵심 조건과 실제 사용 상황에서의 변화부터 확인한 뒤 결정하세요.")

    if kind == "restaurant" and ("소개팅" in str(context.get("purpose", "")) or "데이트" in str(context.get("purpose", ""))):
        recommendations.append("대화가 중요한 일정이라면 분위기·소음과 안락함 관련 최신 후기를 우선 확인하세요.")
    if kind == "product" and str(context.get("purpose") or ""):
        recommendations.append(f"특히 '{context.get('purpose')}' 용도에서 직접 체감되는 조건의 장기 사용 후기를 한 번 더 확인하세요.")

    headline_context = f" · {context_text}" if context_text else ""
    headline = f"{label} · {score:.1f}/100{headline_context}"

    return {
        "document_title": "Dike's Conditional View",
        "matter": f"{target} 선택 적합성 분석",
        "scope": context_text or "별도 상황조건 없음",
        "headline": headline,
        "summary": summary,
        "strengths": strengths[:5],
        "findings": findings[:4],
        "conflicting_evidence": risks[:5],
        "missing_side": limitations[:3],
        "limitations": limitations[:4],
        "recommendations": recommendations[:3],
        "why": (strengths + findings)[:5],
        "risks": (risks + limitations)[:5],
        "actions": recommendations[:3],
        "confidence_note": f"판단 신뢰도 {confidence:.0f}% · 조건 적합도 {score:.1f}/100",
        "method_note": "사용자가 말한 중요조건을 조건별로 분리하고, 각 조건의 전체 Evidence와 상황별 변화를 따로 계산했습니다.",
        "closing_note": "점수는 성공확률이나 절대 품질점수가 아니라 현재 공개 Evidence와 사용자가 제시한 조건 사이의 적합도입니다.",
    }
