from __future__ import annotations

from typing import Any

from src.condition_taxonomy import CONDITION_RULES, aspect_label


ASPECT_LABELS = {aspect: str(rule.get("label") or aspect) for aspect, rule in CONDITION_RULES.items()}
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
    return {"GO": "추천", "CONDITIONAL": "조건부 추천", "AVOID": "대안 비교 권고"}.get(verdict, verdict)


def _context_text(context: dict[str, Any]) -> str:
    values = [
        str(context.get("date_or_day") or "").strip(),
        str(context.get("time") or "").strip(),
        str(context.get("purpose") or "").strip(),
    ]
    return " · ".join(x for x in values if x)


def _condition_sentence(item: dict[str, Any]) -> str:
    label = str(item.get("label") or aspect_label(str(item.get("aspect"))))
    total = int(item.get("total_count", 0))
    pos = int(item.get("positive_count", 0))
    neg = int(item.get("negative_count", 0))
    fit = float(item.get("fit", 0.5)) * 100
    direction = str(item.get("direction") or "prefer")
    if total < 3:
        return f"{label}은 관련 의견 {total}건으로 근거가 아직 충분하지 않습니다."
    if direction == "avoid":
        return f"{label}은 피하고 싶은 조건인데, 부정 {neg}건 / 긍정 {pos}건으로 조건 적합도는 {fit:.0f}/100입니다."
    if direction == "tolerate":
        return f"{label}은 어느 정도 허용 가능한 조건이며, 긍정 {pos}건 / 부정 {neg}건으로 조건 적합도는 {fit:.0f}/100입니다."
    return f"{label}은 긍정 {pos}건 / 부정 {neg}건으로 조건 적합도 {fit:.0f}/100입니다."


def build_user_report(analysis: dict[str, Any], target: str, kind: str) -> dict[str, Any]:
    decision = analysis.get("decision", {})
    context = analysis.get("context", {})
    verdict = str(decision.get("verdict", "CONDITIONAL"))
    label = _verdict_label(verdict)
    score = float(decision.get("fit_score", 50))
    confidence = float(decision.get("confidence", 0))
    context_text = _context_text(context)
    condition_results = analysis.get("rca", {}).get("condition_results", [])
    conflicts = analysis.get("conflict_insights", [])
    consensus = analysis.get("consensus", {})
    wald_counts = analysis.get("wald", {}).get("signal_counts", {})

    consensus_score = float(decision.get("components", {}).get("consensus_score", consensus.get("opinion_score", 50)))
    condition_score = float(decision.get("components", {}).get("condition_score", 50))
    consensus_n = int(consensus.get("sample_count", 0))
    consensus_pos = int(consensus.get("positive_count", 0))
    consensus_neg = int(consensus.get("negative_count", 0))

    strengths: list[str] = []
    findings: list[str] = []
    risks: list[str] = []
    limitations: list[str] = []
    recommendations: list[str] = []

    if consensus_n:
        findings.append(
            f"일반 후기 {consensus_n}건에서는 긍정 {consensus_pos}건 / 부정 {consensus_neg}건으로 전체 여론 점수는 {consensus_score:.0f}/100입니다."
        )

    for item in condition_results[:6]:
        sentence = _condition_sentence(item)
        fit = float(item.get("fit", 0.5))
        direction = str(item.get("direction") or "prefer")
        if int(item.get("total_count", 0)) < 3:
            limitations.append(sentence)
        elif (direction == "avoid" and fit < 0.55) or (direction != "avoid" and fit < 0.50):
            risks.append(sentence)
        else:
            strengths.append(sentence)

        situ_count = int(item.get("situational_count", 0))
        lift = float(item.get("situational_lift", 0)) * 100
        if situ_count >= 3:
            situ_neg = int(item.get("situational_negative_count", 0))
            if lift > 3:
                risks.append(
                    f"{item.get('label')}은 현재 사용 상황과 맞는 {situ_count}건에서 부정 {situ_neg}건으로, 전체보다 부정 비율이 {lift:.1f}%p 높았습니다."
                )
            elif lift < -3:
                strengths.append(
                    f"{item.get('label')}은 현재 사용 상황과 맞는 {situ_count}건에서 전체보다 부정 비율이 {abs(lift):.1f}%p 낮았습니다."
                )

    for conflict in conflicts[:2]:
        label_conflict = str(conflict.get("label") or "쟁점")
        pos = int(conflict.get("positive_count", 0))
        neg = int(conflict.get("negative_count", 0))
        pos_kw = ", ".join(str(x.get("keyword")) for x in conflict.get("positive_keywords", [])[:3])
        neg_kw = ", ".join(str(x.get("keyword")) for x in conflict.get("negative_keywords", [])[:3])
        findings.append(
            f"{label_conflict}은 긍정 {pos}건 vs 부정 {neg}건으로 의견이 갈렸습니다. 긍정 쪽은 '{pos_kw or '반복 키워드 부족'}', 부정 쪽은 '{neg_kw or '반복 키워드 부족'}'가 자주 나타났습니다."
        )

    if wald_counts:
        category, count = max(wald_counts.items(), key=lambda x: x[1])
        limitations.append(
            f"일반 리뷰 밖에서 {WALD_LABELS.get(category, category)} 신호가 {int(count)}건 확인됐습니다. 실제 발생률이 아니라 놓치기 쉬운 이탈 신호입니다."
        )

    delta = condition_score - consensus_score
    if verdict == "GO":
        summary = f"{target}은 전체 평가와 당신의 조건을 함께 놓고 보면 선택 쪽으로 기웁니다."
        recommendations.append(
            f"전체 여론은 {consensus_score:.0f}/100, 당신의 조건 적합은 {condition_score:.0f}/100입니다. 두 축이 모두 크게 충돌하지 않아 선택을 권합니다."
        )
        recommendations.append("다만 예약·가격·재고처럼 실시간으로 변하는 항목은 결정 직전에 한 번 더 확인하세요.")
    elif verdict == "AVOID":
        summary = f"{target} 자체의 장점과 별개로, 이번 조건에서는 다른 대안을 함께 보는 편이 낫습니다."
        recommendations.append(
            f"전체 여론은 {consensus_score:.0f}/100이지만 당신의 조건 적합은 {condition_score:.0f}/100입니다. 이번 선택에서는 평균보다 개인 조건을 더 우선하는 것이 타당합니다."
        )
        recommendations.append("점수가 낮은 조건과 같은 목적의 대체 후보 1~2개를 비교한 뒤 결정하세요.")
    else:
        summary = f"{target}은 나쁜 선택은 아니지만, 당신의 조건에 따라 결론이 달라지는 대상입니다."
        if delta <= -7:
            recommendations.append(
                f"전체 여론은 {consensus_score:.0f}/100으로 더 좋지만, 당신의 조건에서는 {condition_score:.0f}/100으로 {abs(delta):.0f}점 낮습니다. 평균 평판만 보고 선택하지 않는 편이 좋습니다."
            )
        elif delta >= 7:
            recommendations.append(
                f"전체 여론은 {consensus_score:.0f}/100이지만 당신의 조건에서는 {condition_score:.0f}/100으로 {delta:.0f}점 더 유리합니다. 평균보다 당신에게 더 맞는 선택일 수 있습니다."
            )
        else:
            recommendations.append(
                f"전체 여론 {consensus_score:.0f}/100과 당신의 조건 {condition_score:.0f}/100이 비슷한 방향입니다. 다만 의견이 갈린 쟁점을 확인한 뒤 조건부로 선택하세요."
            )
        low = sorted(
            [x for x in condition_results if int(x.get("total_count", 0)) >= 3],
            key=lambda x: float(x.get("fit", 0.5)),
        )[:2]
        if low:
            labels = ", ".join(str(x.get("label")) for x in low)
            recommendations.append(f"최종 결정 전에는 특히 {labels} 관련 최신 Evidence를 확인하세요.")

    if confidence < 50:
        limitations.append("판단 신뢰도가 낮아 현재 결론은 강한 판정보다 참고 의견에 가깝습니다.")

    headline_context = f" · {context_text}" if context_text else ""
    headline = f"{label} · {score:.1f}/100{headline_context}"

    return {
        "document_title": "Dike's Judgment",
        "matter": f"{target} 조건부 선택 판정",
        "scope": context_text or "별도 상황조건 없음",
        "headline": headline,
        "summary": summary,
        "strengths": strengths[:6],
        "findings": findings[:6],
        "conflicting_evidence": risks[:6],
        "missing_side": limitations[:4],
        "limitations": limitations[:5],
        "recommendations": recommendations[:4],
        "why": (findings + strengths)[:7],
        "risks": (risks + limitations)[:7],
        "actions": recommendations[:4],
        "confidence_note": f"판단 신뢰도 {confidence:.0f}% · 최종 적합도 {score:.1f}/100",
        "method_note": "전체 일반 후기 여론, 사용자 중요조건, 상황별 변화, 의견 충돌, 리뷰 밖 이탈 신호를 각각 계산한 뒤 함께 저울질했습니다.",
        "closing_note": "전체 여론은 공개 검색으로 확보한 일반 후기 표본의 방향이며 모집단 전체를 대표하지 않습니다. 최종 점수 역시 성공확률이 아니라 조건 적합도입니다.",
    }
