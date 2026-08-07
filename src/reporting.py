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
    conflicts = analysis.get("rca", {}).get("conflicts", [])
    return conflicts[0] if conflicts else None


def _top_rca(analysis: dict[str, Any]) -> dict[str, Any] | None:
    candidates = analysis.get("rca", {}).get("cause_candidates", [])
    aligned = [x for x in candidates if x.get("user_aligned")]
    return (aligned or candidates or [None])[0]


def _wald_signals(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    counts = analysis.get("wald", {}).get("signal_counts", {})
    rows = [
        {"key": key, "label": WALD_LABELS.get(key, key), "count": int(count)}
        for key, count in counts.items()
        if int(count) > 0
    ]
    return sorted(rows, key=lambda x: x["count"], reverse=True)


def build_user_report(analysis: dict[str, Any], target: str, kind: str) -> dict[str, Any]:
    decision = analysis.get("decision", {})
    context = analysis.get("context", {})
    verdict = str(decision.get("verdict", "CONDITIONAL"))
    verdict_label = _verdict_label(verdict)
    score = float(decision.get("fit_score", 50))
    confidence = float(decision.get("confidence", 0))
    context_text = _context_text(context)

    visible_count = len(analysis.get("rows", []))
    hidden_count = len(analysis.get("hidden_rows", []))
    total_count = visible_count + hidden_count

    conflict = _top_conflict(analysis)
    rca = _top_rca(analysis)
    wald_signals = _wald_signals(analysis)

    data_points: list[dict[str, Any]] = []
    reasons: list[str] = []
    cautions: list[str] = []
    actions: list[str] = []

    if conflict:
        aspect = ASPECT_LABELS.get(str(conflict.get("aspect")), str(conflict.get("aspect")))
        pos = int(conflict.get("positive_count", 0))
        neg = int(conflict.get("negative_count", 0))
        opinion_total = max(1, int(conflict.get("opinion_count", pos + neg)))
        pos_rate = round(pos / opinion_total * 100)
        neg_rate = round(neg / opinion_total * 100)
        data_points.append({
            "type": "conflict",
            "label": aspect,
            "positive_count": pos,
            "negative_count": neg,
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,
            "total": opinion_total,
        })
        reasons.append(
            f"{aspect} 관련 의견 {opinion_total}건 중 긍정 {pos}건({pos_rate}%), 부정 {neg}건({neg_rate}%)으로 평가가 갈렸어요."
        )

    if rca:
        aspect = ASPECT_LABELS.get(str(rca.get("aspect")), str(rca.get("aspect")))
        ctx = str(rca.get("context") or "특정 조건")
        base_total = int(rca.get("baseline_total_count", 0))
        base_neg = int(rca.get("baseline_negative_count", 0))
        ctx_total = int(rca.get("context_total_count", 0))
        ctx_neg = int(rca.get("context_negative_count", 0))
        base_rate = round(float(rca.get("baseline_negative_rate", 0)) * 100)
        ctx_rate = round(float(rca.get("context_negative_rate", 0)) * 100)
        diff = round((float(rca.get("context_negative_rate", 0)) - float(rca.get("baseline_negative_rate", 0))) * 100, 1)
        data_points.append({
            "type": "context",
            "label": f"{ctx} · {aspect}",
            "baseline_total": base_total,
            "baseline_negative": base_neg,
            "baseline_rate": base_rate,
            "context_total": ctx_total,
            "context_negative": ctx_neg,
            "context_rate": ctx_rate,
            "difference_pp": diff,
            "effect": rca.get("effect", ""),
        })
        direction = "높았어요" if diff > 0 else "낮았어요"
        reasons.append(
            f"{ctx} 조건에서는 {aspect} 부정 의견이 {ctx_neg}/{ctx_total}건({ctx_rate}%)으로, 전체 {base_neg}/{base_total}건({base_rate}%)보다 {abs(diff):.1f}%p {direction}."
        )
        cautions.append("이 차이는 현재 수집된 후기 안에서 관찰된 패턴이며, 원인 자체를 확정하는 값은 아니에요.")

    if wald_signals:
        for signal in wald_signals[:3]:
            data_points.append({"type": "wald", **signal})
        top = wald_signals[0]
        cautions.append(
            f"일반 만족 후기 밖에서 '{top['label']}' 신호가 {top['count']}건 확인됐어요. 리뷰만 볼 때 놓치기 쉬운 경험이라 별도로 반영했어요."
        )
        cautions.append("이 건수는 실제 발생률이 아니라 검색된 위험 신호의 수예요. 전체 이용자 대비 비율로 해석하면 안 됩니다.")

    if confidence < 55:
        cautions.append(
            f"판단 신뢰도는 {confidence:.0f}%예요. Evidence가 더 쌓이면 결론이 달라질 여지가 있습니다."
        )

    if kind == "restaurant":
        if verdict == "GO":
            actions.append("현재 조건과 크게 충돌하는 반복 신호는 적어요. 예약 가능 여부만 확인하고 선택해도 괜찮습니다.")
        elif verdict == "AVOID":
            actions.append("현재 목적과 맞물리는 부정 신호가 커 보여요. 같은 목적의 다른 식당을 한 곳 이상 같이 비교해보는 편이 좋습니다.")
        else:
            actions.append("예약 가능 여부와 예상 웨이팅을 먼저 확인한 뒤 결정하세요.")
            if "소개팅" in str(context.get("purpose", "")) or "데이트" in str(context.get("purpose", "")):
                actions.append("대화가 중요한 일정이라면 소음·좌석 간격 관련 최신 후기부터 확인하세요.")
    else:
        if verdict == "GO":
            actions.append("현재 용도와 크게 충돌하는 반복 단점은 적어요. 가격과 보증 조건이 맞으면 구매 후보로 볼 수 있습니다.")
        elif verdict == "AVOID":
            actions.append("핵심 사용 조건에서 불리한 Evidence가 커 보여요. 같은 예산대 대체 제품을 같이 비교해보세요.")
        else:
            actions.append("구매 전 핵심 조건과 직접 관련된 단점 후기와 반품·불량 신호를 한 번 더 확인하세요.")

    if verdict == "GO":
        summary = f"{target}은 현재 조건에서는 꽤 잘 맞는 선택으로 보여요."
    elif verdict == "AVOID":
        summary = f"{target}은 지금 조건에서는 바로 선택하기보다 한 번 더 비교해보는 편이 좋아 보여요."
    else:
        summary = f"{target}은 조건만 잘 확인하면 선택할 수 있지만, 무조건 추천하기는 어려워요."

    return {
        "document_title": "Dike's View",
        "matter": target,
        "scope": context_text or "추가 조건 없음",
        "verdict_label": verdict_label,
        "headline": f"{verdict_label} · {score:.0f}/100",
        "summary": summary,
        "visible_count": visible_count,
        "hidden_count": hidden_count,
        "total_count": total_count,
        "score": score,
        "confidence": confidence,
        "data_points": data_points,
        "reasons": reasons[:4],
        "cautions": cautions[:4],
        "actions": actions[:3],
        "why": reasons[:3],
        "risks": cautions[:3],
        "recommendations": actions[:3],
        "confidence_note": f"총 {total_count}건의 Evidence를 검토했고, 현재 판단 신뢰도는 {confidence:.0f}%입니다.",
        "method_note": "최신성(R)·반복성(F)·내 조건과의 일치도(M)를 반영하고, 상반된 의견과 리뷰 밖 이탈 신호를 함께 비교했습니다.",
    }
