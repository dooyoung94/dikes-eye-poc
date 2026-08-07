from __future__ import annotations

import json
from typing import Any

try:
    from openai import OpenAI
except (ImportError, ModuleNotFoundError):
    OpenAI = None


def fallback_explanation(analysis: dict[str, Any]) -> dict[str, Any]:
    decision = analysis.get("decision", {})
    verdict = decision.get("verdict", "CONDITIONAL")
    score = decision.get("fit_score", 50)
    confidence = decision.get("confidence", 0)
    rca_risk = decision.get("components", {}).get("rca_risk", 0)
    wald_risk = decision.get("components", {}).get("wald_risk", 0)
    kind = analysis.get("kind", "restaurant")
    target = analysis.get("target", "대상")

    label = {
        "GO": "추천",
        "CONDITIONAL": "조건부 추천",
        "AVOID": "비추천",
    }.get(verdict, verdict)

    reasons = []
    if confidence < 55:
        reasons.append("Evidence의 양·최신성·사용자 조건 일치도가 충분하지 않아 판단 신뢰도가 제한적입니다.")
    else:
        reasons.append("Evidence의 양과 최신성, 사용자 조건 일치도를 함께 반영했습니다.")

    if rca_risk > 0.35:
        reasons.append("사용자 조건과 맞물리는 부정적 RCA 신호가 일부 확인됩니다.")
    else:
        reasons.append("사용자 조건과 직접 맞물리는 강한 RCA 위험은 제한적으로 확인됩니다.")

    if wald_risk > 0.35:
        if kind == "product":
            reasons.append("반품·환불·고장·재판매 같은 일반 만족 리뷰 밖의 이탈 신호도 함께 고려했습니다.")
        else:
            reasons.append("예약 실패·웨이팅 포기 같은 일반 방문 리뷰 밖의 이탈 신호도 함께 고려했습니다.")
    else:
        reasons.append("Wald 관점의 이탈 신호는 현재 Evidence에서 강하게 나타나지 않았습니다.")

    if kind == "product":
        risks = [
            "Wald 신호는 실제 반품률이나 불량률이 아니라 놓치기 쉬운 이탈 Evidence입니다.",
            "RCA는 사용조건별 관찰 연관성이며 인과관계를 확정하지 않습니다.",
        ]
    else:
        risks = [
            "Wald 신호는 실제 예약 실패율이나 방문 포기율이 아니라 선택편향 위험 신호입니다.",
            "RCA는 조건별 관찰 연관성이며 인과관계를 확정하지 않습니다.",
        ]

    return {
        "headline": f"{label} · {target} 적합도 {score}/100",
        "answer": (
            f"현재 입력한 조건 기준으로는 {label} 판단입니다. 판단 신뢰도는 {confidence}%이며, "
            "점수는 LLM이 아니라 deterministic scoring engine이 계산했습니다."
        ),
        "reasons": reasons[:3],
        "risks": risks,
        "source": "fallback",
    }


def generate_explanation(
    analysis: dict[str, Any],
    *,
    api_key: str = "",
    model: str = "gpt-5-mini",
) -> dict[str, Any]:
    if not api_key or OpenAI is None:
        return fallback_explanation(analysis)

    payload = {
        "kind": analysis.get("kind", "restaurant"),
        "target": analysis.get("target", ""),
        "context": analysis.get("context", {}),
        "decision": analysis.get("decision", {}),
        "eda": analysis.get("eda", {}),
        "rfm": analysis.get("rfm", {}),
        "rashomon": analysis.get("rashomon", {}),
        "rca": {
            "aligned_risk": analysis.get("rca", {}).get("aligned_risk", 0),
            "cause_candidates": analysis.get("rca", {}).get("cause_candidates", [])[:5],
            "interpretation": analysis.get("rca", {}).get("interpretation", ""),
        },
        "wald": analysis.get("wald", {}),
    }

    instruction = """
너는 Dike's Eye의 설명 레이어다.
절대로 verdict, fit_score, confidence, score components를 재계산하거나 변경하지 마라.
주어진 deterministic 분석 결과만 설명하라.
대상이 restaurant인지 product인지 반드시 구분해 그 도메인에 맞는 표현을 사용하라.
RCA는 인과 확정이 아니라 관찰된 조건별 연관성으로 표현하라.
Wald는 실제 실패율/반품률/불량률이 아니라 보이지 않는 이탈·선택편향 위험 신호로 표현하라.
사용자가 입력한 목적과 중요한 조건을 결론의 중심에 놓아라.
기술용어를 남발하지 말고 실제 선택에 도움이 되는 한국어로 간결하게 답하라.
반드시 아래 JSON 형식으로만 출력하라.
{
  "headline": "한 줄 결론",
  "answer": "2~4문장 설명",
  "reasons": ["근거1", "근거2", "근거3"],
  "risks": ["주의점1", "주의점2"]
}
""".strip()

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            return fallback_explanation(analysis)

        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].lstrip()

        parsed = json.loads(text)
        result = {
            "headline": str(parsed.get("headline", "")),
            "answer": str(parsed.get("answer", "")),
            "reasons": [str(x) for x in parsed.get("reasons", [])][:3],
            "risks": [str(x) for x in parsed.get("risks", [])][:3],
            "source": "openai",
        }
        if not result["headline"] or not result["answer"]:
            return fallback_explanation(analysis)
        return result
    except Exception:
        return fallback_explanation(analysis)
