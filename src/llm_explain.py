from __future__ import annotations

import json
from typing import Any

try:
    from openai import OpenAI
except (ImportError, ModuleNotFoundError):
    OpenAI = None


def fallback_explanation(analysis: dict[str, Any]) -> dict[str, Any]:
    decision = analysis.get("decision", {})
    verdict = str(decision.get("verdict", "CONDITIONAL"))
    score = float(decision.get("fit_score", 50))
    confidence = float(decision.get("confidence", 0))
    conditions = analysis.get("rca", {}).get("condition_results", [])

    label = {"GO": "추천", "CONDITIONAL": "조건부 추천", "AVOID": "신중 추천"}.get(verdict, verdict)
    reasons: list[str] = []
    for item in conditions[:3]:
        total = int(item.get("total_count", 0))
        pos = int(item.get("positive_count", 0))
        neg = int(item.get("negative_count", 0))
        label_cond = str(item.get("label") or item.get("aspect") or "조건")
        if total >= 3:
            reasons.append(f"{label_cond}: 관련 의견 {total}건 중 긍정 {pos}건, 부정 {neg}건입니다.")
        else:
            reasons.append(f"{label_cond}: 관련 의견이 {total}건이라 이 조건은 판단 근거가 아직 부족합니다.")

    if not reasons:
        reasons.append("별도 중요조건이 충분히 구조화되지 않아 전체 Evidence 중심으로 판단했습니다.")

    answer = (
        f"현재 결과는 {label}이고 조건 적합도는 {score:.1f}/100, 판단 신뢰도는 {confidence:.0f}%입니다. "
        "중요하게 말한 조건은 각각 따로 계산했고, 전용 검색 결과가 적더라도 해당 조건과 연결되는 전체 Evidence를 함께 사용했습니다. "
        "요일·시간·목적에 맞는 Evidence가 충분한 경우에는 같은 조건이 실제 사용 상황에서 더 좋아지는지 또는 나빠지는지도 별도로 비교했습니다."
    )
    return {
        "headline": f"Dike의 조건부 판단 · {label}",
        "answer": answer,
        "reasons": reasons,
        "risks": [
            "조건별 건수는 공개 검색 Evidence의 범위이며 전체 이용자의 실제 발생률을 뜻하지 않습니다.",
            "상황별 차이는 관찰된 연관성으로만 해석하며 인과관계를 확정하지 않습니다.",
        ],
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
        "condition_results": analysis.get("rca", {}).get("condition_results", [])[:6],
        "situational_changes": analysis.get("rca", {}).get("main_candidates", [])[:6],
        "rashomon": analysis.get("rashomon", {}),
        "wald": analysis.get("wald", {}),
    }

    instruction = """
너는 Dike's Eye Conditional Decision Agent의 설명 보조인이다.
말투는 차분하고 정리된 리서치 보조인처럼 쓰되, 지나치게 법률 문서처럼 딱딱하게 쓰지 않는다.
사용자가 바로 이해할 수 있도록 숫자를 먼저 제시하고 그 의미를 짧게 설명한다.

절대 규칙:
1. verdict, fit_score, confidence, Evidence 건수·비율을 다시 계산하거나 바꾸지 않는다.
2. 각 중요조건은 condition_results의 total_count / positive_count / negative_count를 근거로 설명한다.
3. direct_count는 '조건 전용 검색/직접 표현 근거'일 뿐, 전체 조건 Evidence의 유무를 결정하는 기준이 아니다.
4. situational_changes는 요일·시간·목적에 따른 관찰 차이이며 인과관계라고 말하지 않는다.
5. Wald는 실제 실패율이 아니라 리뷰 밖 이탈 신호의 검색 건수라고 명시한다.
6. Evidence가 부족한 조건은 부족하다고 말하되 다른 조건으로 대신 판단하지 않는다.
7. 사용자가 '비싸도 괜찮다'처럼 tolerate로 말한 조건은 강한 감점요인처럼 설명하지 않는다.
8. 사용자가 '웨이팅은 싫다'처럼 avoid로 말한 조건은 중요 회피조건으로 설명한다.

반드시 JSON만 반환한다.
{
  "headline": "한 줄 결론",
  "answer": "3~6문장 설명",
  "reasons": ["조건별 숫자 근거1", "조건별 숫자 근거2", "상황 또는 Wald 근거3"],
  "risks": ["판단 한계1", "결정 전 확인사항2"]
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
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].lstrip()
        parsed = json.loads(text)
        result = {
            "headline": str(parsed.get("headline", "")),
            "answer": str(parsed.get("answer", "")),
            "reasons": [str(x) for x in parsed.get("reasons", [])][:4],
            "risks": [str(x) for x in parsed.get("risks", [])][:3],
            "source": "openai",
        }
        if not result["headline"] or not result["answer"]:
            return fallback_explanation(analysis)
        return result
    except Exception:
        return fallback_explanation(analysis)
