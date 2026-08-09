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
    consensus = analysis.get("consensus", {})
    conditions = analysis.get("rca", {}).get("condition_results", [])
    conflicts = analysis.get("conflict_insights", [])

    label = {"GO": "추천", "CONDITIONAL": "조건부 추천", "AVOID": "대안 비교 권고"}.get(verdict, verdict)
    reasons: list[str] = []

    sample_n = int(consensus.get("sample_count", 0))
    if sample_n:
        reasons.append(
            f"전체 여론: 일반 후기 {sample_n}건 중 긍정 {int(consensus.get('positive_count', 0))}건, 부정 {int(consensus.get('negative_count', 0))}건입니다."
        )

    for item in conditions[:3]:
        total = int(item.get("total_count", 0))
        pos = int(item.get("positive_count", 0))
        neg = int(item.get("negative_count", 0))
        label_cond = str(item.get("label") or item.get("aspect") or "조건")
        if total >= 3:
            reasons.append(f"{label_cond}: 관련 의견 {total}건 중 긍정 {pos}건, 부정 {neg}건입니다.")

    if conflicts:
        c = conflicts[0]
        pos_kw = ", ".join(str(x.get("keyword")) for x in c.get("positive_keywords", [])[:2])
        neg_kw = ", ".join(str(x.get("keyword")) for x in c.get("negative_keywords", [])[:2])
        reasons.append(
            f"쟁점: {c.get('label')}은 긍정 {c.get('positive_count', 0)}건 vs 부정 {c.get('negative_count', 0)}건이며, 긍정 쪽은 '{pos_kw}', 부정 쪽은 '{neg_kw}'가 반복됐습니다."
        )

    answer = (
        f"현재 최종 판정은 {label}, 적합도 {score:.1f}/100, 판단 신뢰도 {confidence:.0f}%입니다. "
        "전체 일반 후기의 방향과 사용자가 중요하게 말한 조건을 따로 계산한 뒤 함께 비교했습니다. "
        "의견이 갈리는 항목은 긍정·부정 키워드와 대표 Evidence를 나눠 확인했고, 요일·시간·목적에 맞는 표본이 충분하면 그 상황에서의 변화도 별도로 반영했습니다."
    )
    return {
        "headline": f"Dike의 판정 · {label}",
        "answer": answer,
        "reasons": reasons[:5],
        "risks": [
            "전체 여론은 공개 검색으로 확보한 일반 후기 표본의 방향이며 모집단 전체를 대표하지 않습니다.",
            "상황별 차이는 관찰된 연관성으로만 해석하고, Wald 신호도 실제 발생률로 해석하지 않습니다.",
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
        "consensus": analysis.get("consensus", {}),
        "condition_results": analysis.get("rca", {}).get("condition_results", [])[:6],
        "situational_changes": analysis.get("rca", {}).get("main_candidates", [])[:6],
        "conflict_insights": analysis.get("conflict_insights", [])[:3],
        "condition_evidence": analysis.get("condition_evidence", [])[:6],
        "wald": analysis.get("wald", {}),
    }

    instruction = """
너는 Dike's Eye의 판정 설명 보조인이다.
말투는 차분하고 이해하기 쉬운 리서치 보조인처럼 작성한다. 지나치게 법률 문서처럼 딱딱하게 쓰지 않는다.

설명 순서는 반드시 다음을 따른다.
1. 전체 여론: 일반 후기 표본의 긍정/부정 건수와 방향
2. 사용자 조건: 각 조건의 긍정/부정 건수, 조건 적합도, 중요도
3. 의견 충돌: 긍정과 부정이 갈리는 핵심 키워드와 대표 Evidence 차이
4. 상황 변화: 요일·시간·목적에서 전체와 달라진 비율이 있으면 설명
5. 숨은 위험: Wald 신호 건수
6. 최종 판정: 전체 여론과 내 조건이 같은 방향인지, 충돌하는지 설명하고 행동 권고로 끝낸다.

절대 규칙:
- verdict, fit_score, confidence, Evidence 건수·비율을 다시 계산하거나 바꾸지 않는다.
- 전체 여론은 모집단 전체의 통계가 아니라 현재 공개 검색에서 확보한 일반 후기 표본이라고 명시한다.
- direct_count가 적더라도 해당 aspect 전체 Evidence가 충분하면 '조건 Evidence가 없다'고 말하지 않는다.
- situational_changes는 관찰 연관성일 뿐 인과관계라고 말하지 않는다.
- Wald는 실제 실패율/반품률/예약 실패율이 아니라 검색된 이탈 신호 건수라고 말한다.
- 사용자가 tolerate라고 한 조건은 약한 가중치, avoid라고 한 조건은 회피 요구로 설명한다.
- 대표 Evidence는 payload에 있는 snippet 내용만 사용하고 새로운 사례를 만들어내지 않는다.

반드시 JSON만 반환한다.
{
  "headline": "최종 판정 한 줄",
  "answer": "5~8문장으로 전체→조건→쟁점→최종선택을 연결한 설명",
  "reasons": ["전체 여론 근거", "조건 근거", "쟁점 키워드/대표 Evidence", "상황 또는 Wald 근거"],
  "risks": ["판단 한계", "최종 결정 전 확인사항"]
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
            "reasons": [str(x) for x in parsed.get("reasons", [])][:5],
            "risks": [str(x) for x in parsed.get("risks", [])][:3],
            "source": "openai",
        }
        if not result["headline"] or not result["answer"]:
            return fallback_explanation(analysis)
        return result
    except Exception:
        return fallback_explanation(analysis)
