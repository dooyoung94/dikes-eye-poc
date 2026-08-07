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

    label = {
        "GO": "권고",
        "CONDITIONAL": "조건부 권고",
        "AVOID": "권고 유보",
    }.get(verdict, verdict)

    findings = []
    findings.append(
        f"현재 확보된 공개 Evidence를 기준으로 검토한 결과, 본 건은 '{label}' 의견에 해당합니다. "
        f"조건 적합도는 {score}/100, 판단 신뢰도는 {confidence}%입니다."
    )

    if rca_risk > 0.35:
        findings.append(
            "사용자 조건과 직접 맞물리는 부정적 패턴이 일부 확인되어, 평균적인 만족도만으로 판단하기에는 주의가 필요합니다."
        )
    else:
        findings.append(
            "사용자 조건과 직접 맞물리는 강한 부정적 패턴은 현재 Evidence 범위에서 제한적으로 확인됩니다."
        )

    if wald_risk > 0.35:
        if kind == "product":
            findings.append(
                "일반 만족 후기 바깥에서 반품·불량·재판매 등 이탈 측 Evidence가 일부 확인되어 별도 위험요소로 고려했습니다."
            )
        else:
            findings.append(
                "일반 방문 후기 바깥에서 예약 실패·웨이팅 포기 등 선택 이전 이탈 신호가 일부 확인되어 별도 위험요소로 고려했습니다."
            )
    else:
        findings.append(
            "현재 검색 범위에서는 일반 리뷰 바깥의 이탈 신호가 강하게 나타나지는 않았습니다."
        )

    return {
        "headline": f"검토 의견 · {label}",
        "answer": " ".join(findings),
        "reasons": [
            "판단은 리뷰 평균이 아니라 사용자 조건과 맞는 Evidence의 최신성·반복성·일치도를 함께 반영했습니다.",
            "상반된 의견은 별도로 분리해 검토했으며, 조건별 차이는 관찰 연관성 수준으로만 해석했습니다.",
            "리뷰에 남기 어려운 이탈 측 Evidence도 별도 검색해 누락 가능성을 보완했습니다.",
        ],
        "risks": [
            "RCA는 인과관계를 확정하는 분석이 아니라 현재 Evidence에서 관찰되는 조건별 차이입니다.",
            "Wald 신호는 실제 실패율·반품률·예약 실패율을 의미하지 않으며 선택편향 가능성을 알리는 보조 신호입니다.",
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

    decision = analysis.get("decision", {})
    payload = {
        "kind": analysis.get("kind", "restaurant"),
        "target": analysis.get("target", ""),
        "context": analysis.get("context", {}),
        "decision": decision,
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
너는 Dike's Eye의 '검토 의견 작성 보조인'이다.
말투는 한국어 법률보조인·조사관이 검토의견서를 작성하는 것처럼 차분하고 정중하며 구체적으로 작성한다.
과장, 단정, 마케팅 표현은 사용하지 않는다.

중요 원칙:
1. verdict, fit_score, confidence, score components는 이미 deterministic engine에서 확정되었다. 절대로 다시 계산하거나 변경하지 않는다.
2. RCA는 인과관계 확정이 아니라 '현재 Evidence에서 관찰된 조건별 연관성'으로만 표현한다.
3. Wald는 실제 실패율/반품률/예약 실패율이 아니라 리뷰 밖에 존재할 수 있는 선택편향·이탈 위험 신호라고 명시한다.
4. 근거가 부족하면 부족하다고 명시한다. 검색되지 않은 것을 존재하지 않는다고 표현하지 않는다.
5. 사용자가 실제 결정을 내릴 수 있도록 결론뿐 아니라 반대 Evidence, 판단의 한계, 확인해야 할 후속사항까지 설명한다.
6. '법률 자문', '판결', '유죄/무죄'처럼 실제 법률서비스로 오인될 표현은 사용하지 않는다. 검토·의견·Evidence·권고라는 용어를 사용한다.

반드시 아래 JSON 형식만 출력한다.
{
  "headline": "검토 의견 한 줄",
  "answer": "4~7문장으로 상세 검토 의견",
  "reasons": ["확인된 사실 또는 주요 근거1", "상반된 Evidence 또는 조건 차이2", "누락 가능성 또는 추가 근거3"],
  "risks": ["판단의 한계1", "결정 전에 확인할 사항2"]
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
