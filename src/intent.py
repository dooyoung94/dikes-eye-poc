from __future__ import annotations

import json
import re
from typing import Any

from src.condition_taxonomy import (
    CONDITION_RULES,
    aliases_for,
    aspect_for_term,
    aspect_label,
    extract_preference_terms,
)

try:
    from openai import OpenAI
except (ImportError, ModuleNotFoundError):
    OpenAI = None


PRODUCT_HINTS = [
    "살까", "사도", "구매", "제품", "상품", "노트북", "이어폰", "헤드폰", "스마트폰",
    "폰", "태블릿", "모니터", "키보드", "마우스", "신발", "러닝화", "운동화", "가방",
    "카메라", "렌즈", "청소기", "세탁기", "건조기", "TV", "티비", "화장품", "선크림",
    "향수", "의자", "매트리스", "배터리", "충전기",
]

RESTAURANT_HINTS = [
    "식당", "맛집", "카페", "레스토랑", "다이닝", "술집", "바", "브런치", "밥집",
    "소개팅", "데이트", "회식", "모임", "예약", "웨이팅",
]

PURPOSE_RULES = {
    "소개팅": ["소개팅"],
    "데이트": ["데이트"],
    "친구 모임": ["친구 모임", "친구들이랑", "친구랑", "모임"],
    "가족": ["가족", "부모님", "아이랑", "아이와"],
    "회식": ["회식"],
    "업무 미팅": ["업무 미팅", "미팅", "회의"],
    "혼밥": ["혼밥", "혼자"],
    "출퇴근": ["출퇴근용", "출퇴근", "통근용", "통근"],
    "업무": ["업무용", "사무용", "회사에서", "업무"],
    "게임": ["게임용", "게이밍", "게임"],
    "여행": ["여행용", "여행"],
    "운동": ["운동용", "러닝용", "러닝", "헬스", "운동"],
    "영상/사진": ["영상 편집", "영상편집", "사진 편집", "사진촬영", "촬영용"],
}

QUESTION_PATTERNS = [
    r"\b사도\s*(?:될까|돼|되나|괜찮을까)\b.*$",
    r"\b살까\b.*$",
    r"\b구매해도\s*(?:될까|돼|되나)\b.*$",
    r"\b가도\s*(?:될까|돼|되나)\b.*$",
    r"\b어때\??.*$",
    r"\b괜찮아\??.*$",
    r"\b괜찮을까\??.*$",
    r"\b추천해(?:줘)?\??.*$",
]


def _first_purpose(text: str) -> str:
    for label, words in PURPOSE_RULES.items():
        if any(word in text for word in words):
            return label
    return ""


def _extract_day(text: str) -> str:
    patterns = [
        r"(?:이번\s*)?(월요일|화요일|수요일|목요일|금요일|토요일|일요일|평일|주말)",
        r"(오늘|내일|모레)",
        r"(\d{1,2})\s*월\s*(\d{1,2})\s*일",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return ""


def _extract_time(text: str) -> str:
    match = re.search(r"(?:(오전|오후)\s*)?(\d{1,2})(?::(\d{2}))?\s*시", text)
    if match:
        ampm, hour_raw, minute = match.groups()
        hour = int(hour_raw)
        if ampm == "오후" and hour < 12:
            hour += 12
        if ampm == "오전" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute or '00'}"
    match = re.search(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", text)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    return ""


def _detect_kind(text: str) -> str:
    product_score = sum(1 for hint in PRODUCT_HINTS if hint.lower() in text.lower())
    restaurant_score = sum(1 for hint in RESTAURANT_HINTS if hint in text)
    return "product" if product_score > restaurant_score else "restaurant"


def _remove_preference_clause(text: str) -> str:
    preference_words = [
        alias for rule in CONDITION_RULES.values() for alias in rule.get("aliases", [])
    ]
    markers = [
        "중요해", "중요하고", "중요한데", "중요한", "중요합니다",
        "신경써", "신경 쓰", "우선이야", "싫어", "싫고", "피하고",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx < 0:
            continue
        left = text[:idx]
        cut = max(left.rfind("?"), left.rfind("."), left.rfind(","))
        if cut >= 0:
            return text[:cut]
        positions = [text.find(word) for word in preference_words if text.find(word) >= 0]
        if positions:
            return text[:min(positions)]
    return text


def _strip_context(text: str, *, kind: str, day: str, purpose: str) -> str:
    target = _remove_preference_clause(text.strip())
    if day:
        target = target.replace(day, " ")
    target = re.sub(r"(?:(?:오전|오후)\s*)?\d{1,2}(?::\d{2})?\s*시", " ", target)
    target = re.sub(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)", " ", target)
    if purpose:
        for word in PURPOSE_RULES.get(purpose, []):
            target = target.replace(word, " ")
    if kind == "product":
        target = re.sub(r"(?:용)?으로", " ", target)
    for pattern in QUESTION_PATTERNS:
        target = re.sub(pattern, " ", target)
    for token in ["인데", "에서", "기준", "괜찮은지", "알려줘", "좀", "나한테", "어떤가", "어떤가요"]:
        target = target.replace(token, " ")
    if kind == "product":
        for token in ["구매", "제품", "상품"]:
            target = target.replace(token, " ")
    target = re.sub(r"\s+", " ", target).strip(" ?!,.")
    return target or text.strip()


def _direction_for(text: str, term: str) -> str:
    lowered = text.lower()
    idx = lowered.find(term.lower())
    window = lowered[max(0, idx - 18): idx + len(term) + 24] if idx >= 0 else lowered
    if any(x in window for x in ["상관없", "괜찮아", "괜찮고", "감수", "비싸도", "조금은 괜찮"]):
        return "tolerate"
    if any(x in window for x in ["싫", "피하고", "없었으면", "적었으면", "최소화", "안 좋아"]):
        return "avoid"
    return "prefer"


def _importance_for(text: str, term: str, direction: str) -> float:
    lowered = text.lower()
    idx = lowered.find(term.lower())
    window = lowered[max(0, idx - 18): idx + len(term) + 24] if idx >= 0 else lowered
    if direction == "tolerate":
        return 0.35
    if any(x in window for x in ["제일", "가장", "무조건", "꼭", "매우", "정말 중요", "최우선"]):
        return 1.0
    if any(x in window for x in ["중요", "신경", "싫", "필수"]):
        return 0.9
    if any(x in window for x in ["조금", "되도록", "가능하면"]):
        return 0.6
    return 0.75


def _fallback_conditions(text: str) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in extract_preference_terms(text, limit=8):
        aspect = aspect_for_term(term)
        if not aspect or aspect in seen:
            continue
        seen.add(aspect)
        direction = _direction_for(text, term)
        conditions.append({
            "raw": term,
            "aspect": aspect,
            "label": aspect_label(aspect),
            "direction": direction,
            "importance": _importance_for(text, term, direction),
            "search_terms": list(dict.fromkeys([term, *aliases_for(aspect)[:3]]))[:4],
        })
    return conditions


def _fallback_parse(text: str) -> dict[str, Any]:
    original = re.sub(r"\s+", " ", str(text or "")).strip()
    kind = _detect_kind(original)
    day = _extract_day(original)
    time = _extract_time(original)
    purpose = _first_purpose(original)
    conditions = _fallback_conditions(original)
    preference = ", ".join(c["raw"] for c in conditions)
    target = _strip_context(original, kind=kind, day=day, purpose=purpose)
    confidence = 0.55
    if target and target != original:
        confidence += 0.15
    if purpose:
        confidence += 0.10
    if day or time:
        confidence += 0.10
    if conditions:
        confidence += 0.05
    return {
        "kind": kind,
        "target": target,
        "original": original,
        "date_or_day": day,
        "time": time,
        "purpose": purpose,
        "preference": preference,
        "conditions": conditions,
        "parse_confidence": round(min(0.90, confidence), 2),
        "parser_source": "heuristic",
    }


def _validate_conditions(items: Any, fallback_text: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return _fallback_conditions(fallback_text)
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items[:6]:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("raw") or "").strip()
        aspect = str(item.get("aspect") or "").strip()
        if aspect not in CONDITION_RULES:
            aspect = aspect_for_term(raw) or ""
        if aspect not in CONDITION_RULES or aspect in seen:
            continue
        seen.add(aspect)
        direction = str(item.get("direction") or "prefer").strip().lower()
        if direction not in {"prefer", "avoid", "tolerate"}:
            direction = "prefer"
        try:
            importance = float(item.get("importance", 0.8))
        except (TypeError, ValueError):
            importance = 0.8
        importance = max(0.1, min(1.0, importance))
        if direction == "tolerate":
            importance = min(importance, 0.5)
        terms = item.get("search_terms", [])
        if not isinstance(terms, list):
            terms = []
        search_terms = [str(x).strip() for x in terms if str(x).strip()]
        search_terms = list(dict.fromkeys([raw or aspect_label(aspect), *search_terms, *aliases_for(aspect)[:2]]))[:5]
        validated.append({
            "raw": raw or aspect_label(aspect),
            "aspect": aspect,
            "label": aspect_label(aspect),
            "direction": direction,
            "importance": round(importance, 2),
            "search_terms": search_terms,
        })
    return validated or _fallback_conditions(fallback_text)


def parse_intent(text: str, *, api_key: str = "", model: str = "gpt-5-mini") -> dict[str, Any]:
    fallback = _fallback_parse(text)
    if not api_key or OpenAI is None:
        return fallback

    canonical = {
        aspect: {"label": rule["label"], "examples": rule.get("aliases", [])[:6]}
        for aspect, rule in CONDITION_RULES.items()
    }
    instruction = f"""
너는 Dike's Eye Conditional Decision Agent의 입력 해석기다.
사용자의 자연어 질문을 분석용 구조로만 변환한다. 추천이나 평가를 하지 않는다.
반드시 JSON 하나만 반환한다.
{{
  "kind": "restaurant 또는 product",
  "target": "평가 대상의 정확한 이름만",
  "date_or_day": "요일/날짜/사용상황, 없으면 빈 문자열",
  "time": "가능하면 24시간 HH:MM, 없으면 빈 문자열",
  "purpose": "소개팅/데이트/회식/출퇴근/업무 등 목적, 없으면 빈 문자열",
  "conditions": [
    {{
      "raw": "사용자가 말한 조건 표현",
      "aspect": "아래 canonical aspect 중 하나",
      "direction": "prefer | avoid | tolerate",
      "importance": 0.1,
      "search_terms": ["검색에 도움이 되는 한국어 표현"]
    }}
  ],
  "confidence": 0.0
}}
조건 해석 원칙:
- '가격이 중요'는 price_value / prefer / 높은 중요도.
- '비싸도 괜찮아'는 price_value / tolerate / 낮은 중요도.
- '웨이팅은 싫어'는 wait_reservation / avoid / 높은 중요도.
- '아늑하고 편한 곳'은 comfort / prefer.
- '분위기 좋은 곳'은 noise_atmosphere / prefer.
- 같은 의미의 표현은 하나의 aspect로 합친다.
- 사용자가 말하지 않은 중요조건을 만들어내지 않는다.
- target에는 조건 문구를 넣지 않는다.
Canonical aspects:
{json.dumps(canonical, ensure_ascii=False)}
""".strip()

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": fallback["original"]},
            ],
        )
        raw = str(getattr(response, "output_text", "") or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].lstrip()
        parsed = json.loads(raw)
        kind = str(parsed.get("kind") or fallback["kind"]).strip()
        if kind not in {"restaurant", "product"}:
            kind = fallback["kind"]
        conditions = _validate_conditions(parsed.get("conditions"), fallback["original"])
        preference = ", ".join(c["raw"] for c in conditions)
        try:
            confidence = float(parsed.get("confidence", fallback["parse_confidence"]))
        except (TypeError, ValueError):
            confidence = float(fallback["parse_confidence"])
        result = {
            "kind": kind,
            "target": str(parsed.get("target") or fallback["target"]).strip(),
            "original": fallback["original"],
            "date_or_day": str(parsed.get("date_or_day") or fallback["date_or_day"]).strip(),
            "time": str(parsed.get("time") or fallback["time"]).strip(),
            "purpose": str(parsed.get("purpose") or fallback["purpose"]).strip(),
            "preference": preference,
            "conditions": conditions,
            "parse_confidence": round(max(0.0, min(1.0, confidence)), 2),
            "parser_source": "openai",
        }
        return result if result["target"] else fallback
    except Exception:
        return fallback
