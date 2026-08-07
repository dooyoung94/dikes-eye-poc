from __future__ import annotations

import re
from typing import Any


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

PREFERENCE_WORDS = [
    "조용", "분위기", "대화", "웨이팅", "예약", "주차", "친절", "가격", "가성비",
    "안락함", "안락", "편안함", "편안", "아늑함", "아늑", "쾌적함", "쾌적", "좌석",
    "맛", "배터리", "발열", "성능", "휴대", "무게", "착용", "착용감", "발볼", "사이즈",
    "음질", "노이즈캔슬링", "화질", "내구", "불량", "AS", "연결", "디자인", "감성",
]

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


def _extract_preferences(text: str) -> str:
    found = []
    lowered = text.lower()
    for word in PREFERENCE_WORDS:
        if word.lower() in lowered:
            found.append(word)
    # 긴 표현을 먼저 보존하고, 포함관계인 짧은 표현은 제거한다.
    unique = []
    for word in sorted(dict.fromkeys(found), key=len, reverse=True):
        if not any(word in kept for kept in unique):
            unique.append(word)
    return ", ".join(reversed(unique))


def _detect_kind(text: str) -> str:
    product_score = sum(1 for hint in PRODUCT_HINTS if hint.lower() in text.lower())
    restaurant_score = sum(1 for hint in RESTAURANT_HINTS if hint in text)
    if product_score > restaurant_score:
        return "product"
    return "restaurant"


def _remove_preference_clause(text: str) -> str:
    markers = ["중요해", "중요하고", "중요한데", "중요한", "신경써", "신경 쓰", "우선이야"]
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            left = text[:idx]
            cut = max(left.rfind("?"), left.rfind("."), left.rfind(","))
            if cut >= 0:
                return text[:cut]
            positions = [text.find(word) for word in PREFERENCE_WORDS if text.find(word) >= 0]
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

    cleanup = [
        "인데", "에서", "기준", "괜찮은지", "알려줘", "좀", "나한테", "어떤가", "어떤가요",
    ]
    if kind == "product":
        cleanup += ["구매", "제품", "상품"]
    for token in cleanup:
        target = target.replace(token, " ")

    target = re.sub(r"\s+", " ", target).strip(" ?!,.")
    return target or text.strip()


def parse_intent(text: str) -> dict[str, Any]:
    original = re.sub(r"\s+", " ", str(text or "")).strip()
    kind = _detect_kind(original)
    day = _extract_day(original)
    time = _extract_time(original)
    purpose = _first_purpose(original)
    preference = _extract_preferences(original)
    target = _strip_context(original, kind=kind, day=day, purpose=purpose)

    confidence = 0.55
    if target and target != original:
        confidence += 0.15
    if purpose:
        confidence += 0.10
    if day or time:
        confidence += 0.10
    if preference:
        confidence += 0.05
    if any(h.lower() in original.lower() for h in (PRODUCT_HINTS if kind == "product" else RESTAURANT_HINTS)):
        confidence += 0.10

    return {
        "kind": kind,
        "target": target,
        "original": original,
        "date_or_day": day,
        "time": time,
        "purpose": purpose,
        "preference": preference,
        "parse_confidence": round(min(0.95, confidence), 2),
    }
