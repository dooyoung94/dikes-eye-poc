from __future__ import annotations

import re
from typing import Any


CONDITION_RULES: dict[str, dict[str, Any]] = {
    "price_value": {
        "label": "가격·가성비",
        "aliases": ["가격", "가성비", "비용", "금액", "예산", "저렴", "비싸", "가격대", "할인"],
    },
    "noise_atmosphere": {
        "label": "분위기·소음",
        "aliases": ["분위기", "조용", "조용함", "소음", "대화", "감성", "로맨틱", "시끄", "분위기좋은"],
    },
    "comfort": {
        "label": "안락함·편안함",
        "aliases": ["안락", "안락함", "편안", "편안함", "아늑", "아늑함", "쾌적", "쾌적함", "좌석", "의자", "테이블 간격", "공간감"],
    },
    "wait_reservation": {
        "label": "웨이팅·예약",
        "aliases": ["웨이팅", "대기", "예약", "줄", "입장", "혼잡", "기다림"],
    },
    "service": {
        "label": "서비스·응대",
        "aliases": ["친절", "서비스", "응대", "직원", "불친절", "AS", "고객센터"],
    },
    "quality_performance": {
        "label": "품질·성능",
        "aliases": ["맛", "품질", "성능", "속도", "내구", "내구성", "고장", "불량", "연결", "끊김", "발열", "카메라"],
    },
    "battery": {
        "label": "배터리·충전",
        "aliases": ["배터리", "충전", "사용시간", "배터리시간", "충전속도", "배터리수명"],
    },
    "audio_visual": {
        "label": "음질·화질",
        "aliases": ["음질", "화질", "노이즈캔슬링", "노캔", "화면", "디스플레이", "스피커"],
    },
    "convenience_fit": {
        "label": "편의성·적합성",
        "aliases": ["주차", "접근", "접근성", "거리", "휴대", "휴대성", "무게", "사이즈", "발볼", "착용", "착용감", "그립", "설치", "사용성", "조작"],
    },
    "design_experience": {
        "label": "디자인·사용경험",
        "aliases": ["디자인", "색상", "마감", "예쁜", "예쁘", "촉감", "외관"],
    },
}


def aspect_label(aspect: str) -> str:
    rule = CONDITION_RULES.get(str(aspect), {})
    return str(rule.get("label") or aspect)


def aliases_for(aspect: str) -> list[str]:
    rule = CONDITION_RULES.get(str(aspect), {})
    return [str(x) for x in rule.get("aliases", [])]


def preference_aspects(text: str) -> list[str]:
    lowered = str(text or "").lower()
    found: list[str] = []
    for aspect, rule in CONDITION_RULES.items():
        if any(str(alias).lower() in lowered for alias in rule.get("aliases", [])):
            found.append(aspect)
    return found


def extract_preference_terms(text: str, limit: int = 6) -> list[str]:
    lowered = str(text or "").lower()
    matches: list[str] = []
    for rule in CONDITION_RULES.values():
        for alias in sorted(rule.get("aliases", []), key=lambda x: len(str(x)), reverse=True):
            alias_text = str(alias)
            if alias_text.lower() in lowered:
                matches.append(alias_text)

    # '편안함'이 잡혔는데 '편안'까지 중복으로 들어가는 식의 포함관계를 제거한다.
    deduped: list[str] = []
    for item in sorted(dict.fromkeys(matches), key=len, reverse=True):
        if not any(item in kept for kept in deduped):
            deduped.append(item)
    return list(reversed(deduped))[:limit]


def split_user_preferences(text: str, limit: int = 6) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    explicit = [
        x.strip()
        for x in re.split(r"[,/·]|\s+및\s+|\s+그리고\s+|\s+랑\s+|\s+와\s+|\s+과\s+", raw)
        if x.strip()
    ]
    if len(explicit) > 1:
        return list(dict.fromkeys(explicit))[:limit]

    extracted = extract_preference_terms(raw, limit=limit)
    return extracted or [raw]


def aspect_for_term(term: str) -> str | None:
    lowered = str(term or "").lower()
    best: tuple[int, str] | None = None
    for aspect, rule in CONDITION_RULES.items():
        for alias in rule.get("aliases", []):
            alias_text = str(alias).lower()
            if alias_text in lowered or lowered in alias_text:
                score = len(alias_text)
                if best is None or score > best[0]:
                    best = (score, aspect)
    return best[1] if best else None


def normalized_preferences(text: str, limit: int = 6) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_aspects: set[str] = set()
    for term in split_user_preferences(text, limit=limit):
        aspect = aspect_for_term(term)
        if not aspect or aspect in seen_aspects:
            continue
        seen_aspects.add(aspect)
        results.append({
            "term": term,
            "aspect": aspect,
            "label": aspect_label(aspect),
        })
    return results
