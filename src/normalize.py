from __future__ import annotations

import re
from datetime import date, datetime
from math import exp
from typing import Any

from src.condition_analysis import normalize_context_conditions
from src.condition_taxonomy import aliases_for


ASPECT_RULES = {
    "price_value": ["가격", "가성비", "비용", "금액", "예산", "저렴", "비싸", "가격대", "할인"],
    "noise_atmosphere": ["분위기", "조용", "조용함", "소음", "대화", "감성", "로맨틱", "시끄"],
    "comfort": ["안락", "안락함", "편안", "편안함", "아늑", "아늑함", "쾌적", "쾌적함", "좌석", "의자", "테이블 간격", "공간감"],
    "wait_reservation": ["웨이팅", "대기", "예약", "줄", "입장", "혼잡", "기다림"],
    "service": ["친절", "불친절", "서비스", "직원", "응대", "AS", "고객센터"],
    "quality_performance": ["맛", "품질", "성능", "속도", "내구", "고장", "불량", "연결", "끊김", "발열", "카메라"],
    "battery": ["배터리", "충전", "사용시간", "배터리시간", "충전속도", "배터리수명"],
    "audio_visual": ["음질", "화질", "노이즈캔슬링", "노캔", "화면", "디스플레이", "스피커"],
    "convenience_fit": ["주차", "접근", "접근성", "거리", "휴대", "휴대성", "무게", "사이즈", "발볼", "착용", "착용감", "그립", "설치", "사용성", "조작"],
    "design_experience": ["디자인", "색상", "마감", "예쁜", "예쁘", "촉감", "외관"],
}

POSITIVE = [
    "좋", "맛있", "만족", "친절", "조용", "편하", "편안", "안락", "아늑", "쾌적",
    "추천", "괜찮", "예쁘", "빠르", "선명", "가볍", "오래가", "안정", "훌륭",
    "재구매", "재방문", "저렴", "합리적", "넉넉", "충분",
]
NEGATIVE = [
    "나쁘", "별로", "불만", "불친절", "시끄", "불편", "답답", "비싸", "느리", "좁",
    "대기", "웨이팅", "혼잡", "발열", "무겁", "끊김", "불량", "고장", "환불", "반품",
    "후회", "짧", "실패", "취소", "못", "안됨", "부담", "아쉽",
]

CONTEXT_RULES = {
    "weekday": ["평일", "월요일", "화요일", "수요일", "목요일", "금요일"],
    "weekend": ["주말", "토요일", "일요일"],
    "lunch": ["점심", "런치", "12시", "13시", "낮"],
    "dinner": ["저녁", "디너", "18시", "19시", "20시", "21시", "밤"],
    "commute": ["출퇴근", "통근", "지하철", "버스"],
    "office": ["업무용", "사무용", "회사", "회의"],
    "gaming": ["게임", "게이밍", "프레임"],
    "travel": ["여행", "출장"],
    "exercise": ["러닝", "헬스", "운동", "등산"],
}

PURPOSE_CONTEXT = {
    "출퇴근": "commute",
    "업무": "office",
    "업무 미팅": "office",
    "게임": "gaming",
    "여행": "travel",
    "운동": "exercise",
}

PURPOSE_PROXY = {
    "소개팅": ["조용", "대화", "분위기", "안락", "아늑", "예약", "웨이팅"],
    "데이트": ["분위기", "조용", "대화", "안락", "아늑", "예약", "웨이팅"],
    "업무 미팅": ["조용", "대화", "접근", "예약"],
    "친구 모임": ["분위기", "대기", "예약", "가격"],
    "가족": ["주차", "친절", "대기", "예약", "편안"],
    "회식": ["예약", "대기", "가격", "서비스"],
    "혼밥": ["대기", "가격", "편하"],
    "출퇴근": ["휴대", "무게", "배터리", "착용", "편안"],
    "업무": ["배터리", "성능", "무게", "화면", "연결", "AS"],
    "게임": ["성능", "발열", "화질", "배터리", "끊김"],
    "여행": ["휴대", "무게", "배터리", "내구", "충전"],
    "운동": ["착용", "편안", "무게", "내구", "배터리"],
    "영상/사진": ["성능", "화질", "카메라", "배터리", "화면"],
}


def _parse_date(raw: str) -> str:
    digits = re.sub(r"[^0-9]", "", str(raw or ""))
    if len(digits) != 8:
        return ""
    try:
        return datetime.strptime(digits, "%Y%m%d").date().isoformat()
    except ValueError:
        return ""


def _recency(post_date: str) -> tuple[float, int | None]:
    if not post_date:
        return 0.45, None
    try:
        dt = datetime.fromisoformat(post_date).date()
    except ValueError:
        return 0.45, None
    days = max(0, (date.today() - dt).days)
    return round(max(0.15, min(1.0, exp(-days / 180.0))), 4), days


def _contains(text: str, words: list[str]) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in words)


def _aspects(text: str) -> list[str]:
    found = [name for name, words in ASPECT_RULES.items() if _contains(text, words)]
    return found or ["other"]


def _contexts(text: str) -> list[str]:
    return [name for name, words in CONTEXT_RULES.items() if _contains(text, words)]


def _sentiment(text: str) -> int:
    pos = sum(1 for word in POSITIVE if word in text)
    neg = sum(1 for word in NEGATIVE if word in text)
    return 1 if pos > neg else -1 if neg > pos else 0


def _user_context_flags(context: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    day = str(context.get("date_or_day") or "")
    time = str(context.get("time") or "")
    purpose = str(context.get("purpose") or "")
    if any(x in day for x in ["토요일", "일요일", "주말"]):
        flags.add("weekend")
    if any(x in day for x in ["월요일", "화요일", "수요일", "목요일", "금요일", "평일"]):
        flags.add("weekday")
    digits = re.findall(r"\d{1,2}", time)
    if digits:
        hour = int(digits[0])
        if 11 <= hour <= 14:
            flags.add("lunch")
        if 17 <= hour <= 22:
            flags.add("dinner")
    if purpose in PURPOSE_CONTEXT:
        flags.add(PURPOSE_CONTEXT[purpose])
    return flags


def _condition_direct_aspects(
    text: str,
    query: str,
    detected_aspects: list[str],
    context: dict[str, Any],
) -> list[str]:
    haystack = f"{query} {text}".lower()
    direct: list[str] = []
    for cond in normalize_context_conditions(context):
        aspect = str(cond.get("aspect") or "")
        if aspect not in detected_aspects:
            continue
        terms = [
            str(cond.get("raw") or ""),
            *[str(x) for x in cond.get("search_terms", [])],
            *aliases_for(aspect)[:5],
        ]
        if any(term and term.lower() in haystack for term in terms):
            direct.append(aspect)
    return list(dict.fromkeys(direct))


def _match(
    text: str,
    context: dict[str, Any],
    retrieval_scope: str,
    detected_contexts: list[str],
    condition_direct_aspects: list[str],
) -> float:
    raw = " ".join(
        str(context.get(k, ""))
        for k in ("date_or_day", "time", "purpose", "preference")
    )
    tokens = [t for t in re.findall(r"[가-힣A-Za-z0-9]+", raw) if len(t) >= 2]
    tokens.extend(PURPOSE_PROXY.get(str(context.get("purpose") or ""), []))
    tokens = list(dict.fromkeys(tokens))
    lexical = 0.35 if not tokens else min(
        1.0,
        sum(1 for token in tokens if token.lower() in text.lower())
        / max(3, min(8, len(tokens))),
    )
    flags = _user_context_flags(context)
    situational_overlap = (
        len(flags.intersection(detected_contexts)) / max(1, len(flags)) if flags else 0.0
    )
    retrieval_bonus = 0.18 if retrieval_scope == "user_context" else 0.10 if retrieval_scope == "preference" else 0.0
    condition_bonus = min(0.32, len(condition_direct_aspects) * 0.16)
    return round(min(1.0, 0.55 * lexical + 0.25 * situational_overlap + retrieval_bonus + condition_bonus), 4)


def normalize_evidence(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    user_flags = _user_context_flags(context)

    for idx, row in enumerate(rows, start=1):
        title = str(row.get("title") or "").strip()
        snippet = str(row.get("snippet") or "").strip()
        text = f"{title} {snippet}".strip()
        query = str(row.get("query") or "")
        post_date = _parse_date(str(row.get("post_date") or ""))
        r, days = _recency(post_date)
        detected_contexts = _contexts(text)
        detected_aspects = _aspects(text)
        retrieval_scope = str(row.get("retrieval_scope") or "base")
        direct_aspects = _condition_direct_aspects(text, query, detected_aspects, context)
        situational_aligned = bool(user_flags.intersection(detected_contexts)) or retrieval_scope == "user_context"
        condition_aligned = bool(direct_aspects)

        out.append({
            **row,
            "evidence_id": f"E{idx:03d}",
            "post_date": post_date,
            "text": text,
            "aspects": detected_aspects,
            "contexts": detected_contexts,
            "sentiment": _sentiment(text),
            "R": r,
            "recency_days": days,
            "M": _match(text, context, retrieval_scope, detected_contexts, direct_aspects),
            "situational_aligned": situational_aligned,
            "condition_aligned": condition_aligned,
            "context_aligned": situational_aligned or condition_aligned,
            "preference_aligned": condition_aligned,
            "condition_direct_aspects": direct_aspects,
            "user_context_flags": sorted(user_flags),
        })
    return out
