from __future__ import annotations

import re
from datetime import date, datetime
from math import exp
from typing import Any

ASPECT_RULES = {
    "noise_atmosphere": ["조용", "시끄", "소음", "분위기", "대화"],
    "wait_reservation": ["웨이팅", "대기", "예약", "줄"],
    "service": ["친절", "불친절", "서비스", "직원"],
    "price_value": ["가격", "비싸", "가성비", "저렴"],
    "quality": ["맛", "품질", "훌륭", "별로"],
    "convenience": ["주차", "접근", "거리"],
}
POSITIVE = ["좋", "맛있", "만족", "친절", "조용", "편하", "추천", "괜찮", "예쁘", "쾌적"]
NEGATIVE = ["나쁘", "별로", "불만", "불친절", "시끄", "불편", "비싸", "느리", "좁", "대기", "웨이팅", "혼잡"]
CONTEXT_RULES = {
    "weekday": ["평일", "월요일", "화요일", "수요일", "목요일", "금요일"],
    "weekend": ["주말", "토요일", "일요일"],
    "lunch": ["점심", "런치", "12시", "13시", "낮"],
    "dinner": ["저녁", "디너", "18시", "19시", "20시", "21시", "밤"],
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


def _match(text: str, context: dict[str, Any]) -> float:
    raw = " ".join(str(context.get(k, "")) for k in ("date_or_day", "time", "purpose", "preference"))
    tokens = [t for t in re.findall(r"[가-힣A-Za-z0-9]+", raw) if len(t) >= 2]
    if "소개팅" in str(context.get("purpose", "")):
        tokens.extend(["조용", "대화", "분위기", "예약", "웨이팅"])
    tokens = list(dict.fromkeys(tokens))
    if not tokens:
        return 0.35
    hits = sum(1 for token in tokens if token.lower() in text.lower())
    return round(min(1.0, hits / max(3, min(8, len(tokens)))), 4)


def normalize_evidence(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows, start=1):
        title = str(row.get("title") or "").strip()
        snippet = str(row.get("snippet") or "").strip()
        text = f"{title} {snippet}".strip()
        post_date = _parse_date(str(row.get("post_date") or ""))
        r, days = _recency(post_date)
        out.append({**row, "evidence_id": f"E{idx:03d}", "post_date": post_date, "text": text, "aspects": _aspects(text), "contexts": _contexts(text), "sentiment": _sentiment(text), "R": r, "recency_days": days, "M": _match(text, context)})
    return out
