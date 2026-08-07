from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import requests

TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    return TAG_RE.sub("", str(text or "")).strip()


def _auth(*, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> tuple[str, dict[str, str]] | None:
    if hub_id and hub_secret:
        return "hub", {"X-NCP-APIGW-API-KEY-ID": hub_id, "X-NCP-APIGW-API-KEY": hub_secret}
    if legacy_id and legacy_secret:
        return "legacy", {"X-Naver-Client-Id": legacy_id, "X-Naver-Client-Secret": legacy_secret}
    return None


def local_search(query: str, *, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> tuple[list[dict[str, Any]], str]:
    auth = _auth(hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret)
    if not auth:
        return [], "NAVER Secret 미설정"
    mode, headers = auth
    url = "https://naverapihub.apigw.ntruss.com/search/v1/local" if mode == "hub" else "https://openapi.naver.com/v1/search/local.json"
    try:
        response = requests.get(url, headers=headers, params={"query": query, "display": 5, "sort": "comment"}, timeout=6)
        response.raise_for_status()
        items = response.json().get("items", [])
        rows = [{
            "title": clean_html(item.get("title", "")),
            "category": clean_html(item.get("category", "")),
            "address": clean_html(item.get("roadAddress") or item.get("address", "")),
            "link": str(item.get("link") or "").strip(),
        } for item in items]
        return rows, "정상"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=512)
def _search_content_cached(endpoint: str, query: str, hub_id: str, hub_secret: str, legacy_id: str, legacy_secret: str, display: int) -> tuple[tuple[str, str, str, str, str, str], ...]:
    auth = _auth(hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret)
    if not auth:
        return tuple()
    mode, headers = auth
    url = f"https://naverapihub.apigw.ntruss.com/search/v1/{endpoint}" if mode == "hub" else f"https://openapi.naver.com/v1/search/{endpoint}.json"
    params: dict[str, Any] = {"query": query, "display": display, "start": 1}
    if endpoint in {"blog", "cafearticle"}:
        params["sort"] = "sim"
    try:
        response = requests.get(url, headers=headers, params=params, timeout=6)
        response.raise_for_status()
        items = response.json().get("items", [])
    except Exception:
        return tuple()
    return tuple((endpoint, query, clean_html(item.get("title", "")), clean_html(item.get("description", "")), str(item.get("link") or "").strip(), str(item.get("postdate") or "")) for item in items)


def search_content(endpoint: str, query: str, *, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "", display: int = 8) -> list[dict[str, Any]]:
    packed = _search_content_cached(endpoint, query, hub_id, hub_secret, legacy_id, legacy_secret, display)
    return [{"source": source, "query": q, "title": title, "snippet": snippet, "link": link, "post_date": post_date} for source, q, title, snippet, link, post_date in packed]


def _dedupe(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("link") or f"{row.get('title')}|{row.get('snippet')}").strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped[:limit]


def _context_terms(context: dict[str, Any] | None, kind: str) -> list[str]:
    if not context:
        return []
    terms: list[str] = []
    day = str(context.get("date_or_day") or "").strip()
    time = str(context.get("time") or "").strip()
    purpose = str(context.get("purpose") or "").strip()
    preference = str(context.get("preference") or "").strip()

    if day:
        terms.append(day)
    if time:
        digits = re.findall(r"\d{1,2}", time)
        if digits:
            hour = int(digits[0])
            if kind == "restaurant":
                if 11 <= hour <= 14:
                    terms.append("점심")
                elif 17 <= hour <= 22:
                    terms.append("저녁")
            else:
                terms.append(time)
    if purpose:
        terms.append(purpose)
    if preference:
        # 검색어 폭주 방지를 위해 앞의 핵심 조건 2개만 사용
        prefs = [x.strip() for x in re.split(r"[,/·]", preference) if x.strip()]
        terms.extend(prefs[:2])
    return list(dict.fromkeys(terms))[:5]


def _query_plan(target: str, kind: str, hidden: bool, context: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    context_terms = _context_terms(context, kind)
    context_query = " ".join([target, *context_terms]).strip()

    if kind == "product":
        if hidden:
            base = f"{target} 반품 환불 불량 후회"
            contextual = f"{context_query} 반품 단점 문제"
        else:
            base = f"{target} 실사용 후기 장단점"
            contextual = f"{context_query} 실사용 후기"
    else:
        if hidden:
            base = f"{target} 예약 실패 웨이팅 포기 주차 포기"
            contextual = f"{context_query} 예약 웨이팅 실패"
        else:
            base = f"{target} 후기 분위기 서비스 웨이팅"
            contextual = f"{context_query} 후기"

    plan = [(base, "base")]
    if context_terms:
        plan.append((contextual, "user_context"))
    else:
        # 조건이 없으면 기존처럼 보조 일반 질의 사용
        plan.append((f"{target} 리뷰 단점 장점", "base"))
    return plan


def _collect_parallel(target: str, *, kind: str, hidden: bool, context: dict[str, Any] | None = None, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> list[dict[str, Any]]:
    query_plan = _query_plan(target, kind, hidden, context)
    primary_endpoints = ("blog", "cafearticle")
    jobs = [(endpoint, query, scope) for query, scope in query_plan for endpoint in primary_endpoints]
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(search_content, endpoint, query, hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret, display=7 if not hidden else 5): (endpoint, query, scope)
            for endpoint, query, scope in jobs
        }
        for future in as_completed(futures):
            endpoint, query, scope = futures[future]
            try:
                found = future.result()
                for row in found:
                    row["retrieval_scope"] = scope
                    row["context_query"] = query if scope == "user_context" else ""
                rows.extend(found)
            except Exception:
                continue

    minimum = 14 if not hidden else 6
    if len(_dedupe(rows, 100)) < minimum:
        for query, scope in query_plan:
            found = search_content("webkr", query, hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret, display=6 if not hidden else 4)
            for row in found:
                row["retrieval_scope"] = scope
                row["context_query"] = query if scope == "user_context" else ""
            rows.extend(found)

    # 같은 문서가 base/context 양쪽에서 발견되면 user_context 쪽을 우선 보존
    rows.sort(key=lambda x: 0 if x.get("retrieval_scope") == "user_context" else 1)
    return _dedupe(rows, 40 if not hidden else 24)


def collect_visible_evidence(target: str, *, kind: str = "restaurant", context: dict[str, Any] | None = None, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> list[dict[str, Any]]:
    return _collect_parallel(target, kind=kind, hidden=False, context=context, hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret)


def collect_hidden_evidence(target: str, *, kind: str = "restaurant", context: dict[str, Any] | None = None, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> list[dict[str, Any]]:
    return _collect_parallel(target, kind=kind, hidden=True, context=context, hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret)
