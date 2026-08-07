from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import requests

TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    return TAG_RE.sub("", str(text or "")).strip()


def _auth(
    *,
    hub_id: str = "",
    hub_secret: str = "",
    legacy_id: str = "",
    legacy_secret: str = "",
) -> tuple[str, dict[str, str]] | None:
    if hub_id and hub_secret:
        return (
            "hub",
            {
                "X-NCP-APIGW-API-KEY-ID": hub_id,
                "X-NCP-APIGW-API-KEY": hub_secret,
            },
        )
    if legacy_id and legacy_secret:
        return (
            "legacy",
            {
                "X-Naver-Client-Id": legacy_id,
                "X-Naver-Client-Secret": legacy_secret,
            },
        )
    return None


def local_search(
    query: str,
    *,
    hub_id: str = "",
    hub_secret: str = "",
    legacy_id: str = "",
    legacy_secret: str = "",
) -> tuple[list[dict[str, Any]], str]:
    auth = _auth(
        hub_id=hub_id,
        hub_secret=hub_secret,
        legacy_id=legacy_id,
        legacy_secret=legacy_secret,
    )
    if not auth:
        return [], "NAVER Secret 미설정"

    mode, headers = auth
    url = (
        "https://naverapihub.apigw.ntruss.com/search/v1/local"
        if mode == "hub"
        else "https://openapi.naver.com/v1/search/local.json"
    )

    try:
        response = requests.get(
            url,
            headers=headers,
            params={"query": query, "display": 5, "sort": "comment"},
            timeout=6,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        rows = [
            {
                "title": clean_html(item.get("title", "")),
                "category": clean_html(item.get("category", "")),
                "address": clean_html(
                    item.get("roadAddress") or item.get("address", "")
                ),
                "link": str(item.get("link") or "").strip(),
            }
            for item in items
        ]
        return rows, "정상"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=512)
def _search_content_cached(
    endpoint: str,
    query: str,
    hub_id: str,
    hub_secret: str,
    legacy_id: str,
    legacy_secret: str,
    display: int,
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    auth = _auth(
        hub_id=hub_id,
        hub_secret=hub_secret,
        legacy_id=legacy_id,
        legacy_secret=legacy_secret,
    )
    if not auth:
        return tuple()

    mode, headers = auth
    url = (
        f"https://naverapihub.apigw.ntruss.com/search/v1/{endpoint}"
        if mode == "hub"
        else f"https://openapi.naver.com/v1/search/{endpoint}.json"
    )
    params: dict[str, Any] = {
        "query": query,
        "display": display,
        "start": 1,
    }
    if endpoint in {"blog", "cafearticle"}:
        params["sort"] = "sim"

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=7,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
    except Exception:
        return tuple()

    return tuple(
        (
            endpoint,
            query,
            clean_html(item.get("title", "")),
            clean_html(item.get("description", "")),
            str(item.get("link") or "").strip(),
            str(item.get("postdate") or item.get("postDate") or ""),
        )
        for item in items
    )


def search_content(
    endpoint: str,
    query: str,
    *,
    hub_id: str = "",
    hub_secret: str = "",
    legacy_id: str = "",
    legacy_secret: str = "",
    display: int = 12,
) -> list[dict[str, Any]]:
    packed = _search_content_cached(
        endpoint,
        query,
        hub_id,
        hub_secret,
        legacy_id,
        legacy_secret,
        display,
    )
    return [
        {
            "source": source,
            "query": q,
            "title": title,
            "snippet": snippet,
            "link": link,
            "post_date": post_date,
        }
        for source, q, title, snippet, link, post_date in packed
    ]


def _dedupe(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        key = str(
            row.get("link")
            or f"{row.get('title', '')}|{row.get('snippet', '')}"
        ).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped[:limit]


def _context_terms(
    context: dict[str, Any] | None,
    kind: str,
) -> list[str]:
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
        prefs = [
            item.strip()
            for item in re.split(r"[,/·]", preference)
            if item.strip()
        ]
        terms.extend(prefs[:3])

    return list(dict.fromkeys(terms))[:6]


def _preference_terms(
    context: dict[str, Any] | None,
) -> list[str]:
    if not context:
        return []

    preference = str(context.get("preference") or "").strip()
    if not preference:
        return []

    terms = [
        item.strip()
        for item in re.split(r"[,/·]", preference)
        if item.strip()
    ]
    return list(dict.fromkeys(terms))[:3]


def _query_plan(
    target: str,
    kind: str,
    hidden: bool,
    context: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    context_terms = _context_terms(context, kind)
    preference_terms = _preference_terms(context)
    context_query = " ".join([target, *context_terms]).strip()

    plan: list[tuple[str, str]] = []

    if kind == "product":
        if hidden:
            plan.extend(
                [
                    (f"{target} 반품 환불 불량", "hidden_base"),
                    (f"{target} 고장 후회 중고 판매", "hidden_base"),
                ]
            )
            if context_terms:
                plan.extend(
                    [
                        (f"{context_query} 반품 단점", "user_context"),
                        (f"{context_query} 문제 고장", "user_context"),
                    ]
                )
        else:
            plan.extend(
                [
                    (f"{target} 실사용 후기 장단점", "base"),
                    (f"{target} 장점 추천 만족", "positive"),
                    (f"{target} 단점 문제 불편", "negative"),
                ]
            )
            if context_terms:
                plan.extend(
                    [
                        (f"{context_query} 실사용 후기", "user_context"),
                        (f"{context_query} 추천 장점", "user_context"),
                        (f"{context_query} 단점 문제", "user_context"),
                    ]
                )
            for pref in preference_terms:
                plan.append((f"{target} {pref} 실사용 후기", "preference"))
    else:
        if hidden:
            plan.extend(
                [
                    (f"{target} 예약 실패 웨이팅 포기", "hidden_base"),
                    (f"{target} 주차 포기 재방문 안", "hidden_base"),
                ]
            )
            if context_terms:
                plan.extend(
                    [
                        (f"{context_query} 예약 실패 웨이팅", "user_context"),
                        (f"{context_query} 대기 포기 불편", "user_context"),
                    ]
                )
        else:
            plan.extend(
                [
                    (f"{target} 후기 분위기 서비스 웨이팅", "base"),
                    (f"{target} 장점 추천 만족 후기", "positive"),
                    (f"{target} 단점 불편 아쉬운 후기", "negative"),
                ]
            )
            if context_terms:
                plan.extend(
                    [
                        (f"{context_query} 후기", "user_context"),
                        (f"{context_query} 추천 좋은점", "user_context"),
                        (f"{context_query} 단점 불편", "user_context"),
                    ]
                )
            for pref in preference_terms:
                plan.append((f"{target} {pref} 후기", "preference"))

    unique: list[tuple[str, str]] = []
    seen_queries: set[str] = set()
    for query, scope in plan:
        if query not in seen_queries:
            seen_queries.add(query)
            unique.append((query, scope))
    return unique


def _collect_parallel(
    target: str,
    *,
    kind: str,
    hidden: bool,
    context: dict[str, Any] | None = None,
    hub_id: str = "",
    hub_secret: str = "",
    legacy_id: str = "",
    legacy_secret: str = "",
) -> list[dict[str, Any]]:
    query_plan = _query_plan(
        target,
        kind,
        hidden,
        context,
    )
    primary_endpoints = ("blog", "cafearticle")
    jobs = [
        (endpoint, query, scope)
        for query, scope in query_plan
        for endpoint in primary_endpoints
    ]

    rows: list[dict[str, Any]] = []
    max_workers = min(8, max(1, len(jobs)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                search_content,
                endpoint,
                query,
                hub_id=hub_id,
                hub_secret=hub_secret,
                legacy_id=legacy_id,
                legacy_secret=legacy_secret,
                display=12 if not hidden else 8,
            ): (endpoint, query, scope)
            for endpoint, query, scope in jobs
        }

        for future in as_completed(futures):
            _, query, scope = futures[future]
            try:
                found = future.result()
            except Exception:
                continue

            for row in found:
                row["retrieval_scope"] = scope
                row["context_query"] = (
                    query if scope == "user_context" else ""
                )
            rows.extend(found)

    minimum = 30 if not hidden else 12
    if len(_dedupe(rows, 200)) < minimum:
        for query, scope in query_plan:
            found = search_content(
                "webkr",
                query,
                hub_id=hub_id,
                hub_secret=hub_secret,
                legacy_id=legacy_id,
                legacy_secret=legacy_secret,
                display=10 if not hidden else 6,
            )
            for row in found:
                row["retrieval_scope"] = scope
                row["context_query"] = (
                    query if scope == "user_context" else ""
                )
            rows.extend(found)

    scope_priority = {
        "user_context": 0,
        "preference": 1,
        "positive": 2,
        "negative": 2,
        "base": 3,
        "hidden_base": 3,
    }
    rows.sort(
        key=lambda item: scope_priority.get(
            str(item.get("retrieval_scope") or "base"),
            4,
        )
    )

    return _dedupe(
        rows,
        80 if not hidden else 40,
    )


def collect_visible_evidence(
    target: str,
    *,
    kind: str = "restaurant",
    context: dict[str, Any] | None = None,
    hub_id: str = "",
    hub_secret: str = "",
    legacy_id: str = "",
    legacy_secret: str = "",
) -> list[dict[str, Any]]:
    return _collect_parallel(
        target,
        kind=kind,
        hidden=False,
        context=context,
        hub_id=hub_id,
        hub_secret=hub_secret,
        legacy_id=legacy_id,
        legacy_secret=legacy_secret,
    )


def collect_hidden_evidence(
    target: str,
    *,
    kind: str = "restaurant",
    context: dict[str, Any] | None = None,
    hub_id: str = "",
    hub_secret: str = "",
    legacy_id: str = "",
    legacy_secret: str = "",
) -> list[dict[str, Any]]:
    return _collect_parallel(
        target,
        kind=kind,
        hidden=True,
        context=context,
        hub_id=hub_id,
        hub_secret=hub_secret,
        legacy_id=legacy_id,
        legacy_secret=legacy_secret,
    )
