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
        return (
            "hub",
            {"X-NCP-APIGW-API-KEY-ID": hub_id, "X-NCP-APIGW-API-KEY": hub_secret},
        )
    if legacy_id and legacy_secret:
        return (
            "legacy",
            {"X-Naver-Client-Id": legacy_id, "X-Naver-Client-Secret": legacy_secret},
        )
    return None


def local_search(query: str, *, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> tuple[list[dict[str, Any]], str]:
    auth = _auth(hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret)
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
        rows = [{
            "title": clean_html(item.get("title", "")),
            "category": clean_html(item.get("category", "")),
            "address": clean_html(item.get("roadAddress") or item.get("address", "")),
            "link": str(item.get("link") or "").strip(),
        } for item in items]
        return rows, "정상"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=256)
def _search_content_cached(
    endpoint: str,
    query: str,
    hub_id: str,
    hub_secret: str,
    legacy_id: str,
    legacy_secret: str,
    display: int,
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    auth = _auth(hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret)
    if not auth:
        return tuple()

    mode, headers = auth
    url = (
        f"https://naverapihub.apigw.ntruss.com/search/v1/{endpoint}"
        if mode == "hub"
        else f"https://openapi.naver.com/v1/search/{endpoint}.json"
    )
    params: dict[str, Any] = {"query": query, "display": display, "start": 1}
    if endpoint in {"blog", "cafearticle"}:
        params["sort"] = "sim"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=6)
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
            str(item.get("postdate") or ""),
        )
        for item in items
    )


def search_content(endpoint: str, query: str, *, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "", display: int = 8) -> list[dict[str, Any]]:
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
        key = str(row.get("link") or f"{row.get('title')}|{row.get('snippet')}").strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped[:limit]


def _query_plan(target: str, kind: str, hidden: bool) -> list[str]:
    if kind == "product":
        if hidden:
            return [
                f"{target} 반품 환불 불량",
                f"{target} 후회 고장 중고 판매",
            ]
        return [
            f"{target} 실사용 후기 장단점",
            f"{target} 리뷰 단점 성능",
        ]

    if hidden:
        return [
            f"{target} 예약 실패 웨이팅 포기",
            f"{target} 주차 포기 재방문 안",
        ]
    return [
        f"{target} 후기 분위기 서비스",
        f"{target} 웨이팅 예약 리뷰",
    ]


def _collect_parallel(
    target: str,
    *,
    kind: str,
    hidden: bool,
    hub_id: str = "",
    hub_secret: str = "",
    legacy_id: str = "",
    legacy_secret: str = "",
) -> list[dict[str, Any]]:
    queries = _query_plan(target, kind, hidden)
    primary_endpoints = ("blog", "cafearticle")
    jobs = [(endpoint, query) for query in queries for endpoint in primary_endpoints]
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                search_content,
                endpoint,
                query,
                hub_id=hub_id,
                hub_secret=hub_secret,
                legacy_id=legacy_id,
                legacy_secret=legacy_secret,
                display=7 if not hidden else 5,
            ): (endpoint, query)
            for endpoint, query in jobs
        }
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                continue

    # Blog/Cafe 결과가 부족할 때만 Web 검색 1회씩 fallback.
    minimum = 14 if not hidden else 6
    if len(_dedupe(rows, 100)) < minimum:
        for query in queries:
            rows.extend(
                search_content(
                    "webkr",
                    query,
                    hub_id=hub_id,
                    hub_secret=hub_secret,
                    legacy_id=legacy_id,
                    legacy_secret=legacy_secret,
                    display=6 if not hidden else 4,
                )
            )

    return _dedupe(rows, 40 if not hidden else 24)


def collect_visible_evidence(target: str, *, kind: str = "restaurant", hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> list[dict[str, Any]]:
    return _collect_parallel(
        target,
        kind=kind,
        hidden=False,
        hub_id=hub_id,
        hub_secret=hub_secret,
        legacy_id=legacy_id,
        legacy_secret=legacy_secret,
    )


def collect_hidden_evidence(target: str, *, kind: str = "restaurant", hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> list[dict[str, Any]]:
    return _collect_parallel(
        target,
        kind=kind,
        hidden=True,
        hub_id=hub_id,
        hub_secret=hub_secret,
        legacy_id=legacy_id,
        legacy_secret=legacy_secret,
    )
