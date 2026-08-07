from __future__ import annotations

import re
from typing import Any

import requests

TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    return TAG_RE.sub("", str(text or "")).strip()


def local_search(query: str, *, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> tuple[list[dict[str, Any]], str]:
    if hub_id and hub_secret:
        url = "https://naverapihub.apigw.ntruss.com/search/v1/local"
        headers = {"X-NCP-APIGW-API-KEY-ID": hub_id, "X-NCP-APIGW-API-KEY": hub_secret}
    elif legacy_id and legacy_secret:
        url = "https://openapi.naver.com/v1/search/local.json"
        headers = {"X-Naver-Client-Id": legacy_id, "X-Naver-Client-Secret": legacy_secret}
    else:
        return [], "NAVER Secret 미설정"

    try:
        response = requests.get(url, headers=headers, params={"query": query, "display": 5, "sort": "comment"}, timeout=8)
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


def search_content(endpoint: str, query: str, *, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "", display: int = 10) -> list[dict[str, Any]]:
    if hub_id and hub_secret:
        url = f"https://naverapihub.apigw.ntruss.com/search/v1/{endpoint}"
        headers = {"X-NCP-APIGW-API-KEY-ID": hub_id, "X-NCP-APIGW-API-KEY": hub_secret}
    elif legacy_id and legacy_secret:
        url = f"https://openapi.naver.com/v1/search/{endpoint}.json"
        headers = {"X-Naver-Client-Id": legacy_id, "X-Naver-Client-Secret": legacy_secret}
    else:
        return []

    params: dict[str, Any] = {"query": query, "display": display, "start": 1}
    if endpoint in {"blog", "cafearticle"}:
        params["sort"] = "sim"
    try:
        response = requests.get(url, headers=headers, params=params, timeout=8)
        response.raise_for_status()
        items = response.json().get("items", [])
    except Exception:
        return []

    rows = []
    for item in items:
        rows.append({
            "source": endpoint,
            "query": query,
            "title": clean_html(item.get("title", "")),
            "snippet": clean_html(item.get("description", "")),
            "link": str(item.get("link") or "").strip(),
            "post_date": str(item.get("postdate") or ""),
        })
    return rows


def collect_visible_evidence(target: str, *, hub_id: str = "", hub_secret: str = "", legacy_id: str = "", legacy_secret: str = "") -> list[dict[str, Any]]:
    queries = [f"{target} 후기", f"{target} 리뷰", f"{target} 분위기", f"{target} 웨이팅"]
    rows: list[dict[str, Any]] = []
    for query in queries:
        for endpoint in ("blog", "cafearticle", "webkr"):
            rows.extend(search_content(endpoint, query, hub_id=hub_id, hub_secret=hub_secret, legacy_id=legacy_id, legacy_secret=legacy_secret, display=8))

    deduped = []
    seen = set()
    for row in rows:
        key = row.get("link") or f"{row.get('title')}|{row.get('snippet')}"
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped[:60]
