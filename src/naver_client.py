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
        return "hub", {"X-NCP-APIGW-API-KEY-ID": hub_id, "X-NCP-APIGW-API-KEY