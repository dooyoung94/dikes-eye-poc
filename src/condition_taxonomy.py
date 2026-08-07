from __future__ import annotations

import re
from typing import Any


CONDITION_RULES: dict[str, dict[str, Any]] = {
    "price_value": {
        "label": "가격·가성비",
        "aliases": ["가격", "가성비", "비용", "금액", "예산", "저렴", "비싸", "할인"],
    },
    "noise_atmosphere": {
        "label": "분위기·소음",
        "aliases": ["분위기", "조용", "조용함", "소음", "대화", "감성", "로맨틱", "