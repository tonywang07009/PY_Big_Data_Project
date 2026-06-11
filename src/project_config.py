"""Legacy scope constants for the older six-county ratio delivery.

The formal mainline now uses all available county labels through
`train_donation_count_xgboost_ensemble.py`. These helpers remain only for
legacy ratio/clustering scripts that still need the historical six-county
scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


COUNTY_NAMES = {
    0: "南投縣",
    1: "台中市",
    2: "台北市",
    3: "台南市",
    4: "台東縣",
    5: "嘉義市",
    6: "嘉義縣",
    7: "基隆市",
    8: "宜蘭縣",
    9: "屏東縣",
    10: "彰化縣",
    11: "新北市",
    12: "新竹市",
    13: "新竹縣",
    14: "桃園市",
    15: "花蓮縣",
    16: "苗栗縣",
    17: "雲林縣",
    18: "高雄市",
}


@dataclass(frozen=True)
class ScopeCounty:
    english_name: str
    chinese_name: str
    code: int | None


SCOPE_COUNTIES = [
    ScopeCounty("Taichung", "台中市", 1),
    ScopeCounty("Taipei", "台北市", 2),
    ScopeCounty("Hualien", "花蓮縣", 15),
    ScopeCounty("Yunlin", "雲林縣", 17),
    ScopeCounty("Kaohsiung", "高雄市", 18),
    ScopeCounty("Lienchiang", "連江縣", None),
]


SCOPE_CODES = tuple(c.code for c in SCOPE_COUNTIES if c.code is not None)
SCOPE_NAMES = tuple(c.chinese_name for c in SCOPE_COUNTIES)
EXPECTED_MISSING_SCOPE = tuple(c.chinese_name for c in SCOPE_COUNTIES if c.code is None)
COUNTY_ENGLISH_NAMES = {c.code: c.english_name for c in SCOPE_COUNTIES if c.code is not None}


def add_county_name(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a stable county_name column."""
    out = df.copy()
    out["county_name"] = out["county_label"].map(COUNTY_NAMES)
    return out


def filter_scope_counties(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the historical six-county scope available in the encoded dataset."""
    scoped = df[df["county_label"].isin(SCOPE_CODES)].copy()
    return add_county_name(scoped).reset_index(drop=True)


def scope_readiness(raw_county_labels: Iterable[int]) -> dict:
    """Build the data-readiness summary for the historical six-county scope."""
    available_codes = set(int(c) for c in raw_county_labels)
    available_scope = [
        {"county_code": c.code, "county_name": c.chinese_name}
        for c in SCOPE_COUNTIES
        if c.code is not None and c.code in available_codes
    ]
    missing_scope = [
        {"county_code": c.code, "county_name": c.chinese_name}
        for c in SCOPE_COUNTIES
        if c.code is None or c.code not in available_codes
    ]
    return {
        "requested_counties": [
            {"county_code": c.code, "county_name": c.chinese_name, "english_name": c.english_name}
            for c in SCOPE_COUNTIES
        ],
        "available_scope_counties": available_scope,
        "missing_scope_counties": missing_scope,
        "scope_count": len(SCOPE_COUNTIES),
        "available_scope_count": len(available_scope),
        "missing_scope_count": len(missing_scope),
    }
