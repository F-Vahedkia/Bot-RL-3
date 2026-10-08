# -*- coding: utf-8 -*-
# f06_news/provider_base.py
# Status in (Bot-RL-3): Completed
"""
Provider base interfaces for news feeds — messages in English; comments in Persian."""
from __future__ import annotations
from typing import Protocol, Iterable
import pandas as pd
from .schemas import NewsEvent

class NewsProvider(Protocol):
    """Persian: رابط مشترک برای همهٔ Providerهای خبر/تقویم."""
    def load(self) -> Iterable[NewsEvent]:
        ...

def _normalize_currency(s: str) -> str:
    return (s or "").strip().upper()

def _normalize_impact(x: str) -> str:
    v = (x or "").strip().lower()
    if v in ("high", "medium", "low"):
        return v
    # Persian: نگاشت ساده از امتیاز/ستاره به impact
    mapping = {"3": "high", "2": "medium", "1": "low", "***": "high", "**": "medium", "*": "low"}
    return mapping.get(v, "low")

def _coerce_utc(ts) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True)
