# -*- coding: utf-8 -*-
# f06_news/runtime_store.py
# Status in (Bot-RL-3): Completed
"""
Runtime store & windowed queries for news events — messages in English; comments in Persian."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional
import bisect
import pandas as pd
from .schemas import NewsEvent

@dataclass
class NewsStore:
    """Persian: انبار رویداد خبری با ایندکس زمانی برای کوئری سریع."""
    # Persian: فهرست زمان‌ها برای جستجوی دودویی
    _times: List[pd.Timestamp]
    # Persian: خود رویدادها به همان ترتیب
    _events: List[NewsEvent]

    @staticmethod
    def from_events(events: Iterable[NewsEvent]) -> "NewsStore":
        evs = list(events)
        pairs = [(pd.to_datetime(e.time_utc, utc=True), e) for e in evs]
        pairs.sort(key=lambda x: (x[0].value, getattr(x[1], "currency", ""), getattr(x[1], "id", "")))
        times = [t for t, _ in pairs]
        evs_sorted = [e for _, e in pairs]
        return NewsStore(_times=times, _events=evs_sorted)

    def window(self, start: pd.Timestamp, end: pd.Timestamp) -> list[NewsEvent]:
        """Persian: همهٔ رویدادهای در بازهٔ [start, end]."""
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)
        i = bisect.bisect_left(self._times, start)
        out: list[NewsEvent] = []
        while i < len(self._times) and self._times[i] <= end:
            out.append(self._events[i])
            i += 1
        return out
