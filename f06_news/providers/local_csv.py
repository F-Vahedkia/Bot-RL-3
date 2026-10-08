# -*- coding: utf-8 -*-
# f06_news/providers/local_csv.py
# Status in (Bot-RL-3): Completed

"""Local CSV/Parquet provider for economic calendar — messages in English; comments in Persian."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Union
import pandas as pd
from ..schemas import NewsEvent
from ..provider_base import _normalize_currency, _normalize_impact, _coerce_utc

@dataclass
class LocalCSVProvider:
    """
    Persian: Provider محلی که یک CSV/Parquet با ستون‌های زیر را می‌خواند:
      - time_utc: زمان رویداد (UTC)
      - currency: کُد ارز (USD, EUR, GBP, JPY, ...)
      - impact: شدت (high/medium/low یا نگاشت امتیازی)
      - title: عنوان
      - window_before_min / window_after_min (اختیاری)
    """
    path: str

    def load(self) -> Iterable[NewsEvent]:
        if self.path.lower().endswith(".parquet"):
            df = pd.read_parquet(self.path)
        else:
            df = pd.read_csv(self.path)
        # Persian: نام ستون‌ها را نرم می‌کنیم
        cols = {c.lower().strip(): c for c in df.columns}
        tcol = cols.get("time_utc") or cols.get("time") or list(df.columns)[0]
        ccol = cols.get("currency") or "currency"
        icol = cols.get("impact") or "impact"
        ttl  = cols.get("title") or "title"
        bcol = cols.get("window_before_min") or "window_before_min"
        acol = cols.get("window_after_min") or "window_after_min"

        out: list[NewsEvent] = []
        for i, row in df.iterrows():
            ts = _coerce_utc(row[tcol])
            cur = _normalize_currency(str(row.get(ccol, "")))
            imp = _normalize_impact(str(row.get(icol, "")))
            tit = str(row.get(ttl, "")).strip()
            before = row.get(bcol, None)
            after = row.get(acol, None)
            ev = NewsEvent(
                id=f"local:{i}",
                time_utc=ts,
                currency=cur,
                impact=imp,  # type: ignore
                title=tit,
                window_before_min=int(before) if pd.notna(before) else None,
                window_after_min=int(after) if pd.notna(after) else None,
                extra=None,
            )
            out.append(ev)
        return out
