# -*- coding: utf-8 -*-
# f06_news/dataset.py
# Status in (Bot-RL-3): Completed
"""
News dataset loader/filters for Env — messages in English; Persian comments.
"""
from __future__ import annotations
from typing import Optional, Iterable, List
import pathlib
import pandas as pd

from .schemas import NewsEvent
from .runtime_store import NewsStore

def news_dir_path(cfg: Optional[dict]) -> pathlib.Path:
    """Persian: مسیر پوشهٔ خبر را از کانفیگ بگیر یا پیش‌فرض بده."""
    paths = (cfg or {}).get("paths", {}) if cfg else {}
    base = pathlib.Path(paths.get("data_dir", "f03_data"))
    news_dir = pathlib.Path(paths.get("news_dir", base / "news"))
    news_dir.mkdir(parents=True, exist_ok=True)
    return news_dir

def cache_path(news_dir: pathlib.Path) -> pathlib.Path:
    return news_dir / "calendar.parquet"

def save_cache(df: pd.DataFrame, news_dir: pathlib.Path) -> pathlib.Path:
    p = cache_path(news_dir)
    df.to_parquet(p)
    return p

def load_cache(news_dir: pathlib.Path) -> pd.DataFrame:
    p = cache_path(news_dir)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=["time_utc","currency","impact","title","window_before_min","window_after_min"])

def load_range(news_dir: pathlib.Path, start, end,
               currencies: Optional[Iterable[str]] = None,
               impacts: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Persian: بارگیری بازهٔ زمانی و فیلتر بر اساس ارزها/شدت‌ها."""
    df = load_cache(news_dir)
    if df.empty:
        return df
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    start = pd.to_datetime(start, utc=True)
    end   = pd.to_datetime(end, utc=True)
    df = df[(df["time_utc"] >= start) & (df["time_utc"] <= end)].copy()
    if currencies:
        cs = {c.upper() for c in currencies}
        df = df[df["currency"].str.upper().isin(cs)]
    if impacts:
        im = {s.lower() for s in impacts}
        df = df[df["impact"].str.lower().isin(im)]
    return df.sort_values("time_utc").reset_index(drop=True)

def to_store(df: pd.DataFrame) -> NewsStore:
    """Persian: DataFrame نرمال‌شده را به NewsStore تبدیل کن."""
    events: list[NewsEvent] = []
    for i, r in df.iterrows():
        # Persian: بستهٔ جزئیات اختیاری (اگر ستون هست و مقدار دارد)
        extra = {}
        for k in ("actual", "forecast", "previous", "revised", "detail"):
            v = r.get(k, pd.NA)
            if pd.notna(v) and str(v).strip() != "":
                extra[k] = str(v).strip()
        if not extra:
            extra = None

        ev = NewsEvent(
            id=f"cache:{i}",
            time_utc=pd.to_datetime(r["time_utc"], utc=True),
            currency=str(r["currency"]).upper(),
            impact=str(r["impact"]).lower(),  # type: ignore
            title=str(r["title"]),
            window_before_min=int(r.get("window_before_min")) if pd.notna(r.get("window_before_min")) else None,
            window_after_min=int(r.get("window_after_min")) if pd.notna(r.get("window_after_min")) else None,
            extra=extra,
        )
        events.append(ev)
    return NewsStore.from_events(events)

