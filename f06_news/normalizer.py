# -*- coding: utf-8 -*-
# f06_news/normalizer.py
# Status in (Bot-RL-3): Completed
"""
Normalize diverse calendar CSVs to the canonical schema — messages in English; Persian comments.

Canonical columns:
- time_utc (UTC timestamp)
- currency (e.g., USD, EUR, GBP)
- impact  ("high" | "medium" | "low")
- title   (event name)
- window_before_min (optional, int)
- window_after_min  (optional, int)
- extra (dict-like; optional)
"""
from __future__ import annotations

from typing import Optional, Iterable, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Persian: نگاشت شدت اثر (Impact) به low/medium/high
_IMPACT_MAP = {
    "high": "high", "medium": "medium", "low": "low",
    "high impact expected": "high", "medium impact expected": "medium",
    "low impact expected": "low",
    "***": "high", "**": "medium", "*": "low",
    "3": "high", "2": "medium", "1": "low",
}

def _to_impact(s: Any) -> str:
    v = str(s or "").strip().lower()
    return _IMPACT_MAP.get(v, "low")

def _to_currency(s: Any) -> str:
    return str(s or "").strip().upper()

def _tz_to_utc(ts: Any, date_col: Optional[str], time_col: Optional[str], tz_hint: str = "UTC") -> pd.Timestamp:
    """
    Persian: از ستون‌های تاریخ/زمان ورودی، زمان UTC بساز.
    - اگر ts مستقیم زمان باشد: همان را به UTC تبدیل کن.
    - اگر فقط date/time داده شده باشد: آن‌ها را ترکیب کن.
    """
    if ts is not None and (isinstance(ts, str) or hasattr(ts, "isoformat")):
        try:
            return pd.to_datetime(ts, utc=True).tz_convert("UTC")  # اگر با tz همراه باشد
        except Exception:
            try:
                return pd.to_datetime(ts, utc=True)  # بدون tz → UTC فرض می‌کنیم
            except Exception:
                pass
    # Persian: ترکیب date/time
    if date_col and time_col:
        def _combine(row):
            d = str(row[date_col]).strip()
            t = str(row[time_col]).strip()
            if not t or t.lower() in ("", "all day", "tentative", "n/a"):
                t = "00:00"
            # Persian: اگر تایم‌زون داده نشده باشد، با tz_hint تفسیر و سپس به UTC تبدیل می‌کنیم
            try:
                local = pd.to_datetime(f"{d} {t}").tz_localize(tz_hint)
            except Exception:
                local = pd.to_datetime(f"{d} {t}", utc=True)  # fallback
            return local.tz_convert("UTC")
        return _combine
    # Persian: ناکافی
    raise ValueError("Cannot resolve timestamp to UTC; please provide proper columns or pre-parse.")

def normalize_forexfactory_csv(path: str, *, tz_hint: str = "UTC",
                               default_before_high: int = 30, default_after_high: int = 15,
                               default_before_medium: int = 15, default_after_medium: int = 10,
                               default_before_low: int = 0, default_after_low: int = 0) -> pd.DataFrame:
    """
    Persian: نرمال‌سازی CSV فارکس‌فکتوری (ستون‌ها معمولاً: Date, Time, Currency, Impact, Event, ...).
    """
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}

    # Persian: ستون‌های رایج
    date_col = lower.get("date")
    time_col = lower.get("time")

    curr_candidates = ["currency","curr","cur","ccy","cur.","country","ccy.","currencies"]
    curr_col = next((lower.get(c) for c in curr_candidates if lower.get(c)), None)
    imp_col  = lower.get("impact") or lower.get("impact (actual)") or "Impact"
    title_col= lower.get("event") or lower.get("title") or "Event"

    # Persian: اگر time_utc مستقیماً در فایل بود
    time_utc_col = lower.get("time_utc")
    if time_utc_col:
        ts = pd.to_datetime(df[time_utc_col], utc=True)
    else:
        combiner = _tz_to_utc(None, date_col, time_col, tz_hint=tz_hint)
        ts = df.apply(combiner, axis=1)  # type: ignore

    # Persian: ساخت اسکیمای واحد
    out = pd.DataFrame({
        "time_utc": pd.to_datetime(ts, utc=True),
        "currency": df[curr_col].map(_to_currency) if curr_col in df.columns else "",
        "impact": df[imp_col].map(_to_impact) if imp_col in df.columns else "low",
        "title": df[title_col].astype(str) if title_col in df.columns else "",
    })
    out["currency"] = out["currency"].astype(str).str.strip()
    out.loc[out["currency"]=="", "currency"] = pd.NA

    # --- OPTIONAL EXTRA FIELDS (Actual/Forecast/Previous/Revised/Detail) --- Start 040609
    # Persian: کشف نام‌های رایج ستون‌ها
    cands = {
        "actual":   ["actual", "act", "actual*", "result"],
        "forecast": ["forecast", "fcst", "exp", "expected"],
        "previous": ["previous", "prior", "prev"],
        "revised":  ["revised", "revision"],
        "detail":   ["detail", "notes", "comment"],
    }
    for key, names in cands.items():
        col = next((lower.get(n) for n in names if lower.get(n)), None)
        if col and col in df.columns:
            out[key] = df[col].astype(str).str.strip()
        else:
            out[key] = pd.NA
    # ---------------------------------------------------------------------- End 040609

    # Persian: پنجره‌ها را بر اساس شدت پیش‌فرض بده؛ اگر ستون‌های مخصوص بودند، از همان بخوان
    bcol = lower.get("window_before_min")
    acol = lower.get("window_after_min")
    if bcol in df.columns: out["window_before_min"] = pd.to_numeric(df[bcol], errors="coerce")
    else:
        out["window_before_min"] = out["impact"].map({"high": default_before_high, "medium": default_before_medium, "low": default_before_low})
    if acol in df.columns: out["window_after_min"] = pd.to_numeric(df[acol], errors="coerce")
    else:
        out["window_after_min"] = out["impact"].map({"high": default_after_high, "medium": default_after_medium, "low": default_after_low})

    # Persian: دسته‌بندی نهایی
    out = out.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    return out

def concat_and_dedupe(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Persian: تجمیع چند فایل نرمال‌شده و حذف رکوردهای تکراری.
    معیار تکراری: (time_utc, currency, title)
    """
    dfs = [f for f in frames if f is not None and len(f) > 0]
    if not dfs:
        return pd.DataFrame(columns=["time_utc","currency","impact","title","window_before_min","window_after_min"])
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.drop_duplicates(subset=["time_utc","currency","title"]).sort_values("time_utc").reset_index(drop=True)
    return df
