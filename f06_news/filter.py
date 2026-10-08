# -*- coding: utf-8 -*-
# f06_news/filter.py
# Status in (Bot-RL-3): Completed
"""
NewsGate: freeze/reduce-risk based on economic calendar — messages in English; comments in Persian."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any
import pandas as pd

from .schemas import GateConfig, NewsEvent
from .runtime_store import NewsStore

# Persian: کمکی برای نگاشت نماد به ارزهای مرتبط (قابل‌گسترش)
_SYMBOL_CCY_MAP = {
    "XAUUSD": ["USD", "XAU"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "EURUSD": ["EUR", "USD"],
    "US30":   ["USD"],
}

def relevant_currencies_for_symbol(symbol: Optional[str], fallback: Optional[list[str]]) -> Optional[list[str]]:
    if symbol and symbol.upper() in _SYMBOL_CCY_MAP:
        return _SYMBOL_CCY_MAP[symbol.upper()]
    return fallback

@dataclass
class NewsGate:
    """Persian: دروازهٔ خبری — تعیین وضعیت Freeze / Reduce و دلیل آن، در هر لحظه t."""
    cfg: GateConfig
    store: NewsStore
    currencies: Optional[list[str]] = None  # Persian: اگر None، یعنی همهٔ ارزها
    symbol: Optional[str] = None            # Persian: برای نگاشت خودکار ارزهای مرتبط

    def status(self, t: pd.Timestamp) -> Dict[str, Any]:
        """English: Returns gate status at time t (UTC)."""
        if not self.cfg.enabled:
            return {"freeze": False, "reduce_risk": False, "reason": "disabled", "events": []}

        t = pd.to_datetime(t, utc=True)

        # Persian: ارزهای هدف (یا از نگاشت نماد)
        target_currencies = relevant_currencies_for_symbol(self.symbol, self.currencies)

        def event_window(ev: NewsEvent) -> tuple[pd.Timestamp, pd.Timestamp]:
            # Persian: اگر در خود رویداد پنجره آمده باشد از همان استفاده شود؛ وگرنه از cfg
            if ev.impact == "high":
                before = ev.window_before_min if ev.window_before_min is not None else self.cfg.high_before
                after  = ev.window_after_min  if ev.window_after_min  is not None else self.cfg.high_after
            elif ev.impact == "medium":
                before = ev.window_before_min if ev.window_before_min is not None else self.cfg.medium_before
                after  = ev.window_after_min  if ev.window_after_min  is not None else self.cfg.medium_after
            else:
                before = ev.window_before_min if ev.window_before_min is not None else self.cfg.low_before
                after  = ev.window_after_min  if ev.window_after_min  is not None else self.cfg.low_after
            start = ev.time_utc - pd.Timedelta(minutes=int(before))
            end   = ev.time_utc + pd.Timedelta(minutes=int(after))
            return start, end

        # Persian: برای جستجو، به‌جای پنجرهٔ متغیر، بیشینهٔ قبل/بعد را در نظر می‌گیریم
        max_before = max(self.cfg.high_before, self.cfg.medium_before, self.cfg.low_before)
        max_after  = max(self.cfg.high_after,  self.cfg.medium_after,  self.cfg.low_after)
        win_events = self.store.window(t - pd.Timedelta(minutes=max_after),
                                       t + pd.Timedelta(minutes=max_before))

        freeze = False
        reduce = False
        picked: list[dict] = []

        for ev in win_events:
            if target_currencies and ev.currency.upper() not in target_currencies:
                continue
            start, end = event_window(ev)
            if not (start <= t <= end):
                continue
            picked.append(ev.to_dict())
            if ev.impact == "high":
                freeze = True
            elif ev.impact == "medium":
                reduce = True

        reason = "ok" if (not freeze and not reduce) else ("freeze" if freeze else "reduce")
        return {
            "freeze": freeze,
            "reduce_risk": reduce,
            "reduce_scale": self.cfg.reduce_scale,  # NEW
            "reason": reason,
            "events": picked
        }
