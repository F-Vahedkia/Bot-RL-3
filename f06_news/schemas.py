# -*- coding: utf-8 -*-
# f06_news/schemas.py
# Status in (Bot-RL-3): Completed
"""
News data models (Bot-RL-3) — messages in English; comments in Persian."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import pandas as pd

# Persian: شدت خبر — منطبق با تقویم‌های متداول
Impact = Literal["low", "medium", "high"]

@dataclass(frozen=True)
class NewsEvent:
    """Persian: مدل رویداد خبری استاندارد شده."""
    id: str
    time_utc: pd.Timestamp
    currency: str
    impact: Impact
    title: str
    # Persian: اگر در منبع CSV/JSON داده شود، همین‌ها استفاده می‌شوند؛ وگرنه از GateConfig می‌آیند
    window_before_min: Optional[int] = None
    window_after_min: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "time_utc": pd.to_datetime(self.time_utc, utc=True),
            "currency": str(self.currency),
            "impact": str(self.impact),
            "title": str(self.title),
            "window_before_min": self.window_before_min,
            "window_after_min": self.window_after_min,
            "extra": self.extra or {},
        }


@dataclass(frozen=True)
class GateConfig:
    """پیکربندی دروازهٔ خبری (News Gate)"""
    enabled: bool = True
    # پنجره‌های پیش/پس از خبر برای شدت‌های مختلف (دقیقه)
    high_before: int = 30
    high_after: int = 15
    medium_before: int = 15
    medium_after: int = 10
    low_before: int = 0
    low_after: int = 0
    # شدت کاهش ریسک در رویدادهای medium
    reduce_scale: float = 0.5
    # فیلتر ارزها (اختیاری)؛ اگر خالی باشد یعنی همهٔ ارزها
    currencies: Optional[list[str]] = None

    @staticmethod
    def from_config_dict(cfg: dict) -> "GateConfig":
        nf = ((cfg or {}).get("safety", {}) or {}).get("news_filter", {}) or {}
        
        # Persian: علیاس‌ها برای medium (اگر جای "reduce" از "freeze" استفاده شده باشد)
        m_before = nf.get("medium_impact_reduce_minutes_before", nf.get("medium_impact_freeze_minutes_before", 15))
        m_after  = nf.get("medium_impact_reduce_minutes_after",  nf.get("medium_impact_freeze_minutes_after", 10))
        
        return GateConfig(
            enabled=bool(nf.get("enabled", True)),
            high_before=int(nf.get("high_impact_freeze_minutes_before", 30)),
            high_after=int(nf.get("high_impact_freeze_minutes_after", 15)),
            medium_before=int(m_before),
            medium_after=int(m_after),
            low_before=int(nf.get("low_impact_window_before", 0)),
            low_after=int(nf.get("low_impact_window_after", 0)),
            reduce_scale=float(nf.get("reduce_risk_scale", 0.5)),
            currencies=(list(nf.get("currencies")) if nf.get("currencies") else None),
        )
