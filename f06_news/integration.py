# -*- coding: utf-8 -*-
# f06_news/integration.py
# Status in (Bot-RL-3): Completed
"""
Build a NewsGate from config/cache — logs English, comments Persian."""
from __future__ import annotations
from typing import Optional
import logging
from f06_news.schemas import GateConfig
from f06_news.dataset import news_dir_path, load_cache, to_store
from f06_news.filter import NewsGate

#from f02_utils.config_loader import load_config
#from f06_news.integration import make_news_gate

log = logging.getLogger(__name__)
#cfg = load_config("f01_config/config.yaml", enable_env_override=True)
#gate = make_news_gate(cfg, symbol="XAUUSD")  # یا هر نماد

def make_news_gate(cfg: Optional[dict], symbol: Optional[str] = None) -> Optional[NewsGate]:
    """Persian: از کانفیگ و کش، NewsGate بساز (اگر غیرفعال بود → None)."""
    gate_cfg = GateConfig.from_config_dict(cfg or {})
    if not gate_cfg.enabled:
        log.info("News filter disabled in config.")
        return None
    news_dir = news_dir_path(cfg)
    df = load_cache(news_dir)
    if df.empty:
        log.warning("News cache is empty. Gate will be inactive.")
        return None
    store = to_store(df)
    return NewsGate(cfg=gate_cfg, store=store, symbol=symbol)
