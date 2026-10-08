# f04_features/indicators/__init__.py
# -*- coding: utf-8 -*-

r"""
Entry اصلی پکیج اندیکاتورها + CLI
- خواندن processed از data_handler
- اعمال Specها بر اساس config.features.indicators
- ذخیره به صورت augment یا features_only
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import pandas as pd
import argparse

from ...f02_utils.functions.parser import parse_spec

#from ..feature_engine import run_specs_v2
def run_specs_v2(*args, **kwargs):
    from f04_features.OLD.feature_engine import run_specs_v2 as _impl
    return _impl(*args, **kwargs)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- New Added ----------------------------------------------------- 040607
"""
Entry/API برای indicators (Bot-RL-3)
- اکسپورت parser/engine/registry (نسخهٔ v2 افزایشی)
- CLI سبک برای اجرا با `python -m f04_features.indicators` (نیازمند __main__.py شیم)
"""
# -----------------------------
# API re-exports (اختیاری)
# -----------------------------
__all__ = [
    "run_specs_v2",
    "parse_spec_v2",
    #"list_all_indicators",
]
# -----------------------------
# CLI سبک
# -----------------------------
def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def main() -> int:
    """
    اجرای سبک:
      - اگر --list بدهید، فقط لیست اندیکاتورها را چاپ می‌کند.
      - اگر --input و --spec بدهید، فایل پردازش‌شدهٔ data_handler را می‌خواند و Specها را اعمال می‌کند،
        سپس خروجی را در --out ذخیره می‌کند (Parquet/CSV بر اساس پسوند).
    """
    p = argparse.ArgumentParser(description="Indicators CLI (Bot-RL-3)")
    #p.add_argument("--symbol", required=True)
    p.add_argument("--list", action="store_true", help="List available indicators and exit")
    p.add_argument("--input", type=str, help="Processed dataset file (parquet/csv)")
    p.add_argument("--base-tf", type=str, default="M1", help="Base timeframe (e.g., M1/H1)")
    p.add_argument("--spec", nargs="*", default=[], help="Indicator specs to apply (e.g., \"rsi_zone(period=14)@H1\")")
    p.add_argument("--out", type=str, help="Output file (parquet/csv)")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    _setup_logging(args.log_level)

    if args.list:
        from ..OLD.feature_registry import list_all_indicators 
        items = list_all_indicators(include_legacy=True)
        for k, src in sorted(items.items()):
            print(f"{k:20s}  [{src}]")
        return 0

    if not args.input:
        logger.error("No --input provided. Use --list or provide --input/--spec/--out.")
        return 2

    if not args.spec:
        logger.error("No --spec provided. Use --list to see available indicators.")
        return 2

    # خواندن دیتاست پردازش‌شده
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input, parse_dates=True, index_col=0)
    else:
        logger.error("Unsupported input extension. Use .parquet or .csv")
        return 2

    # اعمال Specها
    df_out = run_specs_v2(df, specs=args.spec, base_tf=args.base_tf)

    # ذخیرهٔ خروجی
    if args.out:
        if args.out.lower().endswith(".parquet"):
            df_out.to_parquet(args.out)
        elif args.out.lower().endswith(".csv"):
            df_out.to_csv(args.out)
        else:
            logger.error("Unsupported output extension. Use .parquet or .csv")
            return 2
        logger.info("Saved output to: %s (rows=%d, cols=%d)", args.out, len(df_out), len(df_out.columns))
    else:
        # اگر خروجی مشخص نشده، فقط شکل دیتافریم را گزارش کن
        logger.info("Done. Output shape: %s", (len(df_out), len(df_out.columns)))
    return 0

