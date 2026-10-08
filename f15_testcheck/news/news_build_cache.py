# -*- coding: utf-8 -*-
# f15_scripts/news_build_cache.py
# Status in (Bot-RL-3): Completed

"""
Build/refresh local calendar cache — messages in English; Persian comments.

Examples To Run:
python -m f15_scripts.news_build_cache `
  --urls "https://example.com/thisweek.csv" `
  --out-dir f03_data/news `
  -c f01_config/config.yaml
"""
from __future__ import annotations

import argparse, logging, pathlib
import pandas as pd

try:
    from f02_utils.config_loader import load_config as _load_cfg
except Exception:
    _load_cfg = None

from f06_news.providers.forexfactory_http import ForexFactoryHTTPProvider
from f06_news.normalizer import normalize_forexfactory_csv, concat_and_dedupe
from f06_news.dataset import news_dir_path, save_cache


def _prune_old(dirpath: pathlib.Path, days: int) -> int:
    if days <= 0: return 0
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    removed = 0
    for p in dirpath.rglob("*"):
        if p.is_file():
            mtime = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
            if mtime < cutoff:
                try:
                    p.unlink(missing_ok=True); removed += 1
                except Exception:
                    pass
    logging.info("Pruned %d old files (> %d days) in %s", removed, days, dirpath)
    return removed

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser(description="Download and normalize calendar CSV(s), then build Parquet cache.")
    p.add_argument("--urls", nargs="*", default=None, help="List of CSV URLs to download.")
    p.add_argument("--csvs", nargs="*", default=None, help="Local CSV paths (optional).")
    p.add_argument("--out-dir", default=None, help="Output news dir (default: from config or f03_data/news).")
    p.add_argument("-c", "--config", default=None, help="Path to config.yaml (optional).")
    p.add_argument("--tz-hint", default="UTC", help="Calendar timezone hint (e.g., Europe/London)")
    p.add_argument("--retention-days", type=int, default=0,
                    help="Delete raw/cache files older than N days (0=disabled)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    _setup_logging(args.log_level)

    cfg = _load_cfg(args.config, enable_env_override=True) if (args.config and _load_cfg) else {}
    news_dir = pathlib.Path(args.out_dir) if args.out_dir else news_dir_path(cfg)
    raw_dir = news_dir / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[pathlib.Path] = []

    # 1) Download URLs
    if args.urls:
        provider = ForexFactoryHTTPProvider(urls=list(args.urls), dest_dir=raw_dir)
        saved_paths.extend(provider.fetch())

    # 2) Add local CSVs
    if args.csvs:
        for c in args.csvs:
            p = pathlib.Path(c)
            if p.exists():
                saved_paths.append(p)

    if not saved_paths:
        logging.error("No input (URLs/CSVs). Nothing to do.")
        return 2

    # 3) Normalize all CSVs (FF)
    frames = []
    for p in saved_paths:
        try:
            logging.info("Normalizing: %s", p)
            frames.append(normalize_forexfactory_csv(str(p), tz_hint=args.tz_hint))

        except Exception as ex:
            logging.warning("Normalization failed for %s (%s). Skipped.", p, ex)

    if not frames:
        logging.error("No valid frames after normalization.")
        return 3

    # 4) Concat + dedupe + save Parquet cache
    df = concat_and_dedupe(frames)
    out = save_cache(df, news_dir)
    logging.info("Saved cache: %s | rows=%d", out, len(df))

    if args.retention_days > 0:
        _prune_old(raw_dir, args.retention_days)
        _prune_old(news_dir, args.retention_days)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
