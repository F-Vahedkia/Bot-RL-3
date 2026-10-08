# -*- coding: utf-8 -*-
# f15_scripts/news_refresh_daily.py
# Status in (Bot-RL-3): Completed

"""
Daily refresh wrapper for calendar cache — messages in English; Persian comments.

Examples:
python -m f15_scripts.news_refresh_daily ^
    --urls "https://example.com/thisweek.csv" ^
    -c f01_config/config.yaml
"""
from __future__ import annotations
import argparse, logging
from f02_utils.config_loader import load_config

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser(description="Daily refresh for news calendar cache.")
    p.add_argument("--urls", nargs="*", default=None, help="List of CSV URLs to download.")
    p.add_argument("--csvs", nargs="*", default=None, help="Local CSVs (optional).")
    p.add_argument("-c", "--config", default="f01_config/config.yaml", help="Config path.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def _urls_from_cfg(cfg):
    nf = ((cfg or {}).get("safety", {}) or {}).get("news_filter", {}) or {}
    return [str(u) for u in nf.get("urls", [])]

def main() -> int:
    args = parse_args()
    _setup_logging(args.log_level)
    cfg = load_config(args.config, enable_env_override=True)

    if not args.urls:
        args.urls = _urls_from_cfg(cfg)
    
    import subprocess, sys
    cmd = [sys.executable, "-m", "f15_scripts.news_build_cache", "-c", args.config]
    for u in (args.urls or []): cmd += ["--urls", u]
    for c in (args.csvs or []): cmd += ["--csvs", c]
    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
