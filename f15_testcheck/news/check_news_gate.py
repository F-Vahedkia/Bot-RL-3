# -*- coding: utf-8 -*-
# f15_scripts/check_news_gate.py
# Status in (Bot-RL-3): Completed

"""
Check NewsGate over a time range — messages in English; comments in Persian.

Example For Runing:
python -m f15_scripts.check_news_gate ^
    --csv f03_data/news/calendar.csv ^
    --start "2025-08-20 00:00:00Z" --end "2025-08-21 23:59:00Z" ^
    --symbol XAUUSD ^
    -c f01_config/config.yaml
python -m f15_scripts.check_news_gate `
  --csv f03_data/news/calendar.parquet `
  --start "2025-08-20 00:00:00Z" --end "2025-08-21 23:59:00Z" `
  --symbol XAUUSD `
  -c f01_config/config.yaml

"""
from __future__ import annotations
import argparse
import pandas as pd

# Persian: اگر loader کانفیگ داری، می‌توانی این را وصل کنی
try:
    from f02_utils.config_loader import load_config as _load_cfg
except Exception:
    _load_cfg = None

from f06_news.schemas import GateConfig
from f06_news.providers.local_csv import LocalCSVProvider
from f06_news.runtime_store import NewsStore
from f06_news.filter import NewsGate

def _setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser(description="Check NewsGate over a time range.")
    p.add_argument("--csv", required=True, help="Path to local calendar CSV/Parquet.")
    p.add_argument("--start", required=True, help="Start time (UTC).")
    p.add_argument("--end", required=True, help="End time (UTC).")
    p.add_argument("--symbol", default=None, help="Symbol (e.g., XAUUSD).")
    p.add_argument("-c", "--config", default=None, help="Path to config.yaml (optional).")
    return p.parse_args()

def main() -> int:
    _setup_logging()
    args = parse_args()

    # Persian: پیکربندی Gate از فایل کانفیگ یا پیش‌فرض
    if args.config and _load_cfg:
        cfg = _load_cfg(args.config, enable_env_override=True)
        gate_cfg = GateConfig.from_config_dict(cfg)
    else:
        gate_cfg = GateConfig()  # Persian: پیش‌فرض‌های منطقی

    # Persian: بارگیری رویدادها و ساخت Store
    provider = LocalCSVProvider(args.csv)
    events = list(provider.load())
    store = NewsStore.from_events(events)
    gate = NewsGate(cfg=gate_cfg, store=store, symbol=args.symbol)

    t0 = pd.to_datetime(args.start, utc=True)
    t1 = pd.to_datetime(args.end, utc=True)

    print("Scanning NewsGate status...")
    t = t0
    step = pd.Timedelta(minutes=1)
    last_state = None

    while t <= t1:
        st = gate.status(t)
        # Persian: فقط زمانی چاپ کن که وضعیت عوض می‌شود (برای لاگ کوتاه‌تر)
        cur_state = (st["freeze"], st["reduce_risk"])
        if cur_state != last_state:
            print(f"{t.isoformat()} | freeze={st['freeze']} reduce={st['reduce_risk']} reason={st['reason']} events={len(st['events'])}")
            for ev in st["events"]:
                print(f"  - {ev['time_utc']} {ev['currency']} {ev['impact']}: {ev['title']}")
            last_state = cur_state
        t += step

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
