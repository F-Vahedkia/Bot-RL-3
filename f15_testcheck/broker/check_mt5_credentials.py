# -*- coding: utf-8 -*-
# f15_scripts/check_mt5_credentials.py
# Status in (Bot-RL-3): Completed

"""
f15_scripts/check_mt5_credentials.py
تستِ اتصال MT5 با بارگذاری کانفیگِ صریح (بدون Override از ENV)

کارها:
1) لود config (به‌طور پیش‌فرض: f01_config/config.yaml) با enable_env_override=False
2) چاپ mt5_credentials (login/password/server/terminal_path)
3) initialize کانکتور و health_check اختیاری روی یک نماد (مثل XAUUSD)
"""

from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

# --- کشف ریشهٔ ریپو و افزودن به sys.path ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = None
for cand in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]:
    if (cand / "f02_utils").exists() and (cand / "f03_data").exists():
        PROJECT_ROOT = cand
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- ایمپورت ماژول‌ها ---
from f02_utils.config_loader import load_config
from f03_data.mt5_connector import MT5Connector  # اگر نام فایل شما متفاوت است، این ایمپورت را هماهنگ کنید.

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def main() -> int:
    parser = argparse.ArgumentParser(description="بررسی کرِدهای MT5 و تست اتصال (بدون Override از ENV).")
    parser.add_argument("-c", "--config", type=str, default=str(PROJECT_ROOT / "f01_config" / "config.yaml"),
                        help="مسیر فایل کانفیگ")
    parser.add_argument("--symbol", type=str, default="XAUUSD",
                        help="نماد برای health_check (اختیاری)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="سطح لاگ: DEBUG/INFO/WARN/ERROR")
    args = parser.parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("check_mt5")

    try:
        # 1) بارگذاری صریح کانفیگ، بدون ENV override
        cfg = load_config(args.config, enable_env_override=False)
        creds = (((cfg.get("executor") or {}).get("mt5_credentials")) or {})
        log.info("mt5_credentials from config: %s", {
            "login": creds.get("login"),
            "server": creds.get("server"),
            "terminal_path": creds.get("terminal_path") or creds.get("path"),
            "password_is_set": bool(creds.get("password")),
        })

        # 2) ساخت کانکتور با همین کانفیگ و تلاش برای initialize
        conn = MT5Connector(config=cfg)
        ok = conn.initialize()
        if not ok:
            log.critical("Connection/login to MT5 failed. Please check login/password/server/terminal_path.")
            return 2

        # 3) گزارش سلامت (اختیاری: چک تیک یک نماد برای اندازه‌گیری تاخیر)
        health = conn.health_check(sample_symbol=args.symbol)
        log.info("Health Report: %s", health)

        # 4) پایان
        conn.shutdown()
        log.info("✅ The test was successful.")
        return 0

    except Exception as ex:
        logging.getLogger("check_mt5").exception("error: %s", ex)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
