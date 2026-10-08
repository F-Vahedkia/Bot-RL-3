# -*- coding: utf-8 -*-
# f15_scripts/check_dry_run_config.py
# Status in (Bot-RL-3): Completed

#!/usr/bin/env python
"""
تستِ خشک (Dry Run) برای بررسی سلامت کانفیگ پروژه Bot-RL-1

کارهایی که انجام می‌دهد:
1) بارگذاری امنِ کانفیگ (به‌طور پیش‌فرض: f01_config/config.yaml) با پشتیبانی .env و ENV.
2) ایجاد مسیرهای موردنیاز (logs/reports/cache/models/config_versions/tmp) در صورت نبودن (ensure_dirs).
3) چاپ خلاصه‌ای از کلیدهای حیاتی (acceptance_gates، risk، executor.canary_deployment، self_optimize).
4) ذخیرهٔ یک نسخهٔ آرشیوی از کانفیگ با برچسب انتخابی (پیش‌فرض: tag="dryrun") برای اطمینان از کارکرد نسخه‌گذاری.
5) در صورت درخواست، خلاصهٔ «گیت‌های پذیرش مؤثر» برای یک نماد خاص را چاپ می‌کند (با لحاظِ per_symbol_overrides).

روش اجرا (از ریشهٔ ریپو):
    python scripts/dry_run_config_check.py
گزینه‌های مفید:
    -c/--config         مسیر فایل کانفیگ (پیش‌فرض f01_config/config.yaml)
    --env-prefix        پیشوند متغیرهای محیطی (پیش‌فرض BOT_)
    --no-env-override   عدم اعمال Override از ENV
    --prefix            پیشوند نام فایل نسخهٔ خروجی (پیش‌فرض: "prod_")
    --tag               برچسب نسخهٔ خروجی (پیش‌فرض: "dryrun")
    --symbol            نماد برای گزارش گیت‌های مؤثر (مثلاً: XAUUSD)
"""

from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

# اطمینان از دسترسی به ماژول‌های پروژه (افزودن ریشهٔ ریپو به sys.path)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "f02_utils").exists() else SCRIPT_DIR.parent  # scripts/.. → ریشهٔ ریپو
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# اکنون می‌توانیم لودر کانفیگ را ایمپورت کنیم
from f02_utils.config_loader import (
    ConfigLoader,
    load_config,
    save_config_versioned,
)

# ---------------------------
# تنظیمات لاگ
# ---------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("dry_run")

# ---------------------------
# کمک‌تابع: ادغام گیت‌های پذیرش با لحاظ Override نماد
# (هم‌راستا با منطق self_optimizer_2ok)
# ---------------------------
def merge_accept_gates_for_symbol(cfg: Dict[str, Any], symbol: str) -> Dict[str, float]:
    ev = cfg.get("evaluation", {}) or {}
    base_gates = ev.get("acceptance_gates", {}) or {}

    per_sym = (cfg.get("per_symbol_overrides", {}) or {}).get(symbol, {}) or {}
    sym_ev = per_sym.get("evaluation", {}) or {}
    sym_gates = sym_ev.get("acceptance_gates", {}) or {}

    merged = {**base_gates, **sym_gates}

    # سازگاری نام‌ها: max_dd_max → max_drawdown_max
    if "max_drawdown_max" not in merged and "max_dd_max" in merged:
        merged["max_drawdown_max"] = merged["max_dd_max"]

    # تبدیل به float برای چاپ شفاف
    out = {
        "sharpe_min": float(merged.get("sharpe_min", 0.0)),
        "winrate_min": float(merged.get("winrate_min", 0.0)),
        "max_drawdown_max": float(merged.get("max_drawdown_max", 0.0)),
    }
    return out

# ---------------------------
# اجرای اصلی
# ---------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry Run: بررسی سلامت کانفیگ، ساخت مسیرها و نسخه‌گذاری (بدون اجرای معامله/آموزش)."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=str(PROJECT_ROOT / "f01_config" / "config.yaml"),
        help="مسیر فایل کانفیگ (پیش‌فرض: f01_config/config.yaml)"
    )
    parser.add_argument(
        "--env-prefix",
        type=str,
        default="BOT_",
        help="پیشوند متغیرهای محیطی برای Override (پیش‌فرض: BOT_)"
    )
    parser.add_argument(
        "--no-env-override",
        action="store_true",
        help="اگر ست شود، Override از ENV اعمال نمی‌شود."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="prod_",
        help='پیشوند نام فایل نسخهٔ خروجی (مثال: "prod_")'
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="dryrun",
        help='برچسب نسخهٔ خروجی (مثال: "dryrun")'
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="اگر نماد بدهید (مثل XAUUSD)، گیت‌های پذیرش مؤثر برای آن چاپ می‌شود."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="سطح لاگ: DEBUG/INFO/WARN/ERROR (پیش‌فرض: INFO)"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger.info("شروع تست خشک کانفیگ…")
    logger.info("مسیر کانفیگ: %s", args.config)

    try:
        # 1) بارگذاری کانفیگ با لودر پیشرفته (ENV و Alias و Validation در لودر انجام می‌شود)
        loader = ConfigLoader(
            config_path=args.config,
            env_prefix=args.env_prefix,
            enable_env_override=(not args.no_env_override),
        )
        cfg = loader.get_all(copy_="shallow")

        # 2) ایجاد مسیرهای مهم (اگر وجود ندارند)
        made = loader.ensure_dirs()
        if made:
            logger.info("مسیرهای ایجاد/تأییدشده:")
            for k, p in made.items():
                logger.info("  - %s → %s", k, p)
        else:
            logger.info("مسیر خاصی برای ایجاد نبود یا قبلاً موجود بودند.")

        # 3) چاپ خلاصه‌ای از کلیدهای حیاتی
        gates = (cfg.get("evaluation", {}) or {}).get("acceptance_gates", {}) or {}
        risk = (cfg.get("risk", {}) or {})
        exec_ = (cfg.get("executor", {}) or {})
        so = (cfg.get("self_optimize", {}) or {})

        logger.info("گیت‌های پذیرش (evaluation.acceptance_gates): %s", {
            "sharpe_min": gates.get("sharpe_min"),
            "winrate_min": gates.get("winrate_min"),
            "max_drawdown_max": gates.get("max_drawdown_max"),
        })
        logger.info("ریسک (risk): %s", {
            "risk_per_trade": risk.get("risk_per_trade"),
            "max_daily_loss_pct": risk.get("max_daily_loss_pct"),
            "max_total_drawdown_pct": risk.get("max_total_drawdown_pct"),
        })
        if "canary_deployment" in exec_:
            can = exec_["canary_deployment"] or {}
            logger.info("استراتژی کَناری (executor.canary_deployment): %s", {
                "enabled": can.get("enabled"),
                "volume_multiplier": can.get("volume_multiplier"),
                "min_duration_minutes": can.get("min_duration_minutes"),
            })
        else:
            logger.warning("executor.canary_deployment در کانفیگ یافت نشد (توصیه می‌شود اضافه شود).")

        if so:
            logger.info("Self-Optimize: schedule=%s, steps keys=%s",
                        so.get("schedule"), list((so.get("steps") or {}).keys()))
        else:
            logger.warning("بلوک self_optimize یافت نشد (پیشنهاد می‌شود اضافه/تکمیل شود).")

        # 4) اگر نماد مشخص شده، گیت‌های مؤثر آن را چاپ کن
        if args.symbol:
            eff = merge_accept_gates_for_symbol(cfg, args.symbol)
            logger.info("گیت‌های پذیرش مؤثر برای نماد %s: %s", args.symbol, eff)

        # 5) ذخیرهٔ نسخهٔ کانفیگ (نسخه‌گذاری). قبل از ذخیره، لودر خودش اعتبارسنجی را انجام داده است.
        out_path = save_config_versioned(cfg, prefix=args.prefix, tag=args.tag, sort_keys=False)
        logger.info("نسخهٔ کانفیگ با موفقیت ذخیره شد: %s", out_path)

        logger.info("تست خشک با موفقیت پایان یافت ✅")
        return 0

    except Exception as ex:
        logger.exception("تست خشک با خطا مواجه شد ❌: %s", ex)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
