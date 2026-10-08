# -*- coding: utf-8 -*-
# f15_scripts/check_fibo_cfg_smoke.py
# Status in (Bot-RL-3): Completed

"""
اسکریپت اسموک‌تست پارامترهای فیبوناچی (Config-First) — Bot-RL-3

هدف:
    - اطمینان از این‌که «رَپرهای کانفیگ‌محور» در فایل f04_features/indicators/fibonacci.py
      به‌درستی قابل ایمپورت هستند و پارامترهای خود را از config.yaml می‌خوانند.
    - بدون نیاز به دیتای واقعی، لااقل یکی از رَپرها (fib_ext_targets_cfg) را با ورودی‌های
      شبیه‌سازی شده اجرا می‌کنیم تا خروجی DataFrame را ببینیم.
    - برای golden_zone_cfg و fib_cluster_cfg اگر دیتای واقعی نداریم، فقط «پارامترهای
      Resolving شده از config.yaml» را گزارش می‌کنیم و اجرایشان را به مرحله بعد موکول می‌کنیم.

نکات:
    - پیام‌های ترمینال و گزارش‌ها به زبان انگلیسی هستند.
    - توضیحات و کامنت‌ها به زبان فارسی نوشته شده‌اند.
    - این اسکریپت هیچ فایلی را تغییر نمی‌دهد و تنها برای اسموک‌تست «وایرینگ با کانفیگ» است.

روش اجرا:
    - python .\f15_scripts\check_fibo_cfg_smoke.py
    
    برای تغییر دموی ورودی:
    - python .\f15_scripts\check_fibo_cfg_smoke.py --entry 2375.2 --leg-low 2321.5 --leg-high 2410.0 --side long

"""

import argparse
import sys
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import pandas as pd

# ماژول فیبوناچی پروژه (باید قبلاً رَپرها در آن اضافه شده باشند)
# توجه: اگر هنوز رَپرها را اضافه نکرده‌اید، ایمپورت موفق خواهد بود اما رَپرها موجود نخواهند بود.
# --- add root to sys.path + fallback import ---

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from f02_utils.config_loader import ConfigLoader
except ModuleNotFoundError:
    from f02_utils.config_loader import ConfigLoader
# ----------------------------------------------

import f04_features.indicators.fibonacci as fib

# لودر کانفیگ پروژه (طبق ساختار موجود)
#from f02_utils.config_loader import ConfigLoader

# -----------------------------------------------------------------------------
# تنظیم لاگینگ سبک برای این اسکریپت
# -----------------------------------------------------------------------------
logger = logging.getLogger("check_fibo_cfg_smoke")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[SMOKE] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# داده‌ساخت‌ها و توابع کمکی
# -----------------------------------------------------------------------------

@dataclass
class FiboResolvedParams:
    """پارامترهای Resolving‌شده از روی config.yaml یا پیش‌فرض‌ها (برای گزارش)"""
    retracement_ratios: Tuple[float, ...]
    extension_ratios: Tuple[float, ...]
    golden_zone_ratios: Tuple[float, float]
    cluster_tol_pct: float
    cluster_prefer_ratio: float
    cluster_tf_weights: Optional[Dict[str, float]]
    cluster_w_trend: float
    cluster_w_rsi: float
    cluster_w_sr: float
    cluster_sr_tol_pct: float
    sl_atr_mult: float


def resolve_fibo_params_from_config() -> FiboResolvedParams:
    """
    پارامترهای موردنیاز رَپرها را از config.yaml می‌خواند.
    اگر هر کلیدی موجود نباشد، به پیش‌فرض‌های منطقی بازمی‌گردیم.
    """
    try:
        cfg_all: Dict[str, Any] = ConfigLoader().get_all()
    except Exception as e:
        logger.info("Could not load config.yaml via ConfigLoader: %s", e)
        cfg_all = {}

    fibo = fib._deep_get(cfg_all, "features.fibonacci", {}) or {}

    # نسبت‌های رتریسمنت
    retr = fib._deep_get(fibo, "retracement_ratios", None)
    if isinstance(retr, (list, tuple)) and retr:
        retracement_ratios = tuple(float(x) for x in retr)
    else:
        # fallback به ثابت‌های ماژول، اگر موجود؛ وگرنه پیش‌فرض منطقی
        retracement_ratios = tuple(getattr(fib, "DEFAULT_RETR_RATIOS", (0.236, 0.382, 0.5, 0.618, 0.786)))

    # نسبت‌های اکستنشن
    exts = fib._deep_get(fibo, "extension_ratios", None)
    if isinstance(exts, (list, tuple)) and exts:
        extension_ratios = tuple(float(x) for x in exts)
    else:
        extension_ratios = tuple(getattr(fib, "DEFAULT_EXT_RATIOS", (1.272, 1.618, 2.0)))

    # نسبت‌های زون طلایی
    gz = fib._deep_get(fibo, "golden_zone.ratios", None)
    if isinstance(gz, (list, tuple)) and len(gz) == 2:
        golden_zone_ratios = (float(gz[0]), float(gz[1]))
    else:
        golden_zone_ratios = (0.382, 0.618)

    # پارامترهای خوشه‌سازی
    cluster_tol_pct = float(fib._deep_get(fibo, "cluster.tol_pct", 0.08))
    cluster_prefer_ratio = float(fib._deep_get(fibo, "cluster.prefer_ratio", 0.618))
    tfw = fib._deep_get(fibo, "cluster.tf_weights", None)
    cluster_tf_weights: Optional[Dict[str, float]] = None
    if isinstance(tfw, dict) and tfw:
        try:
            cluster_tf_weights = {str(k): float(v) for k, v in tfw.items()}
        except Exception:
            cluster_tf_weights = None

    cluster_w_trend = float(fib._deep_get(fibo, "cluster.w_trend", 10.0))
    cluster_w_rsi   = float(fib._deep_get(fibo, "cluster.w_rsi", 10.0))
    cluster_w_sr    = float(fib._deep_get(fibo, "cluster.w_sr", 10.0))
    cluster_sr_tol_pct = float(fib._deep_get(fibo, "cluster.sr_tol_pct", 0.05))

    # SL ATR Mult (برای fib_ext_targets_cfg)
    sl_atr_mult = float(fib._deep_get(fibo, "sl_atr_mult", 1.5))

    return FiboResolvedParams(
        retracement_ratios=retracement_ratios,
        extension_ratios=extension_ratios,
        golden_zone_ratios=golden_zone_ratios,
        cluster_tol_pct=cluster_tol_pct,
        cluster_prefer_ratio=cluster_prefer_ratio,
        cluster_tf_weights=cluster_tf_weights,
        cluster_w_trend=cluster_w_trend,
        cluster_w_rsi=cluster_w_rsi,
        cluster_w_sr=cluster_w_sr,
        cluster_sr_tol_pct=cluster_sr_tol_pct,
        sl_atr_mult=sl_atr_mult,
    )


def report_resolved_params(p: FiboResolvedParams) -> None:
    """چاپ خوانا و شفاف پارامترهای Resolving‌شده"""
    logger.info("Resolved retracement ratios: %s", p.retracement_ratios)
    logger.info("Resolved extension ratios:   %s", p.extension_ratios)
    logger.info("Resolved golden zone ratios: %s", p.golden_zone_ratios)
    logger.info("Resolved cluster params: tol_pct=%.4f prefer_ratio=%.3f tf_weights=%s w_trend=%.2f w_rsi=%.2f w_sr=%.2f sr_tol_pct=%.3f",
                p.cluster_tol_pct, p.cluster_prefer_ratio, str(p.cluster_tf_weights),
                p.cluster_w_trend, p.cluster_w_rsi, p.cluster_w_sr, p.cluster_sr_tol_pct)
    logger.info("Resolved sl_atr_mult:        %.3f", p.sl_atr_mult)


def check_wrappers_available() -> Dict[str, bool]:
    """
    بررسی می‌کند که رَپرها در ماژول fibonacci در دسترس هستند یا خیر.
    اگر False باشد یعنی باید پچ Option B را به انتهای فایل اضافه کنید.
    """
    availability = {
        "golden_zone_cfg": hasattr(fib, "golden_zone_cfg"),
        "fib_cluster_cfg": hasattr(fib, "fib_cluster_cfg"),
        "fib_ext_targets_cfg": hasattr(fib, "fib_ext_targets_cfg"),
    }
    for name, ok in availability.items():
        logger.info("Wrapper availability - %s: %s", name, "FOUND" if ok else "NOT FOUND")
    return availability


# -----------------------------------------------------------------------------
# دموی اجرای رَپرها
# -----------------------------------------------------------------------------

def demo_fib_ext_targets_cfg(
    p: FiboResolvedParams,
    entry: float,
    leg_low: float,
    leg_high: float,
    side: str) -> Optional[pd.DataFrame]:
    """
    اجرای نمایشی fib_ext_targets_cfg (نیاز به دیتای واقعی ندارد).
    """
    if not hasattr(fib, "fib_ext_targets_cfg"):
        logger.info("Skipping fib_ext_targets_cfg demo (wrapper not found).")
        return None

    logger.info("Running fib_ext_targets_cfg demo with entry=%.3f, leg_low=%.3f, leg_high=%.3f, side=%s", entry, leg_low, leg_high, side)
    try:
        '''
        df = fib.fib_ext_targets_cfg(
            entry_price=entry,
            leg_low=leg_low,
            leg_high=leg_high,
            side=side,
            ext_ratios=None,      # None → از config.yaml خوانده می‌شود
            sl_atr=None,          # برای دمو لازم نیست
            sl_atr_mult=None,     # None → از config.yaml خوانده می‌شود (sl_atr_mult)
        )
        '''
        # امضای فعلی: (last_leg, global_cfg, symbol, tf)
        cfg_all = ConfigLoader().get_all()
        df = fib.fib_ext_targets_cfg(
            last_leg=(leg_low, leg_high),
            global_cfg=cfg_all,
            symbol="DEMO",
            tf="H1",
        )

        # نمایش بخشی از خروجی
        logger.info("fib_ext_targets_cfg: got %d rows", len(df))
        logger.info("fib_ext_targets_cfg: head(5):\n%s", df.head(5).to_string(index=False))
        return df
    except Exception as e:
        logger.info("fib_ext_targets_cfg demo failed: %s", e)
        return None


def demo_golden_zone_cfg(p: FiboResolvedParams) -> None:
    """
    تلاش برای اجرای golden_zone_cfg در حالت بدون دیتای واقعی.
    - اگر دیتای swings نداریم، فقط پارامترهای Resolving‌شده را گزارش می‌کنیم.
    - در صورت تمایل می‌توانیم در آینده یک منبع swings واقعی به این دمو بدهیم.
    """
    if not hasattr(fib, "golden_zone_cfg"):
        logger.info("Skipping golden_zone_cfg demo (wrapper not found).")
        return

    # فقط گزارش پارامترها (بدون اجرای واقعی):
    logger.info("golden_zone_cfg would use ratios=%s and extra=%s",
                p.golden_zone_ratios, p.retracement_ratios)
    logger.info("Skipping golden_zone_cfg execution (no real swings DataFrame provided).")


def demo_fib_cluster_cfg(p: FiboResolvedParams) -> None:
    """
    تلاش برای اجرای fib_cluster_cfg در حالت بدون دیتای واقعی.
    - اگر tf_levels واقعی نداریم، فقط پارامترهای Resolving‌شده را گزارش می‌کنیم.
    """
    if not hasattr(fib, "fib_cluster_cfg"):
        logger.info("Skipping fib_cluster_cfg demo (wrapper not found).")
        return

    logger.info(
        "fib_cluster_cfg would use tol_pct=%.4f prefer_ratio=%.3f tf_weights=%s w_trend=%.2f w_rsi=%.2f w_sr=%.2f sr_tol_pct=%.3f",
        p.cluster_tol_pct, p.cluster_prefer_ratio, str(p.cluster_tf_weights),
        p.cluster_w_trend, p.cluster_w_rsi, p.cluster_w_sr, p.cluster_sr_tol_pct
    )
    logger.info("Skipping fib_cluster_cfg execution (no real tf_levels supplied).")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Fibonacci wrappers with config.yaml")
    parser.add_argument("--entry", type=float, default=2000.0, help="Entry price for fib_ext_targets_cfg demo")
    parser.add_argument("--leg-low", type=float, default=1950.0, help="Leg low for fib_ext_targets_cfg demo")
    parser.add_argument("--leg-high", type=float, default=2050.0, help="Leg high for fib_ext_targets_cfg demo")
    parser.add_argument("--side", type=str, default="long", choices=["long", "short"], help="Side for fib_ext_targets_cfg demo")
    args = parser.parse_args()

    logger.info("Starting Fibonacci config-driven smoke test...")

    # 1) بررسی حضور رَپرها
    availability = check_wrappers_available()

    # 2) Resolve پارامترها از config.yaml
    resolved = resolve_fibo_params_from_config()
    report_resolved_params(resolved)

    # 3) اجرای دمو روی fib_ext_targets_cfg (نیاز به دیتای واقعی ندارد)
    if availability.get("fib_ext_targets_cfg", False):
        demo_fib_ext_targets_cfg(
            p=resolved,
            entry=args.entry, leg_low=args.leg_low, leg_high=args.leg_high, side=args.side
        )

    # 4) گزارش پارامترها برای golden_zone_cfg و fib_cluster_cfg (بدون اجرای واقعی)
    if availability.get("golden_zone_cfg", False):
        demo_golden_zone_cfg(resolved)
    if availability.get("fib_cluster_cfg", False):
        demo_fib_cluster_cfg(resolved)

    logger.info("Fibonacci config-driven smoke test finished.")


if __name__ == "__main__":
    main()
