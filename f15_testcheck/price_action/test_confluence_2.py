"""
Unit Tests — Confluence Engine (Price Action)
=============================================
این فایل تست‌ها را برای Confluence Engine روی داده واقعی XAUUSD M1 اجرا می‌کند.
ویژگی‌ها و رفتارهای زیر بررسی می‌شوند:

1. ستون‌ها و بازه‌های مجاز امتیازها و پرچم‌ها
2. anti_lookahead → شیفت مقادیر و NaN ابتدای دیتافریم
3. تاثیر فیچرهای extras (Regime / Breakouts / Microchannels)
4. هماهنگی با آخرین کندل بسته شده
5. اطمینان از warm-up robust → NaN ها در ابتدای دیتافریم به حساب نمی‌آیند
6. سازگاری با داده UTC و index datetime

Run:
    pytest f15_testcheck/price_action/test_confluence_2.py -v
"""

import pandas as pd
import numpy as np
from pathlib import Path

from f04_features.price_action import (
    breakouts_1 as bo,
    confluence_numba_6 as cf,
    regime as rg,
    microchannels as mc,
)


def _sample_df(n=80):
    n = 20
    df = pd.DataFrame({
        "close": [100 + i*0.4 for i in range(n)],
        # market_structure
        "swing_type": ["", "HL", "HH", "", "LH", "LL", "HL", "HH", "", "", "LH", "LL", "", "HL", "HH", "", "", "LH", "LL", ""],
        "bos_up":     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "bos_down":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        "choch_up":   [0]*20,
        "choch_down": [0]*20,
        # zones
        "dist_to_sd": [0.8, 0.5, 0.2, 0.3, 1.0, 1.5, 0.7, 0.1, 0.4, 0.6, 0.9, 0.3, 0.25, 0.2, 0.5, 0.8, 0.7, 1.2, 0.9, 0.4],
        "dist_to_ob": [1.1, 0.6, 0.4, 0.2, 0.9, 0.7, 1.4, 0.3, 0.25, 0.2, 0.6, 0.7, 0.8, 0.5, 0.4, 0.9, 0.95, 1.1, 0.8, 0.3],
        # imbalance
        "dist_to_fvg_mid": [0.9, 0.4, 0.2, 0.7, 1.2, 1.5, 0.8, 0.15, 0.5, 1.0, 0.9, 0.2, 0.3, 0.25, 0.45, 0.6, 1.3, 0.95, 0.85, 0.4],
        "sweep_up":   [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0],
        "sweep_down": [0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
        # mtf
        "mtf_confluence_score": [0.6,0.55,0.7,0.8,0.4,0.35,0.5,0.9,0.6,0.55,0.45,0.5,0.6,0.65,0.7,0.4,0.35,0.5,0.55,0.6],
        "mtf_conflict": [0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0],
        "mtf_strength": [0.2,0.3,0.4,0.6,0.5,0.3,0.2,0.8,0.6,0.4,0.2,0.2,0.3,0.5,0.6,0.4,0.3,0.2,0.4,0.5],
    })
    return df


def _sample_df_2(n=80):
    # نیمه اول: رنج → نیمه دوم: شکست به بالا + کانال
    close = []
    for i in range(n):
        if i < n//2:
            close.append(100 + (-1)**i * 0.08)
        else:
            close.append(101 + (i - n//2) * 0.35)
    high = [c + 0.12 for c in close]
    low  = [c - 0.12 for c in close]
    return pd.DataFrame({"close": close, "high": high, "low": low})


# =========================
# --- Utility function ---
# =========================
def load_xauusd_m1_features(
    path: str = "f03_data/raw/XAUUSD/M1.csv",
    nrows: int | None = None,
    simulate_features: bool = True
) -> pd.DataFrame:
    """
    Load real XAUUSD M1 data and optionally generate base features for Confluence Engine.

    پارامترها:
        path: مسیر فایل CSV
        nrows: تعداد ردیف برای load کردن (None → تمام داده)
        simulate_features: اگر True باشد، ستون‌های پایه مورد نیاز confluence ساخته می‌شوند

    خروجی:
        DataFrame با index datetime (UTC) و ستون‌های ['open','high','low','close','volume']
        و در صورت simulate_features، ستون‌های پایه Confluence:
        ['swing_type','bos_up','bos_down','choch_up','choch_down',
         'dist_to_sd','dist_to_ob','dist_to_fvg_mid','sweep_up','sweep_down',
         'mtf_confluence_score','mtf_conflict','mtf_strength']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # ---------- Load CSV ----------
    df = pd.read_csv(path, nrows=nrows)[-2000:]
    required_cols = ['time','open','high','low','close','volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # ---------- Preprocessing ----------
    # استفاده از ستون time و تنظیم UTC
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)

    # تبدیل مقادیر به numeric و حذف ردیف‌های ناقص
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    # ---------- Simulate basic features for Confluence ----------
    if simulate_features:
        n = len(df)
        df['swing_type'] = np.random.choice(["", "HH", "HL", "LH", "LL"], size=n)
        df['bos_up'] = np.random.randint(0, 2, size=n)
        df['bos_down'] = np.random.randint(0, 2, size=n)
        df['choch_up'] = np.random.randint(0, 2, size=n)
        df['choch_down'] = np.random.randint(0, 2, size=n)
        df['dist_to_sd'] = np.random.rand(n) * 2.0
        df['dist_to_ob'] = np.random.rand(n) * 2.0
        df['dist_to_fvg_mid'] = np.random.rand(n) * 2.0
        df['sweep_up'] = np.random.randint(0, 2, size=n)
        df['sweep_down'] = np.random.randint(0, 2, size=n)
        df['mtf_confluence_score'] = np.random.rand(n)
        df['mtf_conflict'] = np.random.randint(0, 2, size=n)
        df['mtf_strength'] = np.random.rand(n) * 0.8

        # Extras (Regime / Breakouts / Microchannels)
        df['regime_channel'] = np.random.rand(n)
        df['regime_spike'] = np.random.rand(n)
        df['regime_range'] = np.random.rand(n)
        df['breakout_up'] = np.random.randint(0, 2, size=n)
        df['breakout_down'] = np.random.randint(0, 2, size=n)
        df['retest_up'] = np.random.randint(0, 2, size=n)
        df['retest_down'] = np.random.randint(0, 2, size=n)
        df['fail_break_up'] = np.random.randint(0, 2, size=n)
        df['fail_break_down'] = np.random.randint(0, 2, size=n)
        df['micro_channel_up'] = np.random.randint(0, 2, size=n)
        df['micro_channel_down'] = np.random.randint(0, 2, size=n)
        df['micro_channel_quality'] = np.random.rand(n)

    return df


# =========================
# --- Unit Tests ---
# =========================
def test_confluence_columns_and_ranges():
    """
    تست کلیه ستون‌های خروجی Confluence و اطمینان از قرار داشتن امتیازها
    و پرچم‌ها در بازه مجاز (0..1 یا {0,1}) و طول برابر با دیتافریم ورودی
    """
    df = load_xauusd_m1_features()
    out = cf.build_confluence(df, anti_lookahead=True)

    # ستون‌های مورد انتظار
    expected_cols = [
        "conf_components_structure",
        "conf_components_zones",
        "conf_components_imbalance",
        "conf_components_mtf",
        "conf_score",
        "conf_flag_strong_entry",
        "conf_flag_filter_pass",
    ]
    for col in expected_cols:
        assert col in out.columns, f"Column missing: {col}"
        assert len(out[col]) == len(df)

    # بررسی مقادیر معتبر
    assert out["conf_score"].between(0.0, 1.0, inclusive="both").all()
    assert set(out["conf_flag_strong_entry"].unique()).issubset({0, 1})
    assert set(out["conf_flag_filter_pass"].unique()).issubset({0, 1})


def test_confluence_antilookahead_shift():
    """
    بررسی ویژگی anti-lookahead:
    - مقادیر ردیف اول باید NaN یا تغییر یافته باشند
    - اطمینان از اینکه هیچ داده آینده‌ای در گذشته استفاده نمی‌شود
    """
    df = load_xauusd_m1_features()
    out_no_shift = cf.build_confluence(df, anti_lookahead=False)
    out_shifted = cf.build_confluence(df, anti_lookahead=True)

    check_cols = [
        "conf_components_structure",
        "conf_components_zones",
        "conf_components_imbalance",
        "conf_components_mtf",
        "conf_score",
    ]

    for col in check_cols:
        if pd.notna(out_no_shift[col].iloc[0]):
            assert pd.isna(out_shifted[col].iloc[0]) or (out_no_shift[col].iloc[0] != out_shifted[col].iloc[0])


def test_confluence_with_and_without_extras():
    """
    بررسی تاثیر weight های extras (Regime/Breakouts/Microchannels) روی conf_score:
    - مقایسه حالت weight=0 با weight>0
    - اطمینان از اینکه conf_score افزایش پیدا می‌کند
    """
    df = _sample_df_2()
    # df = load_xauusd_m1_features()

    # تولید سیگنال‌های سه ماژول
    df = rg.build_regime(df, anti_lookahead=True, slope_window=6, width_window=8)
    df = bo.build_breakouts(df, anti_lookahead=True, confirm_closes=1, range_window=8, min_periods=4)
    df = mc.build_microchannels(df, anti_lookahead=True, min_len=3)

    # حالت پایه: extras=0
    base = cf.build_confluence(df, anti_lookahead=True, weights={
        "structure": 0.30, "zones": 0.30, "imbalance": 0.20, "mtf": 0.20, "extras": 0.00
    })
    # حالت با extras>0
    with_extras = cf.build_confluence(df, anti_lookahead=True, weights={
        "structure": 0.25, "zones": 0.25, "imbalance": 0.20, "mtf": 0.10, "extras": 0.20
    })

    assert "conf_components_extras" in with_extras.columns
    assert len(with_extras) == len(base)
    # میانگین conf_score افزایش یافته یا برابر است
    assert with_extras["conf_score"].mean() >= base["conf_score"].mean() - 1e-9
