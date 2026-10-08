# -*- coding: utf-8 -*-
"""
مثال فاز C — اجرای اندیکاتورهای چند‌تایم‌فریمی با هستهٔ فعلی Engine
- M1:  SMA(20) روی close
- M5:  RSI(14)
- M15: MACD(12,26,9)

توضیحات:
- از entry-point استاندارد engine.apply استفاده می‌کنیم (طبق تست‌ها).
- ایندکس زمانی UTC و با فرکانس دقیقه است تا resample داخلی برای M5/M15 کار کند.
- خروجی در کنار چاپ، به فایل CSV ذخیره می‌شود تا برای رهگیری و بررسی بعدی استفاده گردد.
"""

# Run: python f15_testcheck/integration/run_spec_phaseC_mtf.py
# در مورخ 1404/08/18 به درستی اجرا شد.

from __future__ import annotations

import numpy as np
import pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# نکته: طبق ساختار پروژه، indicators زیر f04_features قرار دارد و engine.apply
# طبق تست‌ها به run_specs_v2 متصل است. (No guessing: بر اساس فایل‌های فعلی)
from f04_features.OLD.feature_engine import apply as engine_apply  # entry wrapper

def make_synthetic_ohlcv(n: int = 3000, seed: int = 2025) -> pd.DataFrame:
    """ساخت دیتای OHLCV دقیقه‌ای واقع‌نما برای تست؛ ایندکس UTC."""
    rs = np.random.RandomState(seed)
    # قیمت با فرآیند GBM ساده
    dt = 1.0 / (60.0 * 24.0)
    vol = np.where(rs.rand(n) < 0.05, rs.uniform(0.02, 0.08, size=n), rs.uniform(0.005, 0.025, size=n))
    mu = 0.0001
    log_ret = (mu - 0.5 * (vol ** 2)) * dt + vol * np.sqrt(dt) * rs.randn(n)
    price = 1800.0 * np.exp(np.cumsum(log_ret))

    # ایجاد OHLC با کمی نویز درون‌-کندلی
    spread = np.maximum(0.01, 0.0005 * price)
    open_  = price * (1 + 0.0003 * rs.randn(n))
    close  = price * (1 + 0.0003 * rs.randn(n))
    high   = np.maximum(open_, close) + spread * (1 + 0.5 * rs.rand(n))
    low    = np.minimum(open_, close) - spread * (1 + 0.5 * rs.rand(n))
    volume = (rs.lognormal(mean=12.0, sigma=0.2, size=n)).astype(np.float64)

    idx = pd.date_range("2022-01-01", periods=n, freq="min", tz="UTC")  # UTC index (engine خودش هم UTC را enforce می‌کند)
    df = pd.DataFrame(
        {"open": open_.astype(float),
         "high": high.astype(float),
         "low": low.astype(float),
         "close": close.astype(float),
         "volume": volume},
        index=idx
    )
    # در صورت نیازِ برخی اندیکاتورهای پیشرفته‌تر:
    df.attrs["symbol"] = "SYMBOL"
    return df

def main():
    # ۱) دیتای دقیقه‌ای
    df = make_synthetic_ohlcv(n=3000, seed=1337)

    # ۲) تعریف Specهای چند‌تایم‌فریمی — بدون حدس؛ مطابق parser/engine فعلی
    specs = [
        "sma(close,20)@M1",       # SMA(20) روی close در M1
        "rsi(14)@M5",             # RSI(14) در M5
        "macd(12,26,9)@M15",      # MACD سه‌ستونه در M15
    ]

    # ۳) اجرا با entry استاندارد (engine.apply → run_specs_v2)
    out = engine_apply(df=df, specs=specs)

    # ۴) گزارش و ذخیره
    cols = list(out.columns)
    print("MTF run completed.")
    print("Specs:", specs)
    print("Cols:", cols)
    print("Shape:", out.shape)

    # برای ارجاع و بررسیِ بعدی:
    out_path = "f15_testcheck/_reports/phaseC_mtf_sample.csv"
    out.to_csv(out_path, index=True)
    print("Saved: ", out_path)

if __name__ == "__main__":
    main()
