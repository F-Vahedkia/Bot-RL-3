# -*- coding: utf-8 -*-
"""
فاز C — اجرای ۱۲ اندیکاتور روی ۸ تایم‌فریم (M1,M5,M15,M30,H1,H4,D1,W1)
- از هستهٔ فعلی engine.apply استفاده می‌کند (Spec DSL همان که تست‌ها پاس کرده‌اند).
- دیتای مصنوعیِ دقیقه‌ایِ UTC تولید می‌شود تا همهٔ TFهای بالادست قابل Resample باشند.
- خروجی در فایل CSV ذخیره می‌شود و آمار non-finite پس از warmup گزارش می‌گردد.
"""
# Run: python f15_testcheck/integration/run_spec_phaseC_12x8.py
# در مورخ 1404/08/17 به درستی اجرا شد.

from __future__ import annotations
import numpy as np
import pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
# entry-point هسته (طبق ساختار فعلی پروژه)
from f04_features.OLD.feature_engine import apply as _engine_apply
from f04_features.OLD.feature_registry import REGISTRY  # برای چک‌کردن وجود اندیکاتورها

# ----------------------------- ابزار ساخت دیتای تست -----------------------------
def make_synthetic_ohlcv(n_minutes: int = 80000, seed: int = 2025) -> pd.DataFrame:
    """
    ساخت دیتای دقیقه‌ای واقع‌نما (حدوداً ~55 روز) برای پوشش W1.
    - ایندکس زمانی UTC
    - ستون‌های استاندارد: open/high/low/close/volume
    """
    rs = np.random.RandomState(seed)
    # مسیر قیمت شبه‌تصادفی
    dt = 1.0 / (60.0 * 24.0)
    vol = np.where(rs.rand(n_minutes) < 0.05, rs.uniform(0.02, 0.08, size=n_minutes),
                   rs.uniform(0.005, 0.025, size=n_minutes))
    mu = 0.00008
    log_ret = (mu - 0.5 * (vol ** 2)) * dt + vol * np.sqrt(dt) * rs.randn(n_minutes)
    price = 1800.0 * np.exp(np.cumsum(log_ret))

    # تولید OHLC با نویز درون‌-کندلی
    spread = np.maximum(0.01, 0.0006 * price)
    open_  = price * (1 + 0.0003 * rs.randn(n_minutes))
    close  = price * (1 + 0.0003 * rs.randn(n_minutes))
    high   = np.maximum(open_, close) + spread * (1 + 0.5 * rs.rand(n_minutes))
    low    = np.minimum(open_, close) - spread * (1 + 0.5 * rs.rand(n_minutes))
    volume = (rs.lognormal(mean=12.0, sigma=0.25, size=n_minutes)).astype(np.float64)

    idx = pd.date_range("2022-01-01", periods=n_minutes, freq="min", tz="UTC")
    df = pd.DataFrame(
        {"open": open_.astype(float),
         "high": high.astype(float),
         "low":  low.astype(float),
         "close": close.astype(float),
         "volume": volume},
        index=idx
    )
    df.attrs["symbol"] = "SYMBOL"
    return df

# ----------------------------- تعریف Specها (۱۲×۸) -----------------------------
# نکته: نگاشت آرگومان‌های موقعیتی → نام‌دار قبلاً در parser پچ شده است.
INDICATORS_12 = [
    "sma(close,20)",       # Simple MA
    "ema(close,50)",       # Exponential MA
    "rsi(14)",             # RSI
    "macd(12,26,9)",       # MACD
    "atr(14)",             # ATR
    "bbands(20,2)",        # Bollinger Bands
    "cci(20)",             # CCI
    "mfi(14)",             # Money Flow Index
    "stoch(high=high,low=low,close=close,14,3)",       # Stochastic
    "wma(close,20)",       # Weighted MA
    "wr(14)",              # Williams %R
    "sar(0.02,0.2)",       # Parabolic SAR
]

TIMEFRAMES_8 = ["M1","M5","M15","M30","H1","H4","D1","W1"]


def build_specs(indicators: list[str], tfs: list[str]) -> list[str]:
    """ساخت Spec فقط برای اندیکاتورهایی که واقعاً در رجیستری هستند؛ بقیه گزارش می‌شوند."""
    # اسم اندیکاتور را از رشتهٔ "name(args)" جدا کنیم: "sma(close,20)" -> "sma"
    def _name_only(s: str) -> str:
        p = s.find("(")
        return s[:p] if p != -1 else s

    names = [_name_only(x) for x in indicators]
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        print("Skipping missing indicators (not in REGISTRY):", missing)

    present = [ind for ind in indicators if _name_only(ind) in REGISTRY]
    out: list[str] = []
    for ind in present:
        for tf in tfs:
            out.append(f"{ind}@{tf}")
    return out


# ----------------------------- اعتبارسنجی non-finite -----------------------------
def nonfinite_report(df: pd.DataFrame) -> dict:
    """
    محاسبهٔ مجموع non-finite (NaN/Inf) برای هر ستون فیچر بعد از اولین مقدار معتبر.
    فقط برای ستون‌هایی که با '__' شروع می‌شوند (ستون‌های فیچر).
    """
    feat_cols = [c for c in df.columns if c.startswith("__")]
    rep = {}
    for c in feat_cols:
        s = df[c]
        fv = s.first_valid_index()
        tail = s if fv is None else s.loc[fv:]
        rep[c] = int((~np.isfinite(tail.to_numpy(dtype=float))).sum())
    return rep

# ----------------------------- اجرای اصلی -----------------------------
def main():
    from f04_features.OLD.feature_registry import REGISTRY
    print("stoch in REGISTRY:", "stoch" in REGISTRY)
    print("keys like stoch:", [k for k in REGISTRY if "stoch" in k])



    # ۱) ساخت دیتای دقیقه‌ای
    df = make_synthetic_ohlcv()

    # ۲) ساخت لیست Specها
    specs = build_specs(INDICATORS_12, TIMEFRAMES_8)

    # ۳) اجرا با entry استاندارد
    print("Running engine.apply() on 12 indicators × 8 TFs ...")
    out = _engine_apply(df=df, specs=specs)

    # ۴) ذخیرهٔ خروجی و گزارش
    out_path = "f15_testcheck/_reports/phaseC_mtf_12x8.csv"
    out.to_csv(out_path, index=True)
    print("Saved:", out_path)
    print("Shape:", out.shape)

    # گزارش کوتاه non-finite
    rep = nonfinite_report(out)
    bad = {k:v for k,v in rep.items() if v != 0}
    print("Non-finite after warmup (nonzero only):", bad if bad else "All good")

    # چاپ چند ستون نمونه برای اطمینان
    sample_cols = [c for c in out.columns if c.startswith("__sma@M1") or c.startswith("__rsi@M5") or c.startswith("__macd@M15")]
    print("Sample feature columns:", sample_cols[:10])


    # --- debug: مخصوص stoch ---//////////////
    specs_stoch = [s for s in specs if s.startswith("stoch(")]
    if specs_stoch:
        try:
            # اگر می‌خواهی سبک‌تر اجرا شود، می‌توانی df کوچک‌تری بسازی:
            df_small = df.iloc[:5000]
            df_dbg = _engine_apply(df=df_small, specs=specs_stoch[:8])  # ← فقط df و specs

            stoch_cols = [c for c in df_dbg.columns if c.startswith("__stoch@")]
            print("STOCH columns:", stoch_cols[:10])
            if not stoch_cols:
                print("STOCH present in REGISTRY but produced no columns (likely all-NaN or filtered).")
            else:
                # چک غیرنهایی بودن داده بعد از warmup
                import numpy as np
                nz = df_dbg[stoch_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
                print("STOCH non-empty rows:", len(nz))
        except Exception as e:
            import traceback; traceback.print_exc()
            print("STOCH failed with exception:", e)
    else:
        print("No stoch specs were built (name mismatch?).")
    #/////////////////////////////////////////


    # --- Summary: how many indicators/TFs actually materialized ---
    import pandas as pd
    df_out = pd.read_csv(out_path)
    feat_cols = [c for c in df_out.columns if c.startswith("__")]
    inds = sorted({c.split("@", 1)[0].lstrip("_") for c in feat_cols})
    tfs  = sorted({c.split("@", 1)[1].split("__", 1)[0] for c in feat_cols})
    print(f"Indicators present ({len(inds)}):", inds)
    print(f"TFs present ({len(tfs)}):", tfs)

if __name__ == "__main__":
    main()
