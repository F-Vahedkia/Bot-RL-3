# -*- coding: utf-8 -*-
# f15_scripts/check_patterns_smoke_real.py
# Status in (Bot-RL-3): Completed

# FINAL — Smoke test for candlestick patterns (config-driven, real data)
# Runtime prints: English | Comments: Persian
r"""
روش اجرا توسط CLI:
python -m f15_scripts.check_patterns_smoke_real  `
    -c .\f01_config\config.yaml                  `
    --symbol XAUUSD                              `
    --base-tf D1                                 `
    --data-path f03_data\processed\XAUUSD\D1.parquet
"""
import argparse
import sys, os, json
from pathlib import Path
import pandas as pd

from f02_utils.config_loader import ConfigLoader
from f04_features.indicators.patterns import registry as patterns_registry
from f02_utils.functions.parser import parse_spec_v2

#-------------------------------
# خواندن فایل داده (Parquet/CSV)
def _read_data_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {ext}")

#-------------------------------
# استخراج ستون‌های OHLC (بدون/با پیشوند TF)
def _extract_ohlc(df: pd.DataFrame, base_tf: str):
    cols_plain = ["open", "high", "low", "close"]
    if all(c in df.columns for c in cols_plain):
        return df["open"], df["high"], df["low"], df["close"]
    if base_tf:
        pref = f"{base_tf}_"
        cand = [pref + "open", pref + "high", pref + "low", pref + "close"]
        if all(c in df.columns for c in cand):
            return df[cand[0]], df[cand[1]], df[cand[2]], df[cand[3]]
    raise ValueError("Cannot infer OHLC columns (need open/high/low/close or <TF>_open/... in file).")

#-------------------------------
# یافتن خودکار فایل processed بر اساس config
def _autodetect_processed(cfg: dict, symbol: str, base_tf: str) -> Path:
    paths = cfg.get("paths") or {}
    processed_dir = Path(paths.get("processed_dir", "f03_data/processed"))
    sym_dir = processed_dir / symbol
    if not sym_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {sym_dir}")
    candidates = [sym_dir / f"{base_tf}.parquet", sym_dir / f"{base_tf}.csv"]
    for p in candidates:
        if p.exists(): return p
    for p in sorted(sym_dir.glob("*.parquet")) + sorted(sym_dir.glob("*.csv")):
        if base_tf.lower() in p.name.lower(): return p
    raise FileNotFoundError(f"No processed file for base_tf={base_tf} in {sym_dir}")

#-------------------------------
def main():
    ap = argparse.ArgumentParser(description="Config-driven smoke test for candlestick patterns on real data")
    ap.add_argument("-c", "--config", default="f01_config/config.yaml")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--base-tf", dest="base_tf", default="H1")
    ap.add_argument("--data-path", dest="data_path", default=None)
    args = ap.parse_args()

    print("[RUN] Loading config ...")
    cfg_all = ConfigLoader(args.config).get_all(copy_="shallow")

    # مسیر فایل داده
    if args.data_path:
        data_path = Path(args.data_path)
        if not data_path.exists():
            print(f"[ERROR] Data file not found: {data_path}")
            sys.exit(2)
    else:
        try:
            data_path = _autodetect_processed(cfg_all, args.symbol, args.base_tf)
        except Exception as e:
            print("[ERROR] Autodetect processed file failed:", e)
            print("        Tip: pass --data-path to a concrete file (parquet/csv).")
            sys.exit(2)

    print(f"[RUN] Reading data: {data_path}")
    try:
        df_all = _read_data_file(data_path)
    except Exception as e:
        print("[ERROR] Failed to read data file:", e); sys.exit(2)
    if df_all is None or df_all.empty:
        print("[ERROR] Empty dataframe from data file."); sys.exit(2)

    # OHLC
    try:
        o, h, l, c = _extract_ohlc(df_all, args.base_tf)
    except Exception as e:
        print("[ERROR] OHLC detection failed:", e)
        print("        Columns available:", list(df_all.columns)[:40], "...")
        sys.exit(2)
    df_ohlc = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).astype("float32").dropna()

    # خواندن Specهای pat_* از config و فیلتر بر اساس TF
    cfg_all = ConfigLoader(args.config).get_all(copy_="shallow")
    specs = (cfg_all.get("features", {}) or {}).get("indicators", []) or []
    pars = [parse_spec_v2(s) for s in specs]
    # فیلتر بر اساس pat_* و تایم‌فریم
    base = str(args.base_tf).upper()
    pars = [p for p in pars if p and p.name.startswith("pat_")
            and (str(p.timeframe or base).upper() == base)]

    if not pars:
        print("[WARN] No pat_* specs found for base-tf:", args.base_tf)

    print("[RUN] Loading patterns registry ...")
    reg = patterns_registry()

    print("[RUN] Running pattern specs from config ...")
    out_cols, missing = [], []
    for p in pars:
        if p.name not in reg:
            print(f"[WARN] Pattern key not in registry: {p.name}"); missing.append(p.name); continue
        res = reg[p.name](df_ohlc, **(p.kwargs or {}))  # انتظار: dict[str -> Series]
        if not isinstance(res, dict):
            print(f"[ERROR] Registry '{p.name}' returned non-dict result."); sys.exit(2)
        for name, series in res.items():
            s = series.astype("int8")
            if len(s) != len(df_ohlc):
                print(f"[ERROR] Length mismatch for {name}: {len(s)} != {len(df_ohlc)}"); sys.exit(2)
            out_cols.append((name, s))

    if not out_cols:
        print("[ERROR] No output columns produced. Check config specs or registry wiring."); sys.exit(2)

    out = pd.DataFrame(index=df_ohlc.index)
    for name, s in out_cols:
        out[name] = s

    print(f"[OK] Output columns: {out.shape[1]}")
    bad_types = [c for c in out.columns if out[c].dtype != "int8"]
    if bad_types:
        print("[WARN] Non-int8 columns detected:", bad_types)
    else:
        print("[OK] All pattern flags are int8.")

    # Bars count from meta (fallback: len(df_ohlc))
    meta_path = Path("f03_data") / "processed" / args.symbol / f"{str(args.base_tf).upper()}.meta.json"
    bars = len(df_ohlc)
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            bars = int(meta.get("rows", meta.get("num_rows", bars)))
    except Exception:
        pass
    print(f"[INFO] Bars ({args.base_tf}): {bars}")
    
    # خلاصهٔ سریع
    sample_cols = list(out.columns)[:10]
    print("\n[HEAD] First 5 rows (subset):")
    print(out[sample_cols].head(5).to_string())

    bars = max(int(df_ohlc.shape[0]), 1)
    rates = (out.sum() / bars).sort_values(ascending=False).head(15)
    print("\n[STATS] Top-15 flags by rate (% of bars):")
    print(rates.apply(lambda v: f"{v:.3%}").to_string())  # 0.001 دقت درصد

    if missing:
        print("\n[NOTE] Missing registry keys:", missing)

    print("\n[DONE] Smoke test (real data, config-driven) finished successfully.")

if __name__ == "__main__":
    main()
