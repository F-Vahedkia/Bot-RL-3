
# Run: python f15_testcheck/integration/run_spec_phaseC_adv_probe.py
# در مورخ 1404/08/18 به درستی اجرا شد.

import numpy as np, pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from f04_features.OLD.feature_engine import apply as engine_apply

# دیتای دقیقه‌ای سبک
idx = pd.date_range("2022-01-01", periods=60*24*30, freq="min", tz="UTC")
rng = np.random.default_rng(0)

close = pd.Series(np.cumsum(rng.normal(0,0.2,60*24*30))+100, index=idx)
open_ = close.shift(1).fillna(close)
high  = np.maximum(open_, close) + rng.random(60*24*30)*0.2
low   = np.minimum(open_, close) - rng.random(60*24*30)*0.2

vol   = pd.Series(rng.integers(100,1000,60*24*30), index=idx)
df = pd.DataFrame({"open":open_, "high":high, "low":low, "close":close, "volume":vol})

specs = [
    "adr(14)@M1",
    "adr_distance_to_open(14)@M1",
    "sr_overlap_score(anchor=100, step=5, n=25, tol_pct=0.02)@M5",
    "round_levels(anchor=100, step=5, n=25)@M15",
]
out = engine_apply(df=df, specs=specs)
print("Cols:", [c for c in out.columns if c.startswith('__')][:12])
print("Shape:", out.shape)

out_path = "f15_testcheck/_reports/phaseC_adv_probe.csv"
out.to_csv(out_path)
print("\n Saved:", out_path)
