# Run:
# python f15_testcheck/diagnostics/probe_engine_adv.py
# در مورخ 1404/08/17 تست شد و درست کار میکند

import numpy as np, pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from f04_features.OLD.feature_engine import apply

idx = pd.date_range("2022-01-01", periods=2000, freq="min", tz="UTC")
rng = np.random.default_rng(0)

close = pd.Series(np.cumsum(rng.normal(0,0.2,2000))+100, index=idx)
open_ = close.shift(1).fillna(close)
high = np.maximum(open_, close)+0.2
low = np.minimum(open_, close)-0.2

df = pd.DataFrame({"open":open_,
                   "high":high,
                   "low":low,
                   "close":close}
                )

specs = ["adr(window=14)@M1",
         "adr_distance_to_open(window=14)@M1",
         "sr_overlap_score(anchor=100,step=5,n=25,tol_pct=0.02)@M5",
         "round_levels(anchor=100,step=5,n=25)@M15"
        ]
out = apply(df=df, specs=specs)
print([c for c in out.columns if c.startswith("__")])
