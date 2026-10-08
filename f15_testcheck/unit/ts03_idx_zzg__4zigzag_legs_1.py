# f15_testcheck/unit/ts03_idx_zzg__4zigzag_legs_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_zzg__4zigzag_legs_1

import numpy as np
import pandas as pd
from f04_features.indicators.zigzag import (
    zigzag,
    zigzag_mtf_adapter,
    zigzag_legs,
)

_PATH = "f16_test_results/"
# --- Load data -------------------------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

high = df["high"]
low  = df["low"]

# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------
STD_COLS = [
    "ltf_start_ts",
    "ltf_end_ts",
    "direction",
    "ltf_start_extr",
    "ltf_end_extr",
    "ltf_start_pos",
    "ltf_end_pos",
]


def normalize_equal_tf(legs_raw: list[dict]) -> pd.DataFrame:
    """Convert zigzag() legs schema → unified LTF schema."""
    mapped = []
    for leg in legs_raw:
        mapped.append({
            "ltf_start_ts": leg["start_ts"],
            "ltf_end_ts": leg["end_ts"],
            "direction": int(leg["direction"]),
            "ltf_start_extr": leg["start_extr"],
            "ltf_end_extr": leg["end_extr"],
            "ltf_start_pos": int(leg["start_pos"]),
            "ltf_end_pos": int(leg["end_pos"]),
        })
    return pd.DataFrame(mapped, columns=STD_COLS).reset_index(drop=True)


def normalize__df(df: pd.DataFrame) -> pd.DataFrame:
    """Force deterministic ordering and dtype consistency."""
    if df.empty:
        return pd.DataFrame(columns=STD_COLS)

    out = df.copy()
    out = out[STD_COLS].reset_index(drop=True)

    out["direction"] = out["direction"].astype(int)
    out["ltf_start_pos"] = out["ltf_start_pos"].astype(int)
    out["ltf_end_pos"] = out["ltf_end_pos"].astype(int)

    return out


# ======================================================================
# TEST 1 — Equal Timeframe (1min)
# ======================================================================
print("Running TEST 1 (Equal TF)...")

res_zigzag = zigzag(
    high=high,
    low=low,
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
    addmeta=True,
    final_check=True,
)
legs_zigzag = res_zigzag.attrs.get("legs", [])

expected_df = normalize_equal_tf(legs_zigzag)

res_zigzag_legs = zigzag_legs(
    high=high,
    low=low,
    tf="1min",
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
    mode="forward_fill",
    extend_last_leg=False,
    use_timeshift=False,
)

actual_df = normalize__df(res_zigzag_legs)

pd.testing.assert_frame_equal(expected_df, actual_df)
print("TEST 1 PASSED ✓")


# ======================================================================
# TEST 2 — Higher Timeframe (5min, last mode, timeshift=True)
# ======================================================================
print("Running TEST 2 (HTF=5min, last mode)...")

zz_last = zigzag_mtf_adapter(
    high=high,
    low=low,
    tf_higher="5min",
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
    mode="last",
    extend_last_leg=False,
    use_timeshift=True,
)

expected_df = normalize__df(pd.DataFrame(
    zz_last.attrs.get("legs", []),
    columns=STD_COLS
))

res_zigzag_legs = zigzag_legs(
    high=high,
    low=low,
    tf="5min",
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
    mode="last",
    extend_last_leg=False,
    use_timeshift=True,
)

actual_df = normalize__df(res_zigzag_legs)

pd.testing.assert_frame_equal(expected_df, actual_df)
print("TEST 2 PASSED ✓")


# ======================================================================
# TEST 3 — Lower Timeframe (invalid case)
# ======================================================================
print("Running TEST 3 (Invalid lower TF)...")

res_invalid = zigzag_legs(
    high=high,
    low=low,
    tf="30s",
)

assert res_invalid.empty
assert list(res_invalid.columns) == STD_COLS

print("TEST 3 PASSED ✓")

print("\nALL TESTS PASSED SUCCESSFULLY ✔")
