# f15_testcheck/unit/ts03_idx_lvl__10sr_overlap_score_1B.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__10sr_overlap_score_1B

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from f04_features.indicators.levels import (
    sr_from_zigzag_legs,
    sr_distance_from_levels,
    sr_overlap_score,
)

_PATH = "f16_test_results/"
_PATH = ""
# ---------------------------------------------------
# Load data
# ---------------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain high, low, close")

df.to_csv(f"{_PATH}ts03_idx_lvl__10sr_overlap_score_1B_data.csv")

# ---------------------------------------------------
# Generate SR from ZigZag legs
# ---------------------------------------------------
sr = sr_from_zigzag_legs(
    df,
    tf="5min",
    depth=12,
    deviation=5.0,
    backstep=10,
)
# ---------------------------------------------------
# Compute normalized distances (ATR-based)
# ---------------------------------------------------
dist_df = sr_distance_from_levels(df, sr, atr_window=14)
df["dist_to_support_norm"] = dist_df["dist_to_support_norm"]
df["dist_to_resistance_norm"] = dist_df["dist_to_resistance_norm"]
# ---------------------------------------------------
# Extract recent SR levels (last 2000 candles)
# ---------------------------------------------------
lookback = 2000

raw_levels = pd.concat([
    sr["sr_support"].iloc[-lookback:],
    sr["sr_resistance"].iloc[-lookback:]
]).dropna().values

# ---------------------------------------------------
# Price clustering (simple distance-based clustering)
# ---------------------------------------------------
cluster_tol = df["close"].iloc[-1] * 0.002  # 0.2% price radius

raw_levels_sorted = np.sort(raw_levels)

clusters = []
current_cluster = [raw_levels_sorted[0]]

for lvl in raw_levels_sorted[1:]:
    if abs(lvl - current_cluster[-1]) <= cluster_tol:
        current_cluster.append(lvl)
    else:
        clusters.append(current_cluster)
        current_cluster = [lvl]

clusters.append(current_cluster)

# centroid each cluster
sr_levels_clustered = np.array([np.mean(c) for c in clusters])

# optional weighting by cluster size
sr_weights = np.array([len(c) for c in clusters], dtype=float)
sr_weights = sr_weights / sr_weights.max()

# ---------------------------------------------------
# Compute overlap score
# ---------------------------------------------------
tol_pct = 0.003

scores = []

for price in df["close"].values:
    score = sr_overlap_score(
        price=price,
        sr_levels=sr_levels_clustered,
        tol_pct=tol_pct,
        sr_weights=sr_weights,
    )
    scores.append(score)

df["sr_overlap_clustered"] = scores

# ---------------------------------------------------
# Save results
# ---------------------------------------------------
df[[
    "close",
    "sr_overlap_clustered",
    "dist_to_support_norm",
    "dist_to_resistance_norm"
]].to_csv(f"{_PATH}ts03_idx_lvl__10sr_overlap_score_1B.csv")

# ---------------------------------------------------
# Histogram
# ---------------------------------------------------
plt.figure()
plt.hist(df["sr_overlap_clustered"], bins=50)
plt.title("Clustered SR Overlap Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.savefig(f"{_PATH}ts03_idx_lvl__10sr_overlap_score_1B_hist_clustered.png")
plt.show()

print("✅ Clustered SR test completed.")
print(df["sr_overlap_clustered"].describe())

print("----------------------------------------")
print("Added 2 test result files to f16_test_results:")
print("✅   ts03_idx_lvl__10sr_overlap_score_1B_data.csv")
print("✅   ts03_idx_lvl__10sr_overlap_score_1B.csv")
print("----------------------------------------")
