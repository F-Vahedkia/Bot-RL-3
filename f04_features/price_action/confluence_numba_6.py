# f04_features/price_action/confluence_numba_6.py
# FINAL v98

"""
**Confluence** در فارسی یعنی:
**همگرایی سیگنال‌ها** یا **تلاقی چند عامل تحلیلی**.
در ترید یعنی جایی که **چند نشانهٔ تحلیلی مختلف همزمان یک ناحیه را تأیید می‌کنند**.
"""

import numpy as np
import pandas as pd
from numba import njit, prange

# =========================
# Numba helpers (PURE numpy)
# =========================

@njit(cache=True, fastmath=True)
def _clip01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

@njit(cache=True, fastmath=True)
def _norm_distance_nb(dist):
    # dist: float32 ndarray
    n = dist.shape[0]
    out = np.empty(n, dtype=np.float32)
    # robust normalization to [0,1]
    mn = 1e30
    mx = -1e30
    for i in range(n):
        v = dist[i]
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    den = mx - mn
    if den <= 0.0:
        for i in range(n):
            out[i] = 0.0
        return out
    for i in range(n):
        out[i] = _clip01((dist[i] - mn) / den)
    return out

@njit(cache=True, fastmath=True, parallel=True)
def structure_score_nb(a, b, c):
    # all inputs float32 ndarray
    n = a.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        out[i] = _clip01((a[i] + b[i] + c[i]) * 0.33333334)
    return out

@njit(cache=True, fastmath=True, parallel=True)
def zones_score_nb(z1, z2):
    n = z1.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        out[i] = _clip01((z1[i] + z2[i]) * 0.5)
    return out

@njit(cache=True, fastmath=True, parallel=True)
def imbalance_score_nb(x):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        out[i] = _clip01(x[i])
    return out

@njit(cache=True, fastmath=True, parallel=True)
def mtf_score_nb(x):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        out[i] = _clip01(x[i])
    return out

@njit(cache=True, fastmath=True, parallel=True)
def extras_score_nb(x):
    # monotonic non-negative effect
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        v = x[i]
        out[i] = v if v > 0.0 else 0.0
    return out

@njit(cache=True, fastmath=True, parallel=True)
def combine_nb(s, z, im, m, ex, ws, wz, wi, wm, wex):
    n = s.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        v = s[i]*ws + z[i]*wz + im[i]*wi + m[i]*wm + ex[i]*wex
        out[i] = _clip01(v)
    return out

# =========================
# Public API
# =========================

def build_confluence(
    df: pd.DataFrame,
    anti_lookahead: bool = True,
    weights: dict | None = None,
):
    if weights is None:
        weights = {
            "structure": 0.30,
            "zones": 0.30,
            "imbalance": 0.20,
            "mtf": 0.20,
            "extras": 0.00,
        }

    # required columns (safe get, default zeros)
    def col(name):
        if name in df.columns:
            return df[name].to_numpy(dtype=np.float32, copy=False)
        return np.zeros(len(df), dtype=np.float32)

    # STRUCTURE inputs
    s1 = col("pa_structure_1")
    s2 = col("pa_structure_2")
    s3 = col("pa_structure_3")

    # ZONES inputs
    z1 = col("pa_zone_1")
    z2 = col("pa_zone_2")

    # IMBALANCE
    im = col("pa_imbalance")

    # MTF
    m = col("pa_mtf")

    # EXTRAS (regime / breakouts / microchannels)
    ex_raw = (
        col("rg_signal")
        + col("bo_signal")
        + col("mc_signal")
    )

    # numba scores
    s_score = structure_score_nb(s1, s2, s3)
    z_score = zones_score_nb(z1, z2)
    im_score = imbalance_score_nb(im)
    m_score = mtf_score_nb(m)
    ex_score = extras_score_nb(ex_raw)

    ws = float(weights.get("structure", 0.0))
    wz = float(weights.get("zones", 0.0))
    wi = float(weights.get("imbalance", 0.0))
    wm = float(weights.get("mtf", 0.0))
    wex = float(weights.get("extras", 0.0))

    conf = combine_nb(
        s_score, z_score, im_score, m_score, ex_score,
        ws, wz, wi, wm, wex
    )

    out = df.copy()
    out["conf_components_structure"] = s_score
    out["conf_components_zones"] = z_score
    out["conf_components_imbalance"] = im_score
    out["conf_components_mtf"] = m_score
    out["conf_components_extras"] = ex_score
    out["conf_score"] = conf

    out["conf_flag_strong_entry"] = (out["conf_score"] > 0.7).astype(np.int32)  # مثال threshold
    out["conf_flag_filter_pass"] = (out["conf_score"] > 0.5).astype(np.int32)    # مثال دیگر


    if anti_lookahead:
        shift_cols = [
            "conf_score",
            "conf_components_structure",
            "conf_components_zones",
            "conf_components_imbalance",
            "conf_components_mtf",
            "conf_components_extras",
        ]
        out[shift_cols] = out[shift_cols].shift(1)



    return out
