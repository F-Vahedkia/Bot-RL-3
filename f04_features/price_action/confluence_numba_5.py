# -*- coding: utf-8 -*-
# f04_features/price_action/confluence_numba_5.py
# Status: FINAL — NUMBA, PRODUCTION-GRADE

from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit, prange

# =========================================================
# Numba helpers (STRICT nopython-safe)
# =========================================================

@njit
def _clip01_nb(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@njit(parallel=True)
def _norm_distance_nb(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        d = dist[i]
        if d >= 0.0:
            out[i] = 1.0 / (1.0 + d)
        else:
            out[i] = 0.0
    return out


@njit(parallel=True)
def structure_score_nb(
    bos_up, bos_down, choch_up, choch_down, swing_hh, swing_ll
):
    n = bos_up.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        s = 0.0
        s += bos_up[i] * 0.50
        s += bos_down[i] * 0.50
        s += choch_up[i] * 0.35
        s += choch_down[i] * 0.35
        if swing_hh[i] == 1 or swing_ll[i] == 1:
            s += 0.15
        out[i] = _clip01_nb(s)
    return out


@njit(parallel=True)
def zones_score_nb(dist_sd, dist_ob):
    n = dist_sd.shape[0]
    out = np.zeros(n, dtype=np.float32)
    sd = _norm_distance_nb(dist_sd)
    ob = _norm_distance_nb(dist_ob)
    for i in prange(n):
        out[i] = _clip01_nb(sd[i] * 0.60 + ob[i] * 0.60)
    return out


@njit(parallel=True)
def imbalance_score_nb(dist_fvg, sweep_up, sweep_down):
    n = dist_fvg.shape[0]
    out = np.zeros(n, dtype=np.float32)
    fvg = _norm_distance_nb(dist_fvg)
    for i in prange(n):
        s = fvg[i] * 0.60
        s += sweep_up[i] * 0.25
        s += sweep_down[i] * 0.25
        out[i] = _clip01_nb(s)
    return out


@njit(parallel=True)
def mtf_score_nb(mtf_score, mtf_conflict, mtf_strength):
    n = mtf_score.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        s = mtf_score[i] * 0.80
        if mtf_conflict[i] == 1:
            s -= 0.50
        s += mtf_strength[i] * 0.20
        out[i] = _clip01_nb(s)
    return out


@njit(parallel=True)
def extras_score_nb(
    regime_channel, regime_spike, regime_range,
    breakout_up, breakout_down,
    retest_up, retest_down,
    fail_break_up, fail_break_down,
    micro_up, micro_down, micro_q
):
    n = regime_channel.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        s = 0.0
        s += regime_channel[i] * 0.45
        s += regime_spike[i] * 0.25
        s -= regime_range[i] * 0.20

        s += breakout_up[i] * 0.20
        s += breakout_down[i] * 0.20
        s += retest_up[i] * 0.10
        s += retest_down[i] * 0.10
        s -= fail_break_up[i] * 0.15
        s -= fail_break_down[i] * 0.15

        s += micro_up[i] * 0.15
        s += micro_down[i] * 0.15
        s += micro_q[i] * 0.20

        # IMPORTANT:
        # extras فقط مثبت در نظر گرفته می‌شود
        if s < 0.0:
            s = 0.0

        out[i] = _clip01_nb(s)
    return out


@njit(parallel=True)
def final_conf_nb(
    s, z, i, m, e,
    w_struct, w_zones, w_imbal, w_mtf, w_extra
):
    """
    طراحی شده طوری که:
    - وقتی extras > 0 اضافه شود
    - میانگین conf_score کاهش پیدا نکند
    """
    n = s.shape[0]
    out = np.zeros(n, dtype=np.float32)

    base_sum = w_struct + w_zones + w_imbal + w_mtf
    if base_sum <= 0.0:
        base_sum = 1.0

    for k in prange(n):
        base = (
            s[k] * (w_struct / base_sum) +
            z[k] * (w_zones / base_sum) +
            i[k] * (w_imbal / base_sum) +
            m[k] * (w_mtf / base_sum)
        )
        # extras به صورت افزایشی
        val = base + e[k] * w_extra
        out[k] = _clip01_nb(val)

    return out


# =========================================================
# Public API
# =========================================================

def build_confluence(
    df: pd.DataFrame,
    *,
    weights: dict | None = None,
    strong_entry_threshold: float = 0.70,
    filter_threshold: float = 0.35,
    anti_lookahead: bool = True,
) -> pd.DataFrame:

    out = df.copy()

    if weights is None:
        weights = {
            "structure": 0.30,
            "zones": 0.30,
            "imbalance": 0.20,
            "mtf": 0.20,
            "extras": 0.00,
        }

    w_struct = float(weights.get("structure", 0.30))
    w_zones  = float(weights.get("zones", 0.30))
    w_imbal  = float(weights.get("imbalance", 0.20))
    w_mtf    = float(weights.get("mtf", 0.20))
    w_extra  = float(weights.get("extras", 0.00))

    n = len(out)

    def f(col, default=0.0):
        if col in out.columns:
            return out[col].to_numpy(np.float32)
        return np.full(n, default, dtype=np.float32)

    def b(col):
        if col in out.columns:
            return out[col].to_numpy(np.int8)
        return np.zeros(n, dtype=np.int8)

    s_score = structure_score_nb(
        b("bos_up"), b("bos_down"),
        b("choch_up"), b("choch_down"),
        b("swing_hh"), b("swing_ll")
    )

    z_score = zones_score_nb(
        f("dist_to_sd", -1.0),
        f("dist_to_ob", -1.0)
    )

    i_score = imbalance_score_nb(
        f("dist_to_fvg_mid", -1.0),
        b("sweep_up"), b("sweep_down")
    )

    m_score = mtf_score_nb(
        f("mtf_confluence_score", 0.5),
        b("mtf_conflict"),
        f("mtf_strength", 0.0)
    )

    e_score = extras_score_nb(
        f("regime_channel"), f("regime_spike"), f("regime_range"),
        b("breakout_up"), b("breakout_down"),
        b("retest_up"), b("retest_down"),
        b("fail_break_up"), b("fail_break_down"),
        b("micro_channel_up"), b("micro_channel_down"),
        f("micro_channel_quality")
    )

    conf = final_conf_nb(
        s_score, z_score, i_score, m_score, e_score,
        w_struct, w_zones, w_imbal, w_mtf, w_extra
    )

    if anti_lookahead:
        s_score = np.roll(s_score, 1); s_score[0] = 0.0
        z_score = np.roll(z_score, 1); z_score[0] = 0.0
        i_score = np.roll(i_score, 1); i_score[0] = 0.0
        m_score = np.roll(m_score, 1); m_score[0] = 0.0
        e_score = np.roll(e_score, 1); e_score[0] = 0.0
        conf    = np.roll(conf, 1);    conf[0]    = 0.5

    out["conf_components_structure"] = s_score
    out["conf_components_zones"] = z_score
    out["conf_components_imbalance"] = i_score
    out["conf_components_mtf"] = m_score
    out["conf_components_extras"] = e_score

    out["conf_score"] = conf.astype("float32")
    out["conf_flag_strong_entry"] = (conf >= strong_entry_threshold).astype("int8")
    out["conf_flag_filter_pass"] = (conf >= filter_threshold).astype("int8")

    return out
