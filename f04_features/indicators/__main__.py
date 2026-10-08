from .__init__ import main as _main
if __name__ == "__main__":
    raise SystemExit(_main())




"""
1- core.py registry:
        "sma": make_sma,
        "ema": make_ema,
        "wma": make_wma,
        "rsi": make_rsi,
        "roc": make_roc,
        "atr": make_atr,
        "macd": make_macd,
        "bbands": make_bbands,
        "keltner": make_keltner,
        "stoch": make_stoch,
        "cci": make_cci,
        "mfi": make_mfi,
        "obv": make_obv,
        "wr": make_wr,
        "sar": make_sar,
        "ha": make_ha,
        "tr": make_tr,

2- divergence.py registry
        "div_macd": make_div_macd,
        "div_rsi": make_div_rsi,

3- extras_chanel.py registry:
        "chaikin_vol": make_ch_vol,
        "donchian": make_donchian,

4- extras_trend.py registry:
        "supertrend": make_supertrend,
        "adx": make_adx,
        "aroon": make_aroon,
        "kama": make_kama,
        "dema": make_dema,
        "tema": make_tema,
        "hma": make_hma,
        "ichimoku": make_ichimoku,
        -------------------------
        "ma_slope_{method}_{window}"
        "rsi_is_overbought": is_ob.astype(bool),
        "rsi_is_oversold": is_os.astype(bool),
        "rsi_mid_zone": is_mid.astype(bool),

from levels.py registry
        "pivots": make_pivots,
        "sr": make_sr,
        "fibo": make_fibo

"""