import numpy as np
import pandas as pd
import io

'''
csv_text = """index,open,high,low,close
1,100.00,100.50,99.80,100.30
2,100.30,100.80,100.00,100.60
3,100.60,101.00,100.40,100.90
4,100.90,101.30,100.70,101.10
5,101.10,101.50,100.90,101.30
6,101.30,101.70,101.10,101.50
7,101.50,101.90,101.30,101.70
8,101.70,102.10,101.50,101.90
9,101.90,102.00,101.80,101.90
10,101.90,102.00,101.80,101.90
11,101.90,102.00,101.80,101.90
12,102.00,103.50,101.90,103.20
13,103.20,103.80,102.90,103.60
14,103.60,103.90,103.20,103.30
15,103.20,104.20,102.80,103.00
16,103.00,103.40,102.60,103.10
17,103.10,103.50,102.90,103.40
18,103.40,104.00,103.20,103.80
19,103.80,104.20,103.60,104.00
20,104.60,105.20,104.50,105.00
21,105.00,105.40,104.80,105.10
22,105.10,105.50,104.90,105.40
23,105.40,106.60,105.30,106.30
24,106.30,106.50,105.00,105.20
25,105.20,105.40,104.60,104.80
26,104.80,105.00,104.40,104.60
27,104.60,105.20,104.50,105.00
28,105.00,105.40,103.90,104.10
29,104.10,104.30,103.70,103.80
30,103.80,104.00,103.60,103.90
31,103.90,104.20,103.50,104.10
32,104.10,104.50,104.00,104.40
33,104.40,104.60,103.20,103.40
34,103.40,103.60,102.80,103.00
35,103.00,103.20,102.60,102.80
36,102.80,103.00,102.40,102.50
37,102.50,103.80,102.40,103.60
38,103.60,103.90,101.80,102.00
39,102.00,102.20,101.60,101.80
40,101.80,101.90,100.80,101.00
41,101.00,101.40,100.60,101.20
42,101.20,102.80,101.10,102.60
43,102.60,102.90,101.80,102.00
44,102.00,103.80,101.90,103.60
45,103.60,104.00,103.20,103.40
46,103.40,104.60,103.30,104.40
47,104.40,104.80,103.90,104.10
48,104.10,105.80,104.00,105.40
49,105.40,105.60,104.20,104.40
50,104.40,106.20,103.80,105.90
"""
df = pd.read_csv(io.StringIO(csv_text), index_col="index")



print("DF ready:", df.shape)
''''''
print("========== old smoke test ========== 1")
out = make_fvg(df, lookback=2, atr_window=3, min_size_pct_of_atr=0.0)
print({k: out[k].astype(bool).sum() if out[k].dtype=='int8' else out[k].notna().sum() for k in out})
''''''
print("========== FVG test ========== 2")
out = make_fvg(df, lookback=2, atr_window=5, min_size_pct_of_atr=0.25, max_bars_alive=3)
print("FVG keys:", sorted(out.keys())[:5], "...")  # sanity
print("FVG flags sum:",
      int(out["fvg_up"].sum()), int(out["fvg_dn"].sum()),
      int(out["fvg_alive_n"].sum()), int(out["fvg_filled_window"].sum()))
print("FVG score (nonzero):", int((out["fvg_score"]>0).sum()))
''''''
print("========== Supply/Demand ========== 3")
out = make_sd(df, base_len=2, atr_window=5, base_atr_max=1.2,
              impulse_atr_min=0.8, max_bars_alive=5)
print("SD keys:", sorted(out.keys())[:5], "...")
print("SD flags sum:",
      int(out["sd_born"].sum()), int(out["sd_alive_n"].sum()),
      int(out["sd_filled_window"].sum()))
print("SD score (nonzero):", int((out["sd_score"]>0).sum()))

print("========== Order Block ========== 4")
out = make_ob(df, atr_window=5, body_atr_min=0.4,
              wick_ratio_max=0.8, bos_lookback=5, max_bars_alive=3)
print("OB keys:", sorted(out.keys())[:5], "...")
print("OB flags sum:",
      int(out["ob_born"].sum()), int(out["ob_alive_n"].sum()),
      int(out["ob_filled_window"].sum()))
print("OB score (nonzero):", int((out["ob_score"]>0).sum()))

print("========== Liquidity Sweep ========== 5")
out = make_liq_sweep(df, lookback=5, atr_window=5, min_tail_atr=0.4, max_bars_alive=3)
print("LS keys:", sorted(out.keys())[:5], "...")
print("LS flags sum:",
      int(out["ls_born"].sum()), int(out["ls_alive_n"].sum()),
      int(out["ls_filled_window"].sum()))
print("LS score (nonzero):", int((out["ls_score"]>0).sum()))

print("========== Breaker/Flip ========== 6")
out = make_breaker_flip(df, atr_window=5, ob_body_atr_min=0.3,
                        ob_wick_ratio_max=1.2, ob_bos_lookback=5,
                        lookback=10, max_bars_alive=5)
print("BF keys:", sorted(out.keys())[:5], "...")
print("BF flags sum:",
      int(out["bf_born"].sum()), int(out["bf_alive_n"].sum()),
      int(out["bf_filled_window"].sum()))
print("BF score (nonzero):", int((out["bf_score"]>0).sum()))

print("========== SR Fusion ========== 7")
out = make_sr_fusion(df, ema_span=3, age_norm=4.0,
                     enter_th=0.35, exit_th=0.25,
                     cooldown_bars=2, min_conf=0.2, tie_eps=0.02,
                     max_bars_alive=3)
print("SR keys:", sorted(out.keys()))
print("SR on sum:", int(out["sr_on"].sum()))
print("SR score (nonzero):", int((out["sr_score"]>0).sum()))
'''


# --- تست برای داده های واقعی ---------------------------------------
'''
from f04_features.indicators.sr_advanced import make_breaker_flip

df = pd.read_parquet("f03_data/processed/XAUUSD/H1.parquet")[["H1_open","H1_high","H1_low","H1_close"]]
df.columns = ["open","high","low","close"]
df = df.tail(20000)  # برای سرعت
out = make_breaker_flip(df, atr_window=14, ob_body_atr_min=0.35, ob_wick_ratio_max=1.0,
                        ob_bos_lookback=10, lookback=20, max_bars_alive=5)
print("BF:", int(out["bf_born"].sum()), int(out["bf_alive_n"].sum()), int((out["bf_score"]>0).sum()))
'''


# --- تست کوچک برای 5 گانه همراه با فیوژن ---------------------------
import pandas as pd
from f04_features.OLD.feature_registry import ADV_INDICATOR_REGISTRY

# Load H1 data
df = pd.read_parquet("f03_data/processed/XAUUSD/H1.parquet")[["H1_open","H1_high","H1_low","H1_close"]]
df.columns = ["open","high","low","close"]
df = df.tail(20000)  # speed-up

# Use registry so config.yaml injection is applied
get = ADV_INDICATOR_REGISTRY
out_fvg = get["fvg"](df)
out_sd  = get["supply_demand"](df)
out_ob  = get["order_block"](df)
out_ls  = get["liq_sweep"](df)
out_bf  = get["breaker_flip"](df)
out_sr  = get["sr_fusion"](df)

print("FVG/SD/OB/LS/BF/SR:",
      int((out_fvg["fvg_score"]>0).sum()),
      int((out_sd["sd_score"]>0).sum()),
      int((out_ob["ob_score"]>0).sum()),
      int((out_ls["ls_score"]>0).sum()),
      int((out_bf["bf_score"]>0).sum()),
      int((out_sr["sr_score"]>0).sum()))

'''

# --- چک سریع برای اطمینان از کارکرد تزریق از کانفیگ
from f04_features.indicators.registry import ADV_INDICATOR_REGISTRY
df = pd.read_parquet("f03_data/processed/XAUUSD/H1.parquet")[["H1_open","H1_high","H1_low","H1_close"]]
df.columns = ["open","high","low","close"]
df = df.tail(20000)  # برای سرعت
print("CFG test:", ADV_INDICATOR_REGISTRY["supply_demand"](df).__class__)  # فقط باید بدون خطا اجرا شه

# --- تست مستقیمِ نرم‌تر (برای اطمینان از اثر پارامترها؛ موقت)
out_sd = ADV_INDICATOR_REGISTRY["supply_demand"](df, base_atr_max=1.0, impulse_atr_min=0.8)
print("SD score>0:", int((out_sd["sd_score"]>0).sum()))
'''