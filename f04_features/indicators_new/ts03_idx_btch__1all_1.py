"""
تستر جامع برای 12 اندیکاتور تکنیکال
این فایل توابع batch را با داده‌های واقعی تست می‌کند

Run: python -m f04_features.indicators_new.ts03_idx_btch__1all_1
"""

import pandas as pd
import numpy as np
from datetime import datetime
import f04_features.indicators_new.indicators_B_batch as ind

def test_indicators(df):
    """
    تست 12 اندیکاتور با داده واقعی
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم با ستون‌های open, high, low, close, tick_volume
    """
    
    # تغییر نام ستون tick_volume به volume
    df = df.rename(columns={'tick_volume': 'volume'})
    
    print("=" * 80)
    print("Start indicators test")
    print(f"Rows number: {len(df)}")
    print("=" * 80)
    
    results = {}
    # =========================================================================
    # --- 1. SMA - Simple Moving Average --------------------------------------
    print("\n[1/23] test of SMA...")
    try:
        sma_20 = ind.sma_batch_df(df, column='close', period=20, result_col='sma_20')
        results['sma'] = sma_20['sma_20']

        print(f"  ✓ SMA: {sma_20['sma_20'].notna().sum()} valid values out of {len(sma_20)}")
        print(f"    columns: {list(sma_20.columns)}")
        print(f"    last value: {sma_20['sma_20'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 2. WMA - Weighted Moving Average ------------------------------------
    print("\n[2/23] test of WMA...")
    try:
        wma_20 = ind.wma_batch_df(df, column='close', period=20, result_col='wma_20')
        results['wma'] = wma_20['wma_20']

        print(f"  ✓ WMA: {wma_20['wma_20'].notna().sum()} valid values out of {len(wma_20)}")
        print(f"    columns: {list(wma_20.columns)}")
        print(f"    last value: {wma_20['wma_20'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 3. EMA - Exponential Moving Average ---------------------------------
    print("\n[3/23] test of EMA...")
    try:
        ema_20 = ind.ema_batch_df(df, column='close', period=20, result_col='ema_20')
        results['ema'] = ema_20['ema_20']

        print(f"  ✓ EMA: {ema_20['ema_20'].notna().sum()} valid values out of {len(ema_20)}")
        print(f"    columns: {list(ema_20.columns)}")
        print(f"    last value: {ema_20['ema_20'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 4. ROC - Rate of Change ---------------------------------------------
    print("\n[4/23] test of ROC...")
    try:
        roc_10 = ind.roc_batch_df(df, column='close', period=10, result_col='roc_10')
        results['roc'] = roc_10['roc_10']

        print(f"  ✓ ROC: {roc_10['roc_10'].notna().sum()} valid values out of {len(roc_10)}")
        print(f"    columns: {list(roc_10.columns)}")
        print(f"    last value: {roc_10['roc_10'].iloc[-1]:.4f}%")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 5. RSI - Relative Strength Index ------------------------------------
    print("\n[5/23] test of RSI...")
    try:
        rsi_14 = ind.rsi_batch_df(df, column='close', period=14, method='ema', result_col='rsi_14')
        results['rsi'] = rsi_14['rsi_14']

        print(f"  ✓ RSI: {rsi_14['rsi_14'].notna().sum()} valid values out of {len(rsi_14)}")
        print(f"    columns: {list(rsi_14.columns)}")
        print(f"    last value: {rsi_14['rsi_14'].iloc[-1]:.2f}")
        print(f"    range: [{rsi_14['rsi_14'].min():.2f}, {rsi_14['rsi_14'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 6. True Range -------------------------------------------------------
    print("\n[6/23] test of True Range...")
    try:
        tr = ind.truerange_batch_df(df, high_column='high', low_column='low', 
                                     close_column='close', result_col='true_range')
        results['true_range'] = tr['true_range']

        print(f"  ✓ True Range: {tr['true_range'].notna().sum()} valid values out of {len(tr)}")
        print(f"    columns: {list(tr.columns)}")
        print(f"    last value: {tr['true_range'].iloc[-1]:.4f}")
        print(f"    average: {tr['true_range'].mean():.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 7. ATR - Average True Range -----------------------------------------
    print("\n[7/23] test of ATR...")
    try:
        atr_14 = ind.atr_batch_df(df, high_column='high', low_column='low', close_column='close',
                                  period=14, method='wilder',  result_col='atr_14')
        results['atr'] = atr_14['atr_14']

        print(f"  ✓ ATR: {atr_14['atr_14'].notna().sum()} valid values out of {len(atr_14)}")
        print(f"    columns: {list(atr_14.columns)}")
        print(f"    last value: {atr_14['atr_14'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 8. MACD - Moving Average Convergence Divergence ---------------------
    print("\n[8/23] test of MACD...")
    try:
        macd_df = ind.macd_batch_df(df, column='close', fast=12, slow=26, signal=9,
                                     macd_col='macd', signal_col='macd_signal', 
                                     hist_col='macd_hist')
        results['macd'] = macd_df['macd']
        results['macd_signal'] = macd_df['macd_signal']
        results['macd_hist'] = macd_df['macd_hist']

        print(f"  ✓ MACD: {len(macd_df.columns)} output columns")
        print(f"    columns: {list(macd_df.columns)}")
        print(f"    last MACD: {macd_df['macd'].iloc[-1]:.4f}")
        print(f"    last Signal: {macd_df['macd_signal'].iloc[-1]:.4f}")
        print(f"    last Histogram: {macd_df['macd_hist'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 9. Bollinger Bands --------------------------------------------------
    print("\n[9/23] test of Bollinger Bands...")
    try:
        bb_df = ind.bollinger_batch_df(df, column='close', period=20, multiplier=2.0,
                                        upper_col='bb_upper', middle_col='bb_middle',
                                        lower_col='bb_lower', width_col='bb_width',
                                        percent_col='bb_percent')
        results['bb_upper'] = bb_df['bb_upper']
        results['bb_middle'] = bb_df['bb_middle']
        results['bb_lower'] = bb_df['bb_lower']
        results['bb_width'] = bb_df['bb_width']
        results['bb_percent'] = bb_df['bb_percent']
        
        print(f"  ✓ Bollinger: {len(bb_df.columns)} output columns")
        print(f"    columns: {list(bb_df.columns)}")
        print(f"    last Upper: {bb_df['bb_upper'].iloc[-1]:.4f}")
        print(f"    last Middle: {bb_df['bb_middle'].iloc[-1]:.4f}")
        print(f"    last Lower: {bb_df['bb_lower'].iloc[-1]:.4f}")
        print(f"    last %B: {bb_df['bb_percent'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 10. Keltner Channel -------------------------------------------------
    print("\n[10/23] test of Keltner Channel...")
    try:
        kc_df = ind.keltner_batch_df(df, high_column='high', low_column='low', close_column='close',
                                      period=20, multiplier=2.0,
                                      upper_col='kc_upper', middle_col='kc_middle',
                                      lower_col='kc_lower', width_col='kc_width',
                                      percent_col='kc_percent')
        results['kc_upper'] = kc_df['kc_upper']
        results['kc_middle'] = kc_df['kc_middle']
        results['kc_lower'] = kc_df['kc_lower']
        results['kc_width'] = kc_df['kc_width']
        results['kc_percent'] = kc_df['kc_percent']

        print(f"  ✓ Keltner: {len(kc_df.columns)} output columns")
        print(f"    columns: {list(kc_df.columns)}")
        print(f"    last Upper: {kc_df['kc_upper'].iloc[-1]:.4f}")
        print(f"    last Middle: {kc_df['kc_middle'].iloc[-1]:.4f}")
        print(f"    last Lower: {kc_df['kc_lower'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 11. Stochastic Oscillator -------------------------------------------
    print("\n[11/23] test of Stochastic...")
    try:
        stoch_df = ind.stochastic_batch_df(df, high_column='high', low_column='low', close_column='close',
                                            k_period=14, d_period=3, smooth_k=3, method='sma',
                                            k_col='stoch_k', d_col='stoch_d')
        results['stoch_k'] = stoch_df['stoch_k']
        results['stoch_d'] = stoch_df['stoch_d']
        
        print(f"  ✓ Stochastic: {len(stoch_df.columns)} output columns")
        print(f"    columns: {list(stoch_df.columns)}")
        print(f"    last %K: {stoch_df['stoch_k'].iloc[-1]:.2f}")
        print(f"    last %D: {stoch_df['stoch_d'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 12. CCI - Commodity Channel Index -----------------------------------
    print("\n[12/23] test of CCI...")
    try:
        cci_20 = ind.cci_batch_df(df, high_column='high', low_column='low', close_column='close',
                                   period=20, result_col='cci_20')
        results['cci'] = cci_20['cci_20']

        print(f"  ✓ CCI: {cci_20['cci_20'].notna().sum()} valid values out of {len(cci_20)}")
        print(f"    columns: {list(cci_20.columns)}")
        print(f"    last value: {cci_20['cci_20'].iloc[-1]:.2f}")
        print(f"    range: [{cci_20['cci_20'].min():.2f}, {cci_20['cci_20'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")

    # --- 13. MFI - Money Flow Index ------------------------------------------
    print("\n[13/23] test of MFI...")
    try:
        mfi_14 = ind.mfi_batch_df(df, high_column='high', low_column='low', close_column='close',
                                   volume_column='volume', period=14, result_col='mfi_14')
        results['mfi'] = mfi_14['mfi_14']

        print(f"  ✓ MFI: {mfi_14['mfi_14'].notna().sum()} valid values out of {len(mfi_14)}")
        print(f"    columns: {list(mfi_14.columns)}")
        print(f"    last value: {mfi_14['mfi_14'].iloc[-1]:.2f}")
        print(f"    range: [{mfi_14['mfi_14'].min():.2f}, {mfi_14['mfi_14'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 14. OBV - On-Balance Volume -----------------------------------------
    print("\n[14/23] test of OBV...")
    try:
        obv = ind.obv_batch_df(df, close_column='close', volume_column='volume', result_col='obv')
        results['obv'] = obv['obv']

        print(f"  ✓ OBV: {obv['obv'].notna().sum()} valid values out of {len(obv)}")
        print(f"    columns: {list(obv.columns)}")
        print(f"    last value: {obv['obv'].iloc[-1]:.2f}")
        print(f"    range: [{obv['obv'].min():.2f}, {obv['obv'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 15. Williams %R -----------------------------------------------------
    print("\n[15/23] test of Williams %R...")
    try:
        williamsr_14 = ind.williamsr_batch_df(df, high_column='high', low_column='low', close_column='close',
                                               period=14, min_periods=5, result_col='williamsr_14')
        results['williamsr'] = williamsr_14['williamsr_14']

        print(f"  ✓ Williams %R: {williamsr_14['williamsr_14'].notna().sum()} valid values out of {len(williamsr_14)}")
        print(f"    columns: {list(williamsr_14.columns)}")
        print(f"    last value: {williamsr_14['williamsr_14'].iloc[-1]:.2f}")
        print(f"    range: [{williamsr_14['williamsr_14'].min():.2f}, {williamsr_14['williamsr_14'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 16. Parabolic SAR ---------------------------------------------------
    print("\n[16/23] test of Parabolic SAR...")
    try:
        psar = ind.parabolicsar_batch_df(df, high_column='high', low_column='low',
                                  af_start=0.02, af_step=0.02, af_max=0.2, result_col='psar')
        results['psar'] = psar['psar']

        print(f"  ✓ Parabolic SAR: {psar['psar'].notna().sum()} valid values out of {len(psar)}")
        print(f"    columns: {list(psar.columns)}")
        print(f"    last value: {psar['psar'].iloc[-1]:.2f}")
        print(f"    range: [{psar['psar'].min():.2f}, {psar['psar'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 17. Heikin-Ashi -----------------------------------------------------
    print("\n[17/23] test of Heikin-Ashi...")
    try:
        ha = ind.heikinashi_batch_df(df, open_column='open', high_column='high', low_column='low', close_column='close',
                                      result_prefix='ha')
        results['ha_open'] = ha['ha_open']
        results['ha_high'] = ha['ha_high']
        results['ha_low'] = ha['ha_low']
        results['ha_close'] = ha['ha_close']

        print(f"  ✓ Heikin-Ashi: {ha['ha_close'].notna().sum()} valid values out of {len(ha)}")
        print(f"    columns: {list(ha.columns)}")
        print(f"    last ha_close: {ha['ha_close'].iloc[-1]:.2f}")
        print(f"    range: [{ha['ha_close'].min():.2f}, {ha['ha_close'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 18. Supertrend ------------------------------------------------------
    print("\n[18/23] test of Supertrend...")
    try:
        super_t = ind.supertrend_batch_df(df, high_column='high', low_column='low', close_column='close',
                                      period=10, multiplier=3.0, method='wilder',
                                      super_col='supertrend', direction_col='st_direction')
        results['supertrend'] = super_t['supertrend']
        results['st_direction'] = super_t['st_direction']

        print(f"  ✓ Supertrend: {super_t['supertrend'].notna().sum()} valid values out of {len(super_t)}")
        print(f"    columns: {list(super_t.columns)}")
        print(f"    last value: {super_t['supertrend'].iloc[-1]:.2f}")
        print(f"    range: [{super_t['supertrend'].min():.2f}, {super_t['supertrend'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 19. Aroon -----------------------------------------------------------
    print("\n[19/23] test of Aroon...")
    try:
        aroon = ind.aroon_batch_df(df, high_column='high', low_column='low', period=25,
                                    aroon_up_col='aroon_up', aroon_down_col='aroon_down',
                                    aroon_osc_col='aroon_osc')
        results['aroon_up'] = aroon['aroon_up']
        results['aroon_down'] = aroon['aroon_down']
        results['aroon_osc'] = aroon['aroon_osc']

        print(f"  ✓ Aroon: {aroon['aroon_up'].notna().sum()} valid values out of {len(aroon)}")
        print(f"    columns: {list(aroon.columns)}")
        print(f"    last aroon_up: {aroon['aroon_up'].iloc[-1]:.2f}")
        print(f"    range: [{aroon['aroon_up'].min():.2f}, {aroon['aroon_up'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 20. DEMA - Double Exponential Moving Average ------------------------
    print("\n[20/23] test of DEMA...")
    try:
        dema_20 = ind.dema_batch_df(df, column='close', period=20, result_col='dema_20')
        results['dema'] = dema_20['dema_20']

        print(f"  ✓ DEMA: {dema_20['dema_20'].notna().sum()} valid values out of {len(dema_20)}")
        print(f"    columns: {list(dema_20.columns)}")
        print(f"    last value: {dema_20['dema_20'].iloc[-1]:.2f}")
        print(f"    range: [{dema_20['dema_20'].min():.2f}, {dema_20['dema_20'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 21. TEMA - Triple Exponential Moving Average ------------------------
    print("\n[21/23] test of TEMA...")
    try:
        tema_20 = ind.tema_batch_df(df, column='close', period=20, result_col='tema_20')
        results['tema'] = tema_20['tema_20']

        print(f"  ✓ TEMA: {tema_20['tema_20'].notna().sum()} valid values out of {len(tema_20)}")
        print(f"    columns: {list(tema_20.columns)}")
        print(f"    last value: {tema_20['tema_20'].iloc[-1]:.2f}")
        print(f"    range: [{tema_20['tema_20'].min():.2f}, {tema_20['tema_20'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 22. KAMA - Kaufman Adaptive Moving Average --------------------------
    print("\n[22/23] test of KAMA...")
    try:
        kama_10 = ind.kama_batch_df(df, column='close', period=10, fast_span=2, slow_span=30,
                                     result_col='kama_10')
        results['kama'] = kama_10['kama_10']

        print(f"  ✓ KAMA: {kama_10['kama_10'].notna().sum()} valid values out of {len(kama_10)}")
        print(f"    columns: {list(kama_10.columns)}")
        print(f"    last value: {kama_10['kama_10'].iloc[-1]:.2f}")
        print(f"    range: [{kama_10['kama_10'].min():.2f}, {kama_10['kama_10'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- 23. HMA - Hull Moving Average ---------------------------------------
    print("\n[23/23] test of HMA...")
    try:
        hma_20 = ind.hma_batch_df(df, column='close', period=20, result_col='hma_20')
        results['hma'] = hma_20['hma_20']

        print(f"  ✓ HMA: {hma_20['hma_20'].notna().sum()} valid values out of {len(hma_20)}")
        print(f"    columns: {list(hma_20.columns)}")
        print(f"    last value: {hma_20['hma_20'].iloc[-1]:.2f}")
        print(f"    range: [{hma_20['hma_20'].min():.2f}, {hma_20['hma_20'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # =========================================================================

    # --- Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    df_result = pd.concat(results, axis=1)  # کلیدها به عنوان MultiIndex
    print(f"\n Result Columns: {list(df_result.columns)}")
    column_number = len(list(df_result.columns))
    print(f"\nNumber of created columns = {column_number}")

    print(f"Number of indicators tested: {len(results)}/{column_number}")
    # print(f"Successful indicators: {list(results.keys())}")

    return results


if __name__ == "__main__":

    # --- Load data -----------------------------------------------------
    t1 = datetime.now()
    data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
    t2 = datetime.now()

    # --- Preparing data ------------------------------------------------
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data.set_index("time", inplace=True)
    data = data.astype("float64")

    elapsed = round((t2 - t1).total_seconds(), 2)
    print(f"Time taken to load data: {elapsed} seconds, length_df:{len(data)}")

    if not {"open", "high", "low", "close"}.issubset(data.columns):
        raise ValueError("Data must contain open, high, low, close")

    df = data[-1_000_000:]
    
    # --- Running Tests -------------------------------------------------

    t1 = datetime.now()
    results = test_indicators(df)
    t2 = datetime.now()

    dt = t2-t1
    second = dt.total_seconds()
    sec_round = round(second, 2)
    print(f"Time taken to run 23 tests: {round(sec_round,2)} seconds, length_df:{len(df)}")
    print("\n✓ The tests ran successfully.")

    total_results = pd.concat([df[['high', 'low', 'close']].iloc[-1_000_000:], results['williamsr'].iloc[-1_000_000:]], axis=1)
    total_results.to_csv("williamsR.csv")

    # شمارش NaN در هر ستون
    nan_counts = total_results.isna().sum()
    print("Number of NaN in every column:")
    print(nan_counts)