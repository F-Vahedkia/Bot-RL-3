# f16_tests/unit/test_normalization_skip_pre_normalized.py
# Run: pytest -q f16_tests/unit/test_normalization_skip_pre_normalized.py

import numpy as np
import pandas as pd
from pathlib import Path

from f02_utils.config_loader import load_config
from f06_env.OLD_files.OLD_trading_env import TradingEnv, EnvConfig
import f06_env.OLD_files.OLD_utils as U  # ← به‌جای import مستقیم تابع، ماژول را می‌آوریم تا پچِ conftest اثر کند.

def _write_processed_with_zcol(cfg, symbol="XAUUSD", tf="M1"):
    paths = U.paths_from_cfg(cfg)  # ← مسیر 'processed' با پچ به 'test_process' می‌رود.
    sym_dir = paths["processed"] / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    n = 600
    ts = pd.date_range("2024-07-01 00:00:00Z", periods=n, freq="1min")
    base = np.linspace(100, 110, n)
    z_feat = (base - base.mean()) / (base.std() or 1.0)  # شبیه ستون از قبل نرمال‌شده
    df = pd.DataFrame({
        "time": ts,
        "open": base,
        "high": base + 0.5,
        "low":  base - 0.5,
        "close": base + 0.1,
        "tick_volume": np.random.randint(10, 30, size=n),
        "spread": np.random.randint(5, 15, size=n),
        "z_dummy": z_feat,  # ← ستون با پیشوند z_
    }).set_index("time")
    df = df.rename(columns={
        "open": f"{tf}_open", "high": f"{tf}_high", "low": f"{tf}_low",
        "close": f"{tf}_close", "tick_volume": f"{tf}_tick_volume", "spread": f"{tf}_spread",
    })
    df = df.rename(columns={"z_dummy": "z__dummy"})

    df.to_parquet(sym_dir / f"{tf}.parquet")

def test_skip_pre_normalized_columns():
    cfg = load_config()
    _write_processed_with_zcol(cfg)

    env = TradingEnv(cfg, EnvConfig(symbol="XAUUSD", base_tf="M1", window_size=32, normalize=True))
    env.reset(split="train")

    # ستون‌های انتخاب‌شده برای scale نباید شامل z__dummy باشند
    assert "z__dummy" in env.obs_cols
    assert hasattr(env, "scale_cols")
    assert "z__dummy" not in env.scale_cols

    # یک observation بگیر و مطمئن شو مقدار z__dummy خراب نشده (حداقل از نظر واریانس)
    obs, _ = env.reset(split="val")
    # آخرین ردیف (سطر جدید)
    last_row = obs[-1, :]
    # اندیس z__dummy در obs
    i = env.obs_cols.index("z__dummy")
    # واریانس روی پنجره نباید نزدیک صفر شود (اثر scale دوباره)
    assert np.var(obs[:, i]) > 1e-6
