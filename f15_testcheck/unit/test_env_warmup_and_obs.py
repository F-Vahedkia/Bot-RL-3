# f16_tests/unit/test_env_warmup_and_obs.py
# Run: pytest -q f16_tests/unit/test_env_warmup_and_obs.py

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from f02_utils.config_loader import load_config
from f02_utils.config_operations import _deep_get
from f06_env.OLD_files.OLD_trading_env import TradingEnv, EnvConfig
import f06_env.OLD_files.OLD_utils as U  # لازم است تا پچ conftest روی paths_from_cfg اثر کند.

def _write_minimal_processed(cfg, symbol: str = "XAUUSD", tf: str = "M1", n: int = 800) -> Path:
    """ساخت دیتای مصنوعی و ذخیرهٔ Parquet در test_process (به واسطهٔ conftest)."""
    paths = U.paths_from_cfg(cfg)
    sym_dir = paths["processed"] / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2024-06-01 00:00:00", periods=n, freq="1min", tz="UTC")
    base = np.linspace(100.0, 120.0, n)
    df = pd.DataFrame({
        "time": ts,
        "open": base,
        "high": base + 0.5,
        "low":  base - 0.5,
        "close": base + 0.1,
        "tick_volume": np.random.randint(10, 30, size=n),
        "spread": np.random.randint(5, 15, size=n),
    }).set_index("time")

    # مهم: نام‌گذاری مطابق انتظار TradingEnv (مثلاً M1_close و ...)
    df = df.rename(columns={
        "open": f"{tf}_open",
        "high": f"{tf}_high",
        "low":  f"{tf}_low",
        "close": f"{tf}_close",
        "tick_volume": f"{tf}_tick_volume",
        "spread": f"{tf}_spread",
    })

    out = sym_dir / f"{tf}.parquet"
    df.to_parquet(out)
    return out

@pytest.mark.usefixtures("_redirect_processed_to_test_process")
def test_warmup_and_obs_shape(monkeypatch):
    cfg = load_config()
    _ = _write_minimal_processed(cfg)

    # پچ ایمن برای جلوگیری از j2 == len(idx)
    def _safe_build_slices_from_ratios(idx, cfg_):
        rat = dict(_deep_get(cfg_, "env.split.ratios"))
        n = len(idx)
        j1 = int(n * float(rat["train"]))
        j2 = int(n * float(rat["train"] + rat["val"]))
        if j1 >= n: j1 = n - 2
        if j2 >= n: j2 = n - 1
        if j1 >= j2: j1 = max(0, j2 - 1)
        return {"train": (idx[0], idx[j1]),
                "val":   (idx[j1], idx[j2]),
                "test":  (idx[j2], idx[-1])}

    # خیلی مهم: پچِ نام مبتنی بر مسیر رشته‌ای در خود ماژول trading_env
    monkeypatch.setattr("f06_env.trading_env.build_slices_from_ratios",
                        _safe_build_slices_from_ratios, raising=False)
    # همچنین در utils (برای هر استفادهٔ غیرمستقیم)
    monkeypatch.setattr("f06_env.utils.build_slices_from_ratios",
                        _safe_build_slices_from_ratios, raising=False)

    env = TradingEnv(cfg, EnvConfig(symbol="XAUUSD", base_tf="M1", window_size=64, normalize=True))
    obs, info = env.reset(split="val")

    # warmup حداقل برابر window_size
    assert info.get("warmup_bars", 0) >= 64
    # شکلِ observation
    assert obs.shape[0] == 64
    assert obs.shape[1] >= 5  # OHLC + Volume (spread غیرفعال است)
