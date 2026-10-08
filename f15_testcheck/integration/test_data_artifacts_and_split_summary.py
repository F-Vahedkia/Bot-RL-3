# f16_tests/integration/test_data_artifacts_and_split_summary.py
# Run: pytest -q f16_tests/integration/test_data_artifacts_and_split_summary.py

import json, os
import numpy as np
import pandas as pd
from pathlib import Path

from f02_utils.config_loader import load_config
from f03_data.OLD.data_handler import DataHandler, BuildParams
from f06_env.OLD_files.OLD_trading_env import TradingEnv, EnvConfig

# ------------------------------
# دادهٔ خام مصنوعی برای XAUUSD/M1
# ------------------------------
def _make_raw(dir_: Path):
    sym = "XAUUSD"; tf = "M1"
    raw_dir = dir_ / "f03_data" / "raw" / sym
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = 2000
    ts = pd.date_range("2024-01-01 00:00:00Z", periods=n, freq="1min")
    base = 2000.0 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        "time": ts,
        "open": base,
        "high": base + 0.5,
        "low":  base - 0.5,
        "close": base + 0.1*np.random.randn(n),
        "tick_volume": np.random.randint(50, 500, size=n),
        "spread": np.random.randint(10, 30, size=n),
    })
    df.to_csv(raw_dir / f"{tf}.csv", index=False)

def test_artifacts_and_split_summary(tmp_path: Path):
    # 1) ساخت دادهٔ processed با DataHandler
    proj_root = Path(__file__).resolve().parents[1]
    os.chdir(proj_root)  # اطمینان از ریشهٔ پروژه
    _make_raw(proj_root)

    cfg = load_config()
    dh = DataHandler(cfg)
    df = dh.build(BuildParams(symbol="XAUUSD", base_tf="M1", timeframes=["M1"], prefer_parquet=True))
    out = dh.save(df, symbol="XAUUSD", base_tf="M1", fmt="parquet")
    
    import f06_env.OLD_files.OLD_utils as U
    from shutil import copyfile
    expected = U.paths_from_cfg(cfg)["processed"] / "XAUUSD" / "M1.parquet"
    expected.parent.mkdir(parents=True, exist_ok=True)
    if out != expected: copyfile(out, expected)


    # خروجی‌های مورد انتظار
    meta = out.with_suffix(".meta.json")
    manf = out.with_suffix(".manifest.json")
    assert out.exists() and meta.exists() and manf.exists()

    # 2) ساخت Env و reset روی val → باید split summary تولید کند
    env = TradingEnv(cfg, EnvConfig(symbol="XAUUSD", base_tf="M1", window_size=128, normalize=True))
    env.reset(split="val")

    #split_json = (out.parent / "M1.split.json")
    import f06_env.OLD_files.OLD_utils as U
    split_json = U.paths_from_cfg(cfg)["processed"] / "XAUUSD" / "M1.split.json"
    
    assert split_json.exists(), "Split summary json was not created!"

    data = json.loads(split_json.read_text(encoding="utf-8"))
    # warmup باید >= window_size یا margin انتخاب‌شده باشد
    assert data["warmup_bars"] >= 128
    # عدم هم‌پوشانی ساده: پایان train <= شروع val
    assert data["train"]["end"] <= data["val"]["start"]
