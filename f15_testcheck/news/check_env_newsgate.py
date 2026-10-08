# -*- coding: utf-8 -*-
# f15_scripts/check_env_newsgate.py
# Status in (Bot-RL-3): Completed

"""Quick smoke-test for NewsGate inside Env (English logs / Persian comments)."""
from __future__ import annotations
import logging, pandas as pd

def _setup():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")

def main():
    _setup()
    # config + gate
    from f02_utils.config_loader import load_config
    from f06_news.integration import make_news_gate
    cfg = load_config("f01_config/config.yaml", enable_env_override=True)
    gate = make_news_gate(cfg, symbol="XAUUSD")

    # Env import (adapt if your path differs)
    try:
        from f06_env.OLD_files.OLD_trading_env import TradingEnv
    except Exception:
        from trading_env import TradingEnv  # fallback

    # Load your processed dataset (base_tf=M1)
    df = pd.read_parquet("f03_data/processed/XAUUSD/M1.parquet")

    # Build env (fill your own params as needed)
    env = TradingEnv(df=df, news_gate=gate)

    # Move to a window around 2025-08-28 12:00Z..13:00Z
    start_ts = pd.Timestamp("2025-08-28 11:58:00Z")
    start_idx = df.index.get_indexer([start_ts], method="nearest")[0]
    env._t = max(start_idx, 0)  # adjust to your indexer if different

    steps = 200
    for i in range(steps):
        # dummy action: try to open long to see freeze in action
        try:
            action = env.ACTIONS.OPEN_LONG
        except Exception:
            action = 1  # if you use ints

        obs, reward, done, info = env.step(action)
        if i % 10 == 0:
            ts = df.index[env._t]
            st = info.get("news_gate", {})
            print(f"{ts} | freeze={st.get('freeze')} reduce={st.get('reduce_risk')} reason={st.get('reason')}")
        if done:
            break

    print("Smoke test finished.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
