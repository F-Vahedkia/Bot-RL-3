
# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import logging

from f03_data.data_handler_G import DataHandler, BuildParams, _setup_logging
from f02_utils.config_completer import config_completer

# --- loggging ----------------------------------
_setup_logging("INFO")

# =============================================================================
# Main
# =============================================================================

symbol = "XAUUSD"
timeframes= ["M4", "M20"]
required_bars = {"M4": 8, "M20": 10}

params = BuildParams(
    symbol=symbol,
    base_tf="M1",
    timeframes= timeframes,
    selected_tf=None,
    load_format="csv",

    mode = "number",
    # --- مربوط به مد number:
    start_lastrows=6,
    end_lastrows=4,

    # --- مربوط به مد time:
    start_time=pd.to_datetime("2026-05-31 00:00:00+00:00", utc=True),
    end_time=pd.to_datetime("2026-06-05 00:00:00+00:00", utc=True),

    # --- مربوط به مد periods:
    period_size = "20m",
    from_last_n = 8,
    to_last_n = 4,
    base_time = pd.to_datetime("2026-06-02 20:00:00").tz_localize("UTC"),
)

cfg = config_completer(enable_env_override=True)

print(cfg.get("__all_required_bars"))

# handler = DataHandler(cfg, symbol, required_bars)
# metadata_info = handler._inspect_raw_metadata(params)

# print(metadata_info["M4"])
# print(metadata_info["M20"])

# download_jobs = handler._download_required_data(metadata_info, params)
# print(download_jobs)
# ============================================================================= END