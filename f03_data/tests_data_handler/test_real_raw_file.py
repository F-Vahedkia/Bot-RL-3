# f03_data/tests_data_handler/test_real_raw_file.py

# Run: pytest f03_data/tests_data_handler/test_real_raw_file.py -s -v

import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import pandas as pd

from f02_utils.config_completer import config_completer
from f02_utils.config_path_funcs import (
    resolve_raw_dir,
    full_file_path,
)


def test_real_bitcoin_raw_files():

    # ==============================================================
    # 1. Load config
    # ==============================================================

    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../f01_config/config.yaml",
        )
    )

    cfg = config_completer(
        config_path,
        enable_env_override=True,
    )

    symbol = "BITCOIN"

    # ==============================================================
    # 2. Check symbol
    # ==============================================================

    assert symbol in cfg["__timeframes_dict"]

    timeframes = cfg["__timeframes_dict"][symbol]

    base_tf = cfg["__base_tfs_dict"][symbol]

    print()
    print("=" * 80)
    print(f"SYMBOL       : {symbol}")
    print(f"BASE TF      : {base_tf}")
    print(f"TIMEFRAMES   : {timeframes}")
    print("=" * 80)

    # ==============================================================
    # 3. Resolve raw directory
    # ==============================================================

    raw_dir = resolve_raw_dir(cfg)

    print(f"RAW DIR      : {raw_dir}")
    print(f"RAW DIR EXIST: {raw_dir.exists()}")
    print("=" * 80)

    assert raw_dir.exists()

    # ==============================================================
    # 4. Inspect every configured timeframe
    # ==============================================================

    for timeframe in [base_tf] + list(timeframes):

        print()
        print("-" * 80)
        print(f"TIMEFRAME: {timeframe}")
        print("-" * 80)

        # ----------------------------------------------------------
        # Parquet
        # ----------------------------------------------------------

        parquet_path = full_file_path(
            raw_dir,
            symbol,
            timeframe,
            fmt="parquet",
        )

        print(f"PARQUET PATH   : {parquet_path}")
        print(f"PARQUET EXISTS : {parquet_path.exists()}")

        # ----------------------------------------------------------
        # CSV
        # ----------------------------------------------------------

        csv_path = full_file_path(
            raw_dir,
            symbol,
            timeframe,
            fmt="csv",
        )

        print(f"CSV PATH       : {csv_path}")
        print(f"CSV EXISTS     : {csv_path.exists()}")

        # ----------------------------------------------------------
        # Select actual file
        # ----------------------------------------------------------

        if parquet_path.exists():
            path = parquet_path
            fmt = "parquet"

        elif csv_path.exists():
            path = csv_path
            fmt = "csv"

        else:
            print("NO RAW FILE FOUND")
            continue

        print(f"SELECTED FILE  : {path}")
        print(f"FORMAT         : {fmt}")
        print(f"FILE SIZE      : {path.stat().st_size:,} bytes")

        # ==========================================================
        # 5. Read WITHOUT _read_raw_df
        # ==========================================================

        if fmt == "parquet":

            df = pd.read_parquet(path)

        else:

            df = pd.read_csv(path)

        # ==========================================================
        # 6. Inspect actual structure
        # ==========================================================

        print()
        print(f"DATAFRAME TYPE : {type(df)}")
        print(f"INDEX TYPE     : {type(df.index)}")
        print(f"INDEX NAME     : {df.index.name}")
        print(f"INDEX TZ       : {getattr(df.index, 'tz', None)}")

        print()
        print("COLUMNS:")
        print(list(df.columns))

        print()
        print("DTYPES:")
        print(df.dtypes)

        print()
        print("HEAD:")
        print(df.head())

        print()
        print("TAIL:")
        print(df.tail())

        print()
        print(f"ROWS           : {len(df)}")

        # ==========================================================
        # 7. The exact condition that causes your error
        # ==========================================================

        is_datetime_index = isinstance(
            df.index,
            pd.DatetimeIndex,
        )

        has_time_column = "time" in df.columns

        print()
        print(f"IS DATETIME INDEX : {is_datetime_index}")
        print(f"HAS TIME COLUMN   : {has_time_column}")

        if not is_datetime_index and not has_time_column:

            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("THIS FILE CAUSES THE _read_raw_df ERROR")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print()

            # عمداً assert نمی‌کنیم؛
            # می‌خواهیم تمام اطلاعات فایل را ببینیم.