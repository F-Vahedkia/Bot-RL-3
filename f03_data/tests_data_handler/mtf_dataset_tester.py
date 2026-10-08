# f03_data/tests_data_handler/mtf_dataset_tester.py
# Date reviewed:
#    1405/05/24- 20:45 --> run result if OK for 5 tests

# Run: pytest -v -s f03_data/tests_data_handler/mtf_dataset_tester.py

import pandas as pd
from f03_data.mtf_dataset import MTFDataset

# ---------------------------------------------------------
def make_df(start="2026-01-01", rows=5):
    idx = pd.date_range(start, periods=rows, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(rows),
            "high": range(rows),
            "low": range(rows),
            "close": range(rows),
        },
        index=idx,
    )


# ---------------------------------------------------------
def test_add_and_get():

    ds = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )

    m1 = make_df()
    h1 = make_df(rows=3)

    ds.add("M1", m1)
    ds.add("H1", h1)

    assert ds.get("M1").equals(m1)
    assert ds["H1"].equals(h1)


# ---------------------------------------------------------
def test_timeframes():

    ds = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )

    ds.add("M1", make_df())
    ds.add("H1", make_df())

    assert set(ds.timeframes) == {"M1", "H1"}


# ---------------------------------------------------------
def test_copy():

    ds = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )

    ds.add("M1", make_df())

    cp = ds.copy()

    cp["M1"].iloc[0, 0] = 999

    assert ds["M1"].iloc[0, 0] != 999


# ---------------------------------------------------------
def test_apply():

    ds = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )

    ds.add("M1", make_df())
    ds.add("H1", make_df())

    def f(df):
        out = df.copy()
        out["x"] = 1
        return out

    ds.apply(f)

    assert "x" in ds["M1"].columns
    assert "x" in ds["H1"].columns


# ---------------------------------------------------------
def test_apply_each():

    ds = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )

    ds.add("M1", make_df())
    ds.add("H1", make_df())

    def f(df, tf):
        out = df.copy()
        out["tf"] = tf
        return out

    ds.apply_each(f)

    assert (ds["M1"]["tf"] == "M1").all()
    assert (ds["H1"]["tf"] == "H1").all()


# ---------------------------------------------------------
