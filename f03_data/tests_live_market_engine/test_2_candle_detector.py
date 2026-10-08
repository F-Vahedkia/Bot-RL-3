# f03_data/tests_live_market_engine/test_2_candle_detector.py
# Date Reviewed:
#    1405/05/19-18:48 ==> run result is OK for 7 tests

# Run:
# pytest f03_data/tests_live_market_engine/test_2_candle_detector.py -v

# اجرای تمام فایلهای تستر داخل پوشه:
# pytest f03_data/tests_live_market_engine/ -v


import pytest
import pandas as pd
from zoneinfo import ZoneInfo
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f03_data.live_market_engine import CandleDetector


class TestCandleDetector:

    @pytest.fixture
    def detector(self):
        """Create CandleDetector with UTC timezone."""
        return CandleDetector(ZoneInfo("UTC"))

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLC dataframe."""
        dates = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=5,
            freq="1h",
            tz="UTC"
        )
        df = pd.DataFrame({
            "open": [1.1000, 1.1010, 1.1020, 1.1015, 1.1025],
            "high": [1.1010, 1.1020, 1.1030, 1.1025, 1.1035],
            "low": [1.0995, 1.1005, 1.1015, 1.1010, 1.1020],
            "close": [1.1005, 1.1015, 1.1025, 1.1020, 1.1030],
            "volume": [1000, 1200, 1100, 1300, 1250]
        }, index=dates)
        return df


    def test_first_call_returns_none(self, detector, sample_df):
        """First call should return None because there is no previous candle."""
        result = detector.detect("EURUSD", "H1", sample_df)
        assert result is None


    def test_new_candle_detection(self, detector, sample_df):
        """Test detecting a new candle."""

        # First call: initialize state with last candle at 03:00.
        detector.detect("EURUSD", "H1", sample_df.iloc[:4])

        # Second call: last candle is now 04:00.
        result = detector.detect("EURUSD", "H1", sample_df)

        assert result is not None
        assert result["symbol"] == "EURUSD"
        assert result["timeframe"] == "H1"
        assert result["close"] == 1.1030

    def test_no_new_candle(self, detector, sample_df):
        """Test that unchanged data produces no event."""

        # First call: initialize state.
        detector.detect("EURUSD", "H1", sample_df)

        # Second call: same last candle.
        result = detector.detect("EURUSD", "H1", sample_df)

        assert result is None

    def test_different_symbols_independent(self, detector, sample_df):
        """Test that different symbols maintain independent states."""

        # EURUSD: initialize state.
        detector.detect("EURUSD", "H1", sample_df.iloc[:4])

        # EURUSD: new candle.
        result_eurusd = detector.detect("EURUSD", "H1", sample_df)

        # GBPUSD: first call, therefore no event.
        result_gbpusd = detector.detect("GBPUSD", "H1", sample_df)

        assert result_eurusd is not None
        assert result_gbpusd is None

    def test_different_timeframes_independent(self, detector, sample_df):
        """Test that different timeframes maintain independent states."""

        # H1: initialize state.
        detector.detect("EURUSD", "H1", sample_df.iloc[:4])

        # H1: new candle.
        result_h1 = detector.detect("EURUSD", "H1", sample_df)

        # M5: first call, therefore no event.
        result_m5 = detector.detect("EURUSD", "M5", sample_df)

        assert result_h1 is not None
        assert result_m5 is None

    def test_empty_dataframe(self, detector):
        """Test with an empty dataframe."""
        result = detector.detect("EURUSD", "H1", pd.DataFrame())
        assert result is None


    def test_missing_columns(self, detector):
        """Test dataframe with missing OHLC columns."""
        dates = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")

        df = pd.DataFrame({
            "close": [1.1000, 1.1010],
            "volume": [100, 200]
        }, index=dates)

        # First call: initialize state.
        detector.detect("EURUSD", "H1", df.iloc[:1])

        # Second call: new candle detected.
        result = detector.detect("EURUSD", "H1", df)
        assert result is not None
        assert result["symbol"] == "EURUSD"
        assert result["timeframe"] == "H1"
        assert result["close"] == 1.1010

        assert result["open"] is None
        assert result["high"] is None
        assert result["low"] is None
        assert result["volume"] == 200.0
        assert result["spread"] is None

