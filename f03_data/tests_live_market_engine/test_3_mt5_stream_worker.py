# f03_data/tests_live_market_engine/test_3_mt5_stream_worker.py
# Date Reviewed:
#    1405/05/19-19:19 ==> run result is OK for 4 tests.

# Run:
#     pytest f03_data/tests_live_market_engine/test_3_mt5_stream_worker.py -v

# اجرای تمام فایلهای تستر داخل پوشه:
# pytest f03_data/tests_live_market_engine/ -v


import pytest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f03_data.live_market_engine import EventBus, MT5StreamWorker


class TestMT5StreamWorker:

    # ------------------------------------------- 1
    @pytest.fixture
    def mock_config(self, warmups_dicts):
        return {
            "executor": {"lookback_bars": 200},
            "project": {"broker_timezone": "UTC"},
            "__warmups_dicts": warmups_dicts,
        }

    # ------------------------------------------- 2
    @pytest.fixture
    def event_bus(self):
        return EventBus()

    # ------------------------------------------- 3
    @pytest.fixture
    def warmups_dicts(self):
        return {
            "EURUSD": {"H1": 100, "M15": 50},
            "GBPUSD": {"H1": 100}
        }

    # ------------------------------------------- 4
    @pytest.fixture
    def sample_df(self):

        dates = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=10,
            freq="1h",
            tz="UTC"
        )
        return pd.DataFrame(
            {
                "open": [1.1000] * 10,
                "high": [1.1010] * 10,
                "low": [1.0990] * 10,
                "close": [1.1005] * 10,
                "volume": [1000] * 10
            },
            index=dates
        )

    # ------------------------------------------- Test-1
    @patch("f03_data.live_market_engine.MT5Connector")
    def test_initialization(
        self,
        MockMT5Connector,
        mock_config,
        event_bus,
        warmups_dicts
    ):
        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            warmups_dicts=warmups_dicts,
            poll_interval_sec=1.0
        )
        assert worker.symbols == ["EURUSD", "GBPUSD"]
        assert worker.timeframes_dict["EURUSD"] == ["H1", "M15"]
        assert worker.poll_interval_sec == 1.0

    # ------------------------------------------- Test-2
    @patch("f03_data.live_market_engine.MT5Connector")
    def test_fetch_closed_calls_connector(
        self,
        MockMT5Connector,
        mock_config,
        event_bus,
        warmups_dicts,
        sample_df
    ):
        mock_connector = Mock()
        MockMT5Connector.return_value = mock_connector
        mock_connector.get_candles_num.return_value = sample_df
        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            warmups_dicts=warmups_dicts
        )
        df = worker._fetch_closed("EURUSD", "H1", 5)
        mock_connector.get_candles_num.assert_called_once_with(
            symbol="EURUSD",
            timeframe="H1",
            num_candles=6,
            result_tz=None,
        )
        assert not df.empty

    # ------------------------------------------- Test-3
    @patch("f03_data.live_market_engine.MT5Connector")
    def test_detection_and_publish(
        self,
        MockMT5Connector,
        mock_config,
        event_bus,
        warmups_dicts,
        sample_df
    ):
        mock_connector = Mock()
        MockMT5Connector.return_value = mock_connector
        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            warmups_dicts=warmups_dicts
        )
        event_bus.subscribe("EURUSD")
        old_df = sample_df.iloc[:-1]
        new_df = sample_df
        event1 = worker.detector.detect("EURUSD", "H1", old_df)
        assert event1 is None

        event2 = worker.detector.detect("EURUSD", "H1", new_df)
        assert event2 is not None
        assert event2["symbol"] == "EURUSD"

    # ------------------------------------------- Test-4
    @patch("f03_data.live_market_engine.MT5Connector")
    def test_connection_failure_handling(
        self,
        MockMT5Connector,
        mock_config,
        event_bus,
        warmups_dicts
    ):
        mock_connector = Mock()
        MockMT5Connector.return_value = mock_connector
        mock_connector.initialize.return_value = False
        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            warmups_dicts=warmups_dicts
        )
        with pytest.raises(
            RuntimeError,
            match="MT5 connection failed"
        ):
            worker.start()

# ============================================================================= END