# f03_data/tests_live_market_engine/test_5_full_pipeline.py
# Date Reviewed:
#    1405/05/19-22:25 ==> run result is OK for 5 tests.

# Run: pytest f03_data/tests_live_market_engine/test_5_full_pipeline.py -v

# اجرای تمام فایلهای تستر داخل پوشه:
# pytest f03_data/tests_live_market_engine/ -v


import pytest
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch
sys.path.insert(0, os.path.dirname(__file__) + "/../..")
from f03_data.live_market_engine import (
    EventBus,
    CandleDetector,
    MarketDataEngine
)
from f03_data.data_handler_G import DataHandler
from zoneinfo import ZoneInfo


class TestFullPipeline:

    # =========================================================================
    @pytest.fixture
    def mock_config(self):
        return {
            "event_bus": {"queue_size": 100},
            "project": {"broker_timezone": "UTC"},
            "download_defaults": {"save_format": "parquet"},
            "__timeframes_dict": {"EURUSD": ["H1"]},
            "__base_tfs_dict": {"EURUSD": "H1"},
            "__warmups_dicts": {"EURUSD": {"H1": 20}},
            "__all_required_bars": {"EURUSD": {"H1": 20}},
        }

    # =========================================================================
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(
            "2024-01-01",
            periods=5,
            freq="1h",
            tz="UTC"
        )
        return pd.DataFrame(
            {
                "open": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                "high": [1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
                "low": [1.0990, 1.1000, 1.1010, 1.1020, 1.1030],
                "close": [1.1005, 1.1015, 1.1025, 1.1035, 1.1045],
                "volume": [100, 120, 130, 140, 150]
            }, index=dates
        )

    # =========================================================================
    # 1
    # =========================================================================
    def test_datahandler_subscribes_to_event_bus(
        self,
        mock_config
    ):
        bus = EventBus()
        handler = DataHandler(cfg=mock_config, symbol="EURUSD")
        handler.subscribe_to_event_bus(bus)

        assert handler.event_bus is bus
        assert bus.get_queue("EURUSD") is not None
        assert bus.subscriber_count() == 1


    # =========================================================================
    # 2
    # =========================================================================
    def test_event_bus_delivers_new_candle_event(
        self,
        mock_config,
        sample_df
    ):
        bus = EventBus()
        handler = DataHandler(cfg=mock_config, symbol="EURUSD")
        handler.subscribe_to_event_bus(bus)
        bus.publish(
            event_type="NEW_CANDLE",
            symbol="EURUSD",
            timeframe="H1",
            all_dfs={"EURUSD:H1": sample_df}
        )
        event = bus.get_event("EURUSD", timeout=1)

        assert event is not None
        assert event["event_type"] == "NEW_CANDLE"
        assert event["symbol"] == "EURUSD"
        assert "all_dfs" in event


    # =========================================================================
    # 3
    # =========================================================================
    def test_candle_detector_pipeline(
        self,
        sample_df
    ):
        detector = CandleDetector(ZoneInfo("UTC"))
        result1 = detector.detect("EURUSD", "H1", sample_df.iloc[:4])
        assert result1 is None

        result2 = detector.detect("EURUSD", "H1", sample_df)
        assert result2 is not None
        assert result2["symbol"] == "EURUSD"
        assert result2["close"] == 1.1045


    # =========================================================================
    # 4
    # =========================================================================
    @patch("f03_data.live_market_engine.MT5StreamWorker")
    def test_market_data_engine_connects_worker(
        self,
        MockWorker,
        mock_config
    ):
        worker = Mock()
        MockWorker.return_value = worker
        engine = MarketDataEngine(cfg=mock_config)
        engine.start(warmups_dicts={"EURUSD": {"H1": 20}})

        MockWorker.assert_called_once()
        worker.start.assert_called_once()
        
        # Mock worker.start immediately returns; the engine must then be stopped.
        assert engine._running is False

    # =========================================================================
    # 5
    # =========================================================================
    def test_multiple_symbol_handlers_are_independent(
        self,
        mock_config
    ):
        mock_config["__timeframes_dict"]["GBPUSD"] = ["H1"]
        mock_config["__base_tfs_dict"]["GBPUSD"] = "H1"
        mock_config["__warmups_dicts"]["GBPUSD"] = {"H1": 20}
        mock_config["__all_required_bars"]["GBPUSD"] = {"H1": 20}

        bus = EventBus()
        eur_handler = DataHandler(cfg=mock_config, symbol="EURUSD")
        gbp_handler = DataHandler(cfg=mock_config, symbol="GBPUSD")

        eur_handler.subscribe_to_event_bus(bus)
        gbp_handler.subscribe_to_event_bus(bus)

        assert bus.subscriber_count() == 2
        assert bus.get_queue("EURUSD") is not None
        assert bus.get_queue("GBPUSD") is not None
        assert bus.get_queue("EURUSD") is not bus.get_queue("GBPUSD")

    # =========================================================================
    # 6
    # =========================================================================
    # def test_market_data_engine_attach_data_handler(self, mock_config):
    #     engine = MarketDataEngine(cfg=mock_config)
    #     handler = DataHandler(cfg=mock_config, symbol="EURUSD")

    #     engine.attach_data_handler(handler)
    #     assert engine.data_handler is handler
    #     assert handler.event_bus is engine.event_bus


# ============================================================================= END
