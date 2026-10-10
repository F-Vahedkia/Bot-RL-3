# f03_data/tests_live_market_engine/test_4_market_data_engine.py
# Date Reviewed:
#    1405/05/19-19:38 ==> run result is OK for 7 tests.

# Run:
#     pytest f03_data/tests_live_market_engine/test_4_market_data_engine.py -v

# اجرای تمام فایلهای تستر داخل پوشه:
# pytest f03_data/tests_live_market_engine/ -v


import pytest
from unittest.mock import Mock, patch
import sys
import os
import threading

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f03_data.live_market_engine import EventBus, MarketDataEngine


class TestMarketDataEngine:

    # ------------------------------------------- 1
    @pytest.fixture
    def mock_config(self, warmups_dicts):
        return {"event_bus": {"queue_size": 500}, 
                "executor": {"lookback_bars": 200}, 
                "project": {"broker_timezone": "UTC"},
                "__warmups_dicts": warmups_dicts,
        }

    # ------------------------------------------- 2
    @pytest.fixture
    def warmups_dicts(self):
        return {"EURUSD": {"H1": 100, "M15": 50},
                "GBPUSD": {"H1": 100}
        }

    # ------------------------------------------- Test-1
    def test_initialization(self, mock_config):
        """
        Test engine initialization
        """
        engine = MarketDataEngine(cfg=mock_config)
        assert engine.event_bus is not None
        assert engine.worker is None
        assert engine._running is False


    # ------------------------------------------- Test-2
    def test_event_bus_queue_size_from_config(self, mock_config):
        """
        Test event bus uses queue_size from config
        """
        engine = MarketDataEngine(cfg=mock_config)
        assert engine.event_bus._queue_size == 500


    # ------------------------------------------- Test-3
    def test_event_bus_default_queue_size(self):
        """
        Test event bus default queue size
        """
        engine = MarketDataEngine(cfg={})
        assert engine.event_bus._queue_size == 1000


    # ------------------------------------------- Test-4
    # @patch("f03_data.live_market_engine.MT5StreamWorker")
    # def test_start_creates_worker_old1(
    #     self,
    #     MockWorker,
    #     mock_config,
    #     warmups_dicts
    # ):
    #     """
    #     Test start creates MT5StreamWorker
    #     """
    #     mock_worker_instance = Mock()
    #     MockWorker.return_value = mock_worker_instance
    #     engine = MarketDataEngine(cfg=mock_config)
    #     engine.start(
    #         warmups_dicts=warmups_dicts,
    #         poll_interval_sec=2.0
    #     )
    #     MockWorker.assert_called_once_with(
    #         cfg=mock_config,
    #         event_bus=engine.event_bus,
    #         warmups_dicts=warmups_dicts,
    #         poll_interval_sec=2.0
    #     )
    #     mock_worker_instance.start.assert_called_once()
    #     assert engine._running is True

    # -----------------------
    @patch("f03_data.live_market_engine.MT5StreamWorker")
    def test_start_creates_worker(
        self,
        MockWorker,
        mock_config,
        warmups_dicts
    ):
        started = threading.Event()
        release = threading.Event()
        errors = []

        mock_worker_instance = Mock()

        def blocking_start():
            started.set()
            release.wait(timeout=2.0)

        def stop_worker():
            release.set()

        mock_worker_instance.start.side_effect = blocking_start
        mock_worker_instance.stop.side_effect = stop_worker
        MockWorker.return_value = mock_worker_instance

        engine = MarketDataEngine(cfg=mock_config)

        def run_engine():
            try:
                engine.start(
                    warmups_dicts=warmups_dicts,
                    poll_interval_sec=2.0,
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run_engine)
        thread.start()

        assert started.wait(timeout=1.0)

        MockWorker.assert_called_once_with(
            cfg=mock_config,
            event_bus=engine.event_bus,
            warmups_dicts=warmups_dicts,
            poll_interval_sec=2.0,
        )

        # تا وقتی worker در حال اجراست، موتور باید فعال باشد.
        assert engine._running is True

        engine.stop()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert errors == []
        mock_worker_instance.stop.assert_called_once()

        # بعد از پایان worker، موتور باید غیرفعال باشد.
        assert engine._running is False

    # ------------------------------------------- Test-5
    def test_start_when_already_running(
        self,
        mock_config,
        warmups_dicts
    ):
        """
        Test start does nothing if already running
        """
        engine = MarketDataEngine(cfg=mock_config)
        engine._running = True
        engine.start(warmups_dicts=warmups_dicts)
        assert engine.worker is None


    # ------------------------------------------- Test-6
    @patch("f03_data.live_market_engine.MT5StreamWorker")
    def test_stop(
        self,
        MockWorker,
        mock_config,
        warmups_dicts
    ):
        """
        Test stop calls worker.stop
        """
        mock_worker = Mock()
        MockWorker.return_value = mock_worker
        engine = MarketDataEngine(cfg=mock_config)
        engine.start(warmups_dicts=warmups_dicts)
        engine.stop()
        mock_worker.stop.assert_called_once()
        assert engine._running is False


    # ------------------------------------------- Test-7
    def test_get_event_bus(self, mock_config):
        """
        Test get_event_bus returns the event bus
        """
        engine = MarketDataEngine(cfg=mock_config)
        bus = engine.get_event_bus()
        assert isinstance(bus, EventBus)
        assert bus is engine.event_bus

# ============================================================================= END