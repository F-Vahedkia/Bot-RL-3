# f03_data/tests_live_market_engine/test_8_engine_lifecycle.py
#
# Run: pytest -q f03_data/tests_live_market_engine/test_8_engine_lifecycle.py

"""
این تست‌ها چهار وضعیت مشخص را بررسی می‌کنند:
    - توقف قبل از شروع worker،
    - توقف حین اتصال اولیه،
    - توقف هنگام انتظار بین دو دور polling
    - و بازنشانی پرچم موتور پس از استثنای worker
"""

import threading
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from f03_data.live_market_engine import (
    EventBus,
    MT5StreamWorker,
    MarketDataEngine,
)


def _run_and_capture(call, errors):
    try:
        call()
    except Exception as exc:
        errors.append(exc)


def _make_config():
    return {
        "project": {"broker_timezone": "UTC"},
        "event_bus": {"queue_size": 100},
    }


def _make_worker(connector, *, poll_interval_sec=60.0):
    with patch(
        "f03_data.live_market_engine.MT5Connector",
        return_value=connector,
    ):
        worker = MT5StreamWorker(
            cfg=_make_config(),
            event_bus=EventBus(),
            warmups_dicts={"EURUSD": {"M1": 10}},
            poll_interval_sec=poll_interval_sec,
        )

    return worker


def test_stop_before_worker_start_prevents_connection():
    connector = Mock()
    worker = _make_worker(connector)

    worker.stop()
    worker.start()

    connector.initialize.assert_not_called()
    connector.shutdown.assert_not_called()
    assert worker._running is False


def test_stop_during_connection_initialization_is_not_lost():
    initialize_entered = threading.Event()
    allow_initialize_to_return = threading.Event()
    loop_entered = threading.Event()
    errors = []

    connector = Mock()

    def initialize():
        initialize_entered.set()
        allow_initialize_to_return.wait(timeout=2.0)
        return True

    connector.initialize.side_effect = initialize
    worker = _make_worker(connector)
    worker._loop = lambda: loop_entered.set()

    thread = threading.Thread(
        target=_run_and_capture,
        args=(worker.start, errors),
    )
    thread.start()

    assert initialize_entered.wait(timeout=1.0)

    worker.stop()
    allow_initialize_to_return.set()

    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []
    assert not loop_entered.is_set()
    assert worker._running is False
    connector.shutdown.assert_called_once()


def test_stop_interrupts_worker_polling_wait():
    fetch_entered = threading.Event()
    errors = []

    connector = Mock()
    connector.initialize.return_value = True
    worker = _make_worker(connector)

    index = pd.date_range(
        "2024-01-01",
        periods=1,
        freq="min",
        tz="UTC",
    )

    frame = pd.DataFrame(
        {
            "open": [1.1000],
            "high": [1.1010],
            "low": [1.0990],
            "close": [1.1005],
            "volume": [100],
            "spread": [10],
        },
        index=index,
    )

    def fake_fetch_closed(symbol, timeframe, num_candles=None):
        fetch_entered.set()
        return frame

    worker._fetch_closed = fake_fetch_closed

    thread = threading.Thread(
        target=_run_and_capture,
        args=(worker.start, errors),
    )
    thread.start()

    assert fetch_entered.wait(timeout=1.0)

    worker.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []
    assert worker._running is False
    connector.shutdown.assert_called_once()


def test_engine_running_flag_resets_when_worker_raises():
    worker = Mock()
    worker.start.side_effect = RuntimeError("simulated worker failure")

    with patch(
        "f03_data.live_market_engine.MT5StreamWorker",
        return_value=worker,
    ):
        engine = MarketDataEngine(cfg=_make_config())

        with pytest.raises(
            RuntimeError,
            match="simulated worker failure",
        ):
            engine.start(
                warmups_dicts={"EURUSD": {"M1": 10}},
                poll_interval_sec=1.0,
            )

    assert engine._running is False

    # stop must remain safe after a failed worker start.
    engine.stop()
    worker.stop.assert_called_once()


