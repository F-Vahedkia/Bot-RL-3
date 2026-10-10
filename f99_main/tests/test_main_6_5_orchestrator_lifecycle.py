"""Focused lifecycle tests for f99_main.main_6_5.BotOrchestrator.

Run from the repository root:
    pytest -q f99_main/tests/test_main_6_5_orchestrator_lifecycle.py

These tests mock the engine/handlers; they do not connect to a real MT5 terminal.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest

# Keep imports working when pytest is invoked with this file path from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from f99_main.main_6_5 import BotOrchestrator  # noqa: E402


TEST_CONFIG = {
    "__warmups_dicts": {"EURUSD": {"M1": 10}},
    "__timeframes_dict": {"EURUSD": ["M1"]},
    "executor": {"poll_interval_sec": 0.01},
}


def _cli_args() -> Namespace:
    return Namespace(
        symbols=["EURUSD"],
        timeframes=["M1"],
        base_tf=None,
        start=None,
        end=None,
    )


def _make_live_bot(engine: Mock, handler: Mock) -> BotOrchestrator:
    """Prepare the real start()/start_live() orchestration without MT5 setup."""
    bot = BotOrchestrator(config_path=None, mode="live")
    bot.cfg = dict(TEST_CONFIG)

    # start() normally performs configuration, signal registration, and component
    # construction. These are isolated here so the tests focus on lifecycle logic.
    bot.load_configuration = Mock()
    bot._setup_signal_handlers = Mock()
    bot._init_components = Mock()
    bot._ensure_feature_pipelines = Mock()

    bot.engine = engine
    bot.data_handlers = {"EURUSD": handler}
    return bot


def _call_and_capture(callable_, result: dict) -> None:
    try:
        callable_()
    except BaseException as exc:  # capture errors raised in the calling test thread
        result["exception"] = exc


def test_engine_thread_preserves_mt5_connection_error(caplog):
    """Scenario 1, wrapper contract: don't lose the original engine exception."""
    failure = RuntimeError("MT5 connection failed")
    engine = Mock()
    engine.start.side_effect = failure
    bot = BotOrchestrator(config_path=None, mode="live")
    bot.engine = engine

    with caplog.at_level(logging.ERROR):
        bot._run_engine_thread({"EURUSD": {"M1": 10}}, 0.01)

    assert bot._engine_error is failure
    assert bot._stop_event.is_set()
    assert "MarketDataEngine thread failed" in caplog.text
    engine.start.assert_called_once_with({"EURUSD": {"M1": 10}}, 0.01)


def test_start_unblocks_and_propagates_mt5_failure_to_caller():
    """Scenario 2: start() must not remain stuck in _stop_event.wait()."""
    failure = RuntimeError("MT5 connection failed")
    engine = Mock()
    engine.start.side_effect = failure

    handler = Mock()
    handler_release = threading.Event()

    def blocking_handler_consume():
        handler_release.wait()

    handler.start_consuming.side_effect = blocking_handler_consume
    handler.stop_consuming.side_effect = handler_release.set

    bot = _make_live_bot(engine, handler)

    result: dict = {}
    caller = threading.Thread(
        target=_call_and_capture,
        args=(lambda: bot.start(_cli_args()), result),
        name="Test-OrchestratorStart",
        daemon=True,
    )
    caller.start()
    caller.join(timeout=2.0)

    if caller.is_alive():
        # Prevent a broken implementation from leaving the test process blocked.
        bot._stop_event.set()
        handler_release.set()
        caller.join(timeout=1.0)

    assert not caller.is_alive(), "BotOrchestrator.start() did not unblock after engine failure"
    assert isinstance(result.get("exception"), RuntimeError)
    assert str(result["exception"]) == "MarketDataEngine thread failed."
    assert result["exception"].__cause__ is failure
    assert bot._stop_event.is_set()
    engine.start.assert_called_once_with({"EURUSD": {"M1": 10}}, 0.01)
    engine.stop.assert_called()
    handler.stop_consuming.assert_called()


def test_start_live_retains_and_stop_joins_engine_and_handler_threads():
    """Scenario 3: start_live stores thread handles and stop joins those threads."""
    engine_entered = threading.Event()
    handler_entered = threading.Event()
    engine_release = threading.Event()
    handler_release = threading.Event()

    engine = Mock()

    def blocking_engine_start(*_args, **_kwargs):
        engine_entered.set()
        engine_release.wait()

    engine.start.side_effect = blocking_engine_start
    engine.stop.side_effect = engine_release.set

    handler = Mock()

    def blocking_handler_consume():
        handler_entered.set()
        handler_release.wait()

    handler.start_consuming.side_effect = blocking_handler_consume
    handler.stop_consuming.side_effect = handler_release.set

    bot = _make_live_bot(engine, handler)
    bot._thread_join_timeout_sec = 0.5

    bot.start_live(
        symbols=["EURUSD"],
        timeframes=["M1"],
        poll_interval=0.01,
    )

    assert engine_entered.wait(timeout=1.0)
    assert handler_entered.wait(timeout=1.0)
    assert bot._engine_thread is not None
    assert "EURUSD" in bot._handler_threads

    engine_thread = bot._engine_thread
    handler_thread = bot._handler_threads["EURUSD"]
    bot.stop()

    assert not engine_thread.is_alive()
    assert not handler_thread.is_alive()
    engine.start.assert_called_once_with({"EURUSD": {"M1": 10}}, 0.01)
    engine.stop.assert_called_once()
    handler.start_consuming.assert_called_once()
    handler.stop_consuming.assert_called_once()


def test_start_live_starts_data_handlers_before_engine(monkeypatch):
    """DataHandler threads must be started before the MarketDataEngine thread.

    این تست ترتیب فراخوانی Thread.start() را ثبت می‌کند؛
    بنابراین صرفاً به زمان‌بندی تصادفی سیستم‌عامل وابسته نیست.
    """
    release = threading.Event()
    engine_entered = threading.Event()
    handler_entered = threading.Event()

    engine = Mock()

    def blocking_engine_start(*_args, **_kwargs):
        engine_entered.set()
        release.wait()

    engine.start.side_effect = blocking_engine_start
    engine.stop.side_effect = release.set

    handler = Mock()

    def blocking_handler_consume():
        handler_entered.set()
        release.wait()

    handler.start_consuming.side_effect = blocking_handler_consume
    handler.stop_consuming.side_effect = release.set

    bot = _make_live_bot(engine, handler)

    start_order = []
    original_start = threading.Thread.start

    def tracked_start(thread, *args, **kwargs):
        if thread.name in {"DataHandler-EURUSD", "MarketDataEngine"}:
            start_order.append(thread.name)
        return original_start(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", tracked_start)

    try:
        bot.start_live(
            symbols=["EURUSD"],
            timeframes=["M1"],
            poll_interval=0.01,
        )

        assert start_order == [
            "DataHandler-EURUSD",
            "MarketDataEngine",
        ]
        assert handler_entered.wait(timeout=1.0)
        assert engine_entered.wait(timeout=1.0)

    finally:
        bot.stop()

    assert not bot._handler_threads["EURUSD"].is_alive()
    assert bot._engine_thread is not None
    assert not bot._engine_thread.is_alive()
    handler.start_consuming.assert_called_once_with()
    engine.start.assert_called_once_with(
        {"EURUSD": {"M1": 10}},
        0.01,
    )


def test_start_live_rejects_duplicate_start_without_spawning_threads():
    """A second start_live() call must not create duplicate worker threads."""
    engine = Mock()
    handler = Mock()

    bot = _make_live_bot(engine, handler)

    bot.start_live(
        symbols=["EURUSD"],
        timeframes=["M1"],
        poll_interval=0.01,
    )

    with pytest.raises(
        RuntimeError,
        match="BotOrchestrator is already running",
    ):
        bot.start_live(
            symbols=["EURUSD"],
            timeframes=["M1"],
            poll_interval=0.01,
        )

    bot.stop()

    engine.start.assert_called_once_with(
        {"EURUSD": {"M1": 10}},
        0.01,
    )
    handler.start_consuming.assert_called_once_with()
    engine.stop.assert_called_once()
    handler.stop_consuming.assert_called_once()


def test_handler_thread_unexpected_normal_exit_is_reported():
    """An unexpected normal return from DataHandler must stop the orchestrator wait.

    این تست بررسی می‌کند که اگر start_consuming() بدون exception تمام شود،
    در حالی که ربات هنوز فعال است، خطا ثبت شود و _stop_event آزاد شود.
    """
    handler = Mock()
    handler.start_consuming.return_value = None

    bot = BotOrchestrator(config_path=None, mode="live")
    bot._running = True

    # Run the wrapper directly; the mocked handler returns immediately.
    bot._run_handler_thread("EURUSD", handler)

    error = bot._handler_errors.get("EURUSD")

    assert isinstance(error, RuntimeError)
    assert str(error) == (
        "DataHandler thread exited unexpectedly for symbol=EURUSD."
    )
    assert bot._stop_event.is_set()
    handler.start_consuming.assert_called_once_with()


def test_start_propagates_unexpected_normal_engine_exit():
    """A normal engine return while running must trigger shutdown and raise.

    این تست از مسیر واقعی BotOrchestrator.start() عبور می‌کند.
    موتور بدون خطا برمی‌گردد، اما چون ارکستراتور هنوز در وضعیت اجراست،
    باید خطای مناسب به فراخواننده منتقل و shutdown اجرا شود.
    """
    handler_entered = threading.Event()
    release_handler = threading.Event()

    engine = Mock()

    def normal_engine_start(*_args, **_kwargs):
        # Ensure the handler has actually entered its consuming loop
        # before the engine exits normally.
        if not handler_entered.wait(timeout=1.5):
            raise AssertionError(
                "DataHandler did not enter start_consuming() in time."
            )

        # Normal return: no exception is raised by the engine.

    engine.start.side_effect = normal_engine_start

    handler = Mock()

    def blocking_handler_consume():
        handler_entered.set()
        release_handler.wait()

    handler.start_consuming.side_effect = blocking_handler_consume
    handler.stop_consuming.side_effect = release_handler.set

    bot = _make_live_bot(engine, handler)

    result: dict = {}
    caller = threading.Thread(
        target=_call_and_capture,
        args=(lambda: bot.start(_cli_args()), result),
        name="Test-NormalEngineExit",
        daemon=True,
    )

    caller.start()
    caller.join(timeout=3.0)

    if caller.is_alive():
        # Cleanup safeguard if a regression leaves start() blocked.
        bot._stop_event.set()
        release_handler.set()
        caller.join(timeout=1.0)

    assert not caller.is_alive(), (
        "BotOrchestrator.start() did not finish after normal engine exit."
    )

    error = result.get("exception")

    assert isinstance(error, RuntimeError)
    assert str(error) == (
        "MarketDataEngine exited unexpectedly "
        "while the orchestrator was running."
    )

    assert bot._running is False
    assert bot._stop_event.is_set()

    engine.start.assert_called_once_with(
        {"EURUSD": {"M1": 10}},
        0.01,
    )
    engine.stop.assert_called_once()
    handler.start_consuming.assert_called_once_with()
    handler.stop_consuming.assert_called_once()


def test_stop_returns_and_logs_when_thread_exceeds_join_timeout(caplog):
    """Scenario 4: a stuck worker cannot make stop() block indefinitely."""
    release_stuck_thread = threading.Event()
    stuck_thread = threading.Thread(
        target=release_stuck_thread.wait,
        name="Deliberately-Stuck-Engine",
        daemon=True,
    )

    engine = Mock()
    # Deliberately do not release the worker from engine.stop().
    engine.stop.return_value = None

    bot = BotOrchestrator(config_path=None, mode="live")
    bot.engine = engine
    bot._engine_thread = stuck_thread
    bot._thread_join_timeout_sec = 0.05
    bot._running = True

    stop_finished = threading.Event()
    timing: dict = {}

    def call_stop() -> None:
        started_at = time.monotonic()
        try:
            bot.stop()
        finally:
            timing["elapsed"] = time.monotonic() - started_at
            stop_finished.set()

    stuck_thread.start()
    try:
        with caplog.at_level(logging.ERROR):
            stopper = threading.Thread(
                target=call_stop,
                name="Test-Stop-Caller",
                daemon=True,
            )
            stopper.start()
            returned_before_release = stop_finished.wait(timeout=0.75)

            if not returned_before_release:
                # Cleanup if the implementation accidentally performs an unbounded join.
                release_stuck_thread.set()

            stopper.join(timeout=1.0)

        assert returned_before_release, "stop() waited indefinitely for a stuck thread"
        assert not stopper.is_alive()
        assert timing["elapsed"] < 0.75
        assert "Thread did not stop within" in caplog.text
        assert "Deliberately-Stuck-Engine" in caplog.text
        engine.stop.assert_called_once()
    finally:
        release_stuck_thread.set()
        stuck_thread.join(timeout=1.0)

    assert not stuck_thread.is_alive()
