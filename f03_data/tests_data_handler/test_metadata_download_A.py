"""
Tester A
--------
Unit/integration-style tests for the two DataHandler methods:

    1) DataHandler._inspect_raw_metadata()
    2) DataHandler._download_required_data()

Run:
    python -m f03_data.tests_data_handler.test_metadata_download_A
    pytest -v -s  f03_data/tests_data_handler/test_metadata_download_A.py

Design:
- No real MT5 connection is required.
- Metadata files are created in a temporary directory.
- MT5DataLoader_batch is replaced with a fake downloader when testing
  _download_required_data().
- Tests verify symbol/timeframe isolation, metadata readiness, and the
  downloader build_plan contract for number/time modes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Optional

from f03_data.data_handler_G import DataHandler, BuildParams
import f03_data.data_handler_G as data_handler_module


def _write_metadata(
    raw_dir: Path,
    symbol: str,
    timeframe: str,
    *,
    rows: int,
    first_index: str,
    last_index: str,
) -> Path:
    symbol_dir = raw_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    path = symbol_dir / f"{timeframe}.meta.json"
    path.write_text(
        json.dumps(
            {
                "rows": rows,
                "first_index": first_index,
                "last_index": last_index,
            }
        ),
        encoding="utf-8",
    )
    return path


def _make_handler(
    raw_dir: Path,
    symbol: str,
    *,
    auto_download: bool = False,
    required_bars: Optional[Dict[str, int]] = None,
) -> DataHandler:
    """Create only the attributes required by the methods under test."""
    handler = object.__new__(DataHandler)
    handler.raw_dir = raw_dir
    handler.symbol = symbol
    handler.cfg = {
        "download_defaults": {
            "auto_download": auto_download,
        }
    }
    handler.broker_timezone = "UTC"

    handler._required_bars = (
        required_bars
        if required_bars is not None
        else {"M1": 50}
    )

    return handler


class FakeBatchLoader:
    """Test double for MT5DataLoader_batch."""

    instances: list["FakeBatchLoader"] = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.build_plan_calls = []
        self.run_calls = []
        FakeBatchLoader.instances.append(self)


    def build_plan(self, symbols, timeframes, **kwargs):
        self.build_plan_calls.append(
            {
                "symbols": list(symbols),
                "timeframes": list(timeframes),
                "kwargs": dict(kwargs),
            }
        )
        plans = []
        for symbol in symbols:
            for timeframe in timeframes:
                plan = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    **dict(kwargs),
                }
                plans.append(plan)
        return plans


    def run_plan(self, plans):
        self.run_calls.append(plans)
        results = []
        for plan in plans:
            results.append(
                {
                    "symbol": plan["symbol"],
                    "timeframe": plan["timeframe"],
                    "rows_written": 0,
                    "status": "fake_success",
                }
            )
        return results


class TestMetadataAndDownload(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.tmp.name)
        FakeBatchLoader.instances.clear()


    def tearDown(self):
        self.tmp.cleanup()


    # =====================================================================
    # _inspect_raw_metadata() - number
    # =====================================================================

    def test_01_number_ready_when_rows_are_sufficient(self):
        symbol = "XAUUSD"
        _write_metadata(
            self.raw_dir, symbol, "M1",
            rows=100,
            first_index="2026-09-01T00:00:00+00:00",
            last_index="2026-09-17T00:00:00+00:00",
        )
        handler = _make_handler(self.raw_dir, symbol)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="number", start_lastrows=50,
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "ready")
        self.assertEqual(result["M1"]["rows"], 100)

    def test_02_number_insufficient_when_rows_are_too_few(self):
        symbol = "XAUUSD"
        _write_metadata(
            self.raw_dir, symbol, "M1",
            rows=40,
            first_index="2026-09-01T00:00:00+00:00",
            last_index="2026-09-17T00:00:00+00:00",
        )
        handler = _make_handler(self.raw_dir, symbol)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="number", start_lastrows=50,
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "insufficient")
        self.assertEqual(result["M1"]["rows"], 40)

    def test_03_number_missing_metadata(self):
        symbol = "XAUUSD"
        handler = _make_handler(self.raw_dir, symbol)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="number", start_lastrows=50,
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "missing")
        self.assertEqual(result["M1"]["rows"], 0)
        self.assertIsNone(result["M1"]["first_index"])
        self.assertIsNone(result["M1"]["last_index"])

    # =====================================================================
    # _inspect_raw_metadata() - time
    # =====================================================================

    def test_04_time_ready_when_metadata_covers_requested_range(self):
        symbol = "XAUUSD"
        _write_metadata(
            self.raw_dir, symbol, "M1",
            rows=500,
            first_index="2026-09-01T00:00:00+00:00",
            last_index="2026-09-17T21:00:00+00:00",
        )
        handler = _make_handler(self.raw_dir, symbol)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="time",
            start_time="2026-09-10 00:00:00+00:00",
            end_time="2026-09-17 20:00:00+00:00",
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "ready")

    def test_05_time_insufficient_when_start_is_before_metadata(self):
        symbol = "XAUUSD"
        _write_metadata(
            self.raw_dir, symbol, "M1",
            rows=500,
            first_index="2026-09-10T00:00:00+00:00",
            last_index="2026-09-17T21:00:00+00:00",
        )
        handler = _make_handler(self.raw_dir, symbol)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="time",
            start_time="2026-09-01 00:00:00+00:00",
            end_time="2026-09-17 20:00:00+00:00",
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "insufficient")

    def test_06_time_insufficient_when_end_is_after_metadata(self):
        symbol = "XAUUSD"
        _write_metadata(
            self.raw_dir, symbol, "M1",
            rows=500,
            first_index="2026-09-01T00:00:00+00:00",
            last_index="2026-09-17T20:00:00+00:00",
        )
        handler = _make_handler(self.raw_dir, symbol)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="time",
            start_time="2026-09-10 00:00:00+00:00",
            end_time="2026-09-17 21:00:00+00:00",
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "insufficient")

    # =====================================================================
    # _inspect_raw_metadata() - isolation / invalid metadata
    # =====================================================================

    def test_07_multiple_timeframes_are_inspected_independently(self):
        symbol = "XAUUSD"
        _write_metadata(
            self.raw_dir, symbol, "M1",
            rows=100,
            first_index="2026-09-01T00:00:00+00:00",
            last_index="2026-09-17T21:00:00+00:00",
        )
        _write_metadata(
            self.raw_dir, symbol, "H1",
            rows=10,
            first_index="2026-09-01T00:00:00+00:00",
            last_index="2026-09-17T20:00:00+00:00",
        )
        required_bars={
            "M1": 50,
            "H1": 50,
        }
        handler = _make_handler(self.raw_dir, symbol, required_bars=required_bars)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1", "H1"],
            mode="number", start_lastrows=50,
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "ready")
        self.assertEqual(result["H1"]["status"], "insufficient")

    def test_08_invalid_metadata_is_reported_as_invalid(self):
        symbol = "XAUUSD"
        symbol_dir = self.raw_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        (symbol_dir / "M1.meta.json").write_text("{invalid json", encoding="utf-8")

        handler = _make_handler(self.raw_dir, symbol)
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="number", start_lastrows=50,
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "invalid")

    def test_09_symbol_isolation(self):
        _write_metadata(
            self.raw_dir, "XAUUSD", "M1",
            rows=100,
            first_index="2026-09-01T00:00:00+00:00",
            last_index="2026-09-17T21:00:00+00:00",
        )
        handler = _make_handler(self.raw_dir, "BITCOIN")
        params = BuildParams(
            symbol="BITCOIN", base_tf="M1", timeframes=["M1"],
            mode="number", start_lastrows=50,
        )
        result = handler._inspect_raw_metadata(params)
        self.assertEqual(result["M1"]["status"], "missing")

    # =====================================================================
    # _download_required_data() - auto_download=False
    # =====================================================================

    def test_10_auto_download_false_returns_jobs_without_loader(self):
        symbol = "XAUUSD"
        handler = _make_handler(self.raw_dir, symbol, auto_download=False)
        metadata_info = {
            "M1": {
                "status": "missing", "rows": 0,
                "first_index": None, "last_index": None,
            }
        }
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="number", start_lastrows=100,
        )
        with patch.object(data_handler_module, "MT5DataLoader_batch") as mocked_loader:
            result = handler._download_required_data(metadata_info, params)
        mocked_loader.assert_not_called()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

        # self.assertEqual(result[0]["symbol"], symbol)
        # self.assertEqual(result[0]["timeframe"], "M1")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    # =====================================================================
    # _download_required_data() - number
    # =====================================================================

    def test_11_number_mode_build_plan_uses_requested_lookback(self):
        symbol = "XAUUSD"
        required_bars={
            "M1": 100,
            "M5": 80,
        }
        handler = _make_handler(self.raw_dir, symbol, auto_download=True, required_bars=required_bars)
        metadata_info = {
            "M1": {
                "status": "insufficient", "rows": 50,
                "first_index": "2026-09-01T00:00:00+00:00",
                "last_index": "2026-09-17T21:00:00+00:00",
                "required_rows": 100,
            },
            "M5": {
                "status": "insufficient", "rows": 40,
                "first_index": "2026-09-01T00:00:00+00:00",
                "last_index": "2026-09-17T21:00:00+00:00",
                "required_rows": 80,
            },
        }
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1", "M5"],
            mode="number", start_lastrows=100,
        )
        with patch.object(data_handler_module, "MT5DataLoader_batch", FakeBatchLoader):
            result = handler._download_required_data(metadata_info, params)

        self.assertIsNotNone(result)
        self.assertEqual(len(FakeBatchLoader.instances), 1)

        loader = FakeBatchLoader.instances[0]

        self.assertEqual(len(loader.build_plan_calls), 2)

        call_m1 = loader.build_plan_calls[0]
        self.assertEqual(call_m1["symbols"], [symbol])
        self.assertEqual(call_m1["timeframes"], ["M1"])
        self.assertEqual(call_m1["kwargs"]["lookback_bars"], 100)
        self.assertIsNone(call_m1["kwargs"]["date_from"])
        self.assertIsNone(call_m1["kwargs"]["date_to"])
        self.assertEqual(call_m1["kwargs"]["range_policy"], "count")

        call_m5 = loader.build_plan_calls[1]
        self.assertEqual(call_m5["symbols"], [symbol])
        self.assertEqual(call_m5["timeframes"], ["M5"])
        self.assertEqual(call_m5["kwargs"]["lookback_bars"], 80)
        self.assertIsNone(call_m5["kwargs"]["date_from"])
        self.assertIsNone(call_m5["kwargs"]["date_to"])
        self.assertEqual(call_m5["kwargs"]["range_policy"], "count")

        self.assertEqual(len(loader.run_calls), 1)

    def test_12_number_mode_without_explicit_start_rows_uses_required_bars(self):
        symbol = "XAUUSD"
        handler = _make_handler(self.raw_dir, symbol, auto_download=True)
        metadata_info = {
            "M1": {
                "status": "missing", "rows": 0,
                "first_index": None, "last_index": None,
                "required_rows": 0,
            }
        }
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="number", start_lastrows=None,
        )
        with patch.object(data_handler_module, "MT5DataLoader_batch", FakeBatchLoader):
            handler._download_required_data(metadata_info, params)

        self.assertEqual(len(FakeBatchLoader.instances), 1)
        loader = FakeBatchLoader.instances[0]
        call = loader.build_plan_calls[0]
        self.assertEqual(call["kwargs"]["lookback_bars"], 50)
        self.assertIsNone(call["kwargs"]["date_from"])
        self.assertIsNone(call["kwargs"]["date_to"])
        self.assertEqual(call["kwargs"]["range_policy"], "count")

    # =====================================================================
    # _download_required_data() - time
    # =====================================================================

    def test_13_time_mode_build_plan_uses_requested_date_range(self):
        symbol = "XAUUSD"
        handler = _make_handler(self.raw_dir, symbol, auto_download=True)
        metadata_info = {
            "M1": {
                "status": "insufficient", "rows": 0,
                "first_index": None, "last_index": None,
            }
        }
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1"],
            mode="time",
            start_time="2026-09-10 00:00:00+00:00",
            end_time="2026-09-17 21:00:00+00:00",
        )
        with patch.object(data_handler_module, "MT5DataLoader_batch", FakeBatchLoader):
            handler._download_required_data(metadata_info, params)

        self.assertEqual(len(FakeBatchLoader.instances), 1)
        loader = FakeBatchLoader.instances[0]
        call = loader.build_plan_calls[0]
        self.assertEqual(call["symbols"], [symbol])
        self.assertEqual(call["timeframes"], ["M1"])
        self.assertIsNone(call["kwargs"]["lookback_bars"])
        self.assertEqual(call["kwargs"]["date_from"], params.start_time)
        self.assertEqual(call["kwargs"]["date_to"], params.end_time)
        self.assertEqual(call["kwargs"]["date_tz"], "UTC")
        self.assertEqual(call["kwargs"]["result_tz"], "UTC")
        self.assertEqual(call["kwargs"]["range_policy"], "date")
        self.assertEqual(len(loader.run_calls), 1)

    # =====================================================================
    # No download when everything is ready
    # =====================================================================

    def test_14_all_ready_timeframes_require_no_download(self):
        symbol = "XAUUSD"
        required_bars={
            "M1": 100,
            "M5": 100,
        }
        handler = _make_handler(self.raw_dir, symbol, auto_download=True, required_bars=required_bars)
        metadata_info = {
            "M1": {
                "status": "ready", "rows": 1000,
                "first_index": "2026-09-01T00:00:00+00:00",
                "last_index": "2026-09-17T21:00:00+00:00",
            },
            "M5": {
                "status": "ready", "rows": 500,
                "first_index": "2026-09-01T00:00:00+00:00",
                "last_index": "2026-09-17T21:00:00+00:00",
            },
        }
        params = BuildParams(
            symbol=symbol, base_tf="M1", timeframes=["M1", "M5"],
            mode="number", start_lastrows=100,
        )
        with patch.object(data_handler_module, "MT5DataLoader_batch", FakeBatchLoader):
            result = handler._download_required_data(metadata_info, params)
        self.assertEqual(result, [])
        self.assertEqual(FakeBatchLoader.instances, [])


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMetadataAndDownload)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
