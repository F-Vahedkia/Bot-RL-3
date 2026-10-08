# -*- coding: utf-8 -*-
# f16_tests/integration/test_executor_trading_hours_wiring.py
#
# [TEST:HOURS_WIRING] — unique anchor
# فقط جای‌گذاری پچ را چک می‌کند
# Run: pytest f16_tests/integration/test_executor_trading_hours_wiring.py -q -rA

import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "f12_execution" / "executor.py"
assert SRC.exists(), f"executor.py not found: {SRC}"

text = SRC.read_text(encoding="utf-8")

def test_hours_gate_exists_and_precedes_spread_guard():
    i_gate = text.find("# [HOURS:GATE]")
    i_spread = text.find("# --- Spread Guard")
    assert i_gate != -1, "HOURS:GATE anchor not found"
    assert i_spread != -1, "Spread Guard anchor not found"
    assert i_gate < i_spread, "HOURS gate must be before Spread Guard"
