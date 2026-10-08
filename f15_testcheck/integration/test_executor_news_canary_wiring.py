# -*- coding: utf-8 -*-
# f16_tests/integration/test_executor_news_canary_wiring.py
# Status in (Bot-RL-3): Completed

# وجود و جایگذاری درست پچ‌های NewsGate و Canary را به‌صورت استاتیک تضمین می‌کنند،
# [TEST:NEWS_CANARY_WIRING]
# Run:
#    pytest f16_tests/integration/test_executor_news_canary_wiring.py -q -rA --disable-warnings
# Run All Files in f16_tests/integration:
#    pytest f16_tests/integration -q -rA --disable-warnings

import re, pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "f12_execution" / "executor.py"
text = SRC.read_text(encoding="utf-8")

def test_news_import_and_build_present():
    assert "from f06_news.integration import make_news_gate" in text
    assert "# [NEWS_GATE:BUILD]" in text
    assert "# [NEWS_GATE:CHECK]" in text

def test_canary_cli_and_cfg_present():
    assert "# [MODE:CLI]" in text
    assert "# [MODE:INIT]" in text
    assert "# [MODE:INIT][CANARY:CFG]" in text
    assert "# [CANARY:APPLY]" in text
    assert re.search(r"if\s+canary_enabled\s+and\s+exec_mode", text)

def test_news_before_spread_guard():
    i1 = text.find("# [NEWS_GATE:CHECK]")
    i2 = text.find("# --- Spread Guard")
    assert i1 != -1 and i2 != -1 and i1 < i2
