# -*- coding: utf-8 -*-
# f16_tests/integration/test_executor_trading_hours_behavior.py
#
# [TEST:HOURS_BEHAVIOR] — unique anchor
# Run: pytest f16_tests/integration/test_executor_trading_hours_behavior.py -q -rA

import pathlib, subprocess, sys, datetime as dt
import pytest
try:
    import yaml
except Exception:  # اگر PyYAML نصب نیست، این تست را رد کن
    yaml = None

def _closed_now_utc(sessions):
    now = dt.datetime.now(dt.timezone.utc).time()
    h, m = now.hour, now.minute
    for v in (sessions or {}).values():
        try:
            h1,m1 = [int(x) for x in str(v.get("start_utc","00:00")).split(":")[:2]]
            h2,m2 = [int(x) for x in str(v.get("end_utc","00:00")).split(":")[:2]]
            t1, t2, t0 = (h1,m1), (h2,m2), (h,m)
            open_now = (t1 <= t0 <= t2) if (t1 <= t2) else (t0 >= t1 or t0 <= t2)
            if open_now:
                return False
        except Exception:
            return False
    return True if sessions else False

@pytest.mark.skipif(yaml is None, reason="PyYAML not available")
def test_trading_hours_skip_when_closed_now(tmp_path):
    """
    cfg_path = pathlib.Path("f01_config") / "config.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    sessions = (data or {}).get("sessions") or {}
    if not sessions:
        pytest.skip("No sessions defined in config.yaml")
    if not _closed_now_utc(sessions):
        pytest.skip("Sessions are open now; can't assert SKIP_TRADING_HOURS deterministically")
    """
    
    # [TEST:HOURS_BEHAVIOR:FORCE_SESSIONS] — unique anchor
    base_cfg_path = pathlib.Path("f01_config") / "config.yaml"
    data = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
    data["sessions"] = {"block": {"start_utc": "00:00", "end_utc": "00:00"}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    # اجرا در حالت dry (نیازی به بازار زنده نیست)
    cmd = [sys.executable, "-m", "f12_execution.executor",
           "--mode", "dry", "--symbol", "XAUUSD", "--steps", "4", "-c", str(cfg_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout

    jcsv = pathlib.Path("f13_models") / "logs" / "journal_XAUUSD.csv"
    assert jcsv.exists(), "journal_XAUUSD.csv not found"
    text = jcsv.read_text(encoding="utf-8", errors="ignore")
    assert "SKIP_TRADING_HOURS" in text, "Expected SKIP_TRADING_HOURS event not found"
