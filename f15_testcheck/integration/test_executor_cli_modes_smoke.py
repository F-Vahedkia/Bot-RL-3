# -*- coding: utf-8 -*-
# f16_tests/integration/test_executor_cli_modes_smoke.py
# Status in (Bot-RL-3): Completed

# و پذیرش پارامترهای CLI را در حالت‌های dry و semi بدون نیاز به بازار زنده چک می‌کنند.
# [TEST:EXEC_MODES_SMOKE]
# Run:
#    pytest f16_tests/integration/test_executor_cli_modes_smoke.py -q -rA --disable-warnings
# Run All Files in f16_tests/integration:
#    pytest f16_tests/integration -q -rA --disable-warnings

import subprocess, sys

def _run(args):
    cmd = [sys.executable, "-m", "f12_execution.executor"] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout

def test_dry_mode_smoke():
    _run(["--mode","dry","--symbol","XAUUSD","--steps","8","-c","f01_config/config.yaml"])

def test_semi_mode_cli_accepts_canary():
    _run(["--mode","semi","--canary-volume","0.5","--symbol","XAUUSD","--steps","8","-c","f01_config/config.yaml"])
