# f14_optimization/test_executor_dry_smoke.py
# Status in (Bot-RL-3): Completed
# Run: pytest f16_tests/optimization/test_executor_dry_smoke.py -q


import subprocess, sys, os

def test_executor_dry_smoke():
    cmd=[sys.executable,"-m","f12_execution.executor","--symbol","XAUUSD","-c","f01_config/config.yaml","--steps","16","--split","test"]
    res=subprocess.run(cmd,capture_output=True,text=True)
    assert res.returncode==0, res.stderr or res.stdout
