# conftest.py
# هدف: ریدایرکت همهٔ خروجی‌های تست از f03_data/processed به f03_data/test_process

import os, sys
from pathlib import Path
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture(autouse=True)
def _redirect_processed_to_test_process(monkeypatch):
    """
    این فیچر به‌صورت خودکار برای همهٔ تست‌ها فعال است:
    مسیر 'processed' را به 'test_process' در کنار آن تغییر می‌دهد تا
    خروجی تست‌ها وارد مسیر تولیدی پروژه نشوند.
    """
    import f06_env.OLD_files.OLD_utils as U
    orig = U.paths_from_cfg

    def patched(cfg):
        paths = orig(cfg)
        # مسیر مقصد: f03_data/test_process
        test_proc = paths["processed"].parent / "test_process"
        try:
            test_proc.mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            print(f"[TEST_SETUP] Failed to create test_process dir: {ex}")
        paths["processed"] = test_proc
        return paths

    # پچ روی هر دو ماژول استفاده‌کننده از paths_from_cfg
    monkeypatch.setattr(U, "paths_from_cfg", patched, raising=False)
    import f03_data.OLD.data_handler as DH
    monkeypatch.setattr(DH, "paths_from_cfg", patched, raising=False)