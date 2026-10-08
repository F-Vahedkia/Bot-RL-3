# f16_tests/price_action/test_registry_adapter.py
"""
Unit Tests — Price Action Registry Adapter (Extended)
=====================================================
این تست‌ها بررسی می‌کنند که:
- get_price_action_registry نگاشت کامل و قابل‌فراخوانی برگرداند.
- register_price_action_to_indicators_registry بدون overwrite، کلیدها را اضافه کند.
"""
# Run: pytest f16_tests/price_action/test_registry_adapter.py -q

import pandas as pd
from f04_features.price_action import registry_adapter as pa_adap


def test_get_price_action_registry_shape_and_callables_extended():
    reg = pa_adap.get_price_action_registry()
    assert isinstance(reg, dict) and len(reg) >= 8
    for name, fn in reg.items():
        assert isinstance(name, str) and callable(fn)


def test_register_into_existing_registry_without_overwrite_extended():
    # رجیستری فرضی موجود
    def dummy_builder(df: pd.DataFrame):
        return df

    existing = {
        "sma": dummy_builder,
        "ema": dummy_builder,
        "pa_market_structure": dummy_builder,  # تداخل عمدی
    }

    out = pa_adap.register_price_action_to_indicators_registry(existing)

    # عدم overwrite
    assert out["pa_market_structure"] is dummy_builder

    # وجود کلید جایگزین برای pa_market_structure
    assert any(k.startswith("pa__pa_market_structure") for k in out.keys()) or "pa__pa_market_structure" in out

    # همهٔ بیلدرهای PA باید اضافه شده باشند (با نام اصلی یا با پیشوند pa__)
    pa_keys = set(pa_adap.get_price_action_registry().keys())
    added = set(k for k in out.keys() if k in pa_keys or k.startswith("pa__"))
    assert len(added) >= len(pa_keys)
