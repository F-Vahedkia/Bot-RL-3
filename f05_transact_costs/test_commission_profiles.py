# f05_transact_costs/test_commission_profiles.py
#
# Run: pytest -q f05_transact_costs/test_commission_profiles.py

"""
این فایل تستر موارد زیر را بررسی میکند:
    - رفتار انتخاب پروفایل،
    - نرمال‌سازی نام نماد،
    - جلوگیری از fallback ناخواسته،
    - تبدیل نرخ round_turn به نرخ یک سمت و
    - رد حداقل کمیسیون پشتیبانی‌نشده.
"""

import pytest

from f05_transact_costs.commission_profiles import (
    CommissionProfileCatalog,
)
from f05_transact_costs.contracts import CommissionModel


def _fallback_model() -> CommissionModel:
    return CommissionModel(
        basis="per_lot",
        rate=5.0,
        currency="USD",
    )


def _profile_mapping() -> dict:
    return {
        "active_broker": "broker_a",
        "active_account_type": "raw",
        "default": {
            "basis": "per_lot",
            "rate": 6.0,
            "currency": "USD",
            "verified": True,
        },
        "brokers": {
            "broker_a": {
                "accounts": {
                    "raw": {
                        "strict_symbols": True,
                        "default": {
                            "basis": "per_lot",
                            "rate": 9.0,
                            "currency": "USD",
                            "verified": True,
                        },
                        "symbols": {
                            "EURUSD": {
                                "basis": "per_lot",
                                "rate": 8.0,
                                "currency": "USD",
                                "billing_basis": "round_turn",
                                "notional_basis": "quote",
                                "verified": True,
                            }
                        },
                    }
                }
            }
        },
    }


def _catalog(
    raw: dict | None = None,
) -> CommissionProfileCatalog:
    return CommissionProfileCatalog.from_mapping(
        _profile_mapping() if raw is None else raw,
        fallback_model=_fallback_model(),
    )


def test_resolve_normalizes_symbol_and_uses_active_account():
    catalog = _catalog()

    profile = catalog.resolve(symbol="eurusd")

    assert profile.key == "broker_a/raw/EURUSD"
    assert profile.model.basis.value == "per_lot"
    assert profile.model.rate == pytest.approx(8.0)
    assert profile.model.currency == "USD"
    assert profile.billing_basis == "round_turn"
    assert profile.verified is True


def test_round_turn_profile_is_normalized_to_one_side():
    profile = _catalog().resolve(symbol="EURUSD")

    per_side = profile.per_side_model

    assert per_side.basis.value == "per_lot"
    assert per_side.rate == pytest.approx(4.0)
    assert per_side.currency == "USD"
    assert per_side.minimum == pytest.approx(0.0)


def test_strict_symbols_rejects_unknown_symbol():
    catalog = _catalog()

    with pytest.raises(KeyError, match="strict_symbols=true"):
        catalog.resolve(symbol="XAUUSD")


def test_round_turn_profile_rejects_nonzero_minimum():
    raw = _profile_mapping()
    raw["brokers"]["broker_a"]["accounts"]["raw"][
        "symbols"
    ]["EURUSD"]["minimum"] = 1.0

    with pytest.raises(
        ValueError,
        match="round_turn profiles require minimum=0",
    ):
        _catalog(raw)


def test_duplicate_symbols_after_normalization_are_rejected():
    raw = _profile_mapping()
    symbols = raw["brokers"]["broker_a"]["accounts"]["raw"][
        "symbols"
    ]
    symbols["eurusd"] = dict(symbols["EURUSD"])

    with pytest.raises(
        ValueError,
        match="duplicate symbol profile after normalization",
    ):
        _catalog(raw)


def test_default_catalog_uses_legacy_fallback_model():
    fallback = _fallback_model()
    catalog = CommissionProfileCatalog.from_mapping(
        {},
        fallback_model=fallback,
    )

    profile = catalog.resolve(symbol="XAUUSD")

    assert profile.key == "legacy_default"
    assert profile.model == fallback
    assert profile.billing_basis == "per_side"
    assert profile.verified is False

