# f03_data/test_instrument_specs.py
#
# Run: pytest -v -s f03_data/test_instrument_specs.py
#      pytest -q f03_data/test_instrument_specs.py

"""
Tests for the f03_data Canonical Instrument Contract.

Scope:
    snapshot -> InstrumentSpec

Design constraints:
    - No MT5 connection.
    - No broker/network access.
    - No f05_transact_costs dependency.
    - No f08_risk dependency.
    - Valid for live, train, backtest, replay, and evaluation because
      the canonical instrument contract itself is execution-mode neutral.

The tests intentionally use the real repository snapshot artifact for the
happy-path contract validation, then use in-memory copies for negative tests.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import math

import pytest
import yaml

from .instrument_specs import (
    InstrumentSpec,
    SYMBOL_CALC_MODE_FOREX,
    SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE,
    resolve_instrument_specs,
)

# =============================================================================
# ????????????
# ============================================================================= DELETED

def _snapshot_for_canonical_resolution(snapshot: dict) -> dict:
    """
    Build an in-memory normalized copy of the snapshot for contract testing.

    The repository snapshot currently contains a legacy UTC representation:
        2025-12-12T16:25:07+00:00Z

    This function does NOT modify the repository snapshot and does NOT modify
    f03_data producers. It only normalizes the representation used by the
    contract resolver in this test process.
    """
    normalized = deepcopy(snapshot)

    meta = normalized.get("meta")
    if not isinstance(meta, dict):
        raise AssertionError("snapshot.meta must be a mapping")

    as_of = meta.get("as_of")
    if not isinstance(as_of, str):
        raise AssertionError("snapshot.meta.as_of must be a string")

    as_of = as_of.strip()

    if as_of.endswith("Z"):
        without_z = as_of[:-1]

        if without_z.endswith("+00:00"):
            as_of = without_z
        else:
            as_of = without_z + "+00:00"

    parsed = datetime.fromisoformat(as_of)

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AssertionError("normalized snapshot.as_of must be timezone-aware")

    meta["as_of"] = parsed.astimezone(timezone.utc).isoformat()

    return normalized


# =============================================================================
# Paths / fixtures
# =============================================================================

SNAPSHOT_PATH = Path(__file__).with_name("specs_snapshot.yaml")


@pytest.fixture(scope="module")
def snapshot() -> dict:
    """
    Load the actual f03 snapshot artifact.

    This test intentionally validates the materialized snapshot used by the
    project rather than constructing the happy-path snapshot entirely in code.
    """
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"f03 snapshot artifact not found: {SNAPSHOT_PATH}"
        )

    with SNAPSHOT_PATH.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        pytest.fail("specs_snapshot.yaml must deserialize to a mapping")

    return value


# =============================================================================
# Basic snapshot structure
# =============================================================================

def test_snapshot_has_required_top_level_structure(snapshot: dict) -> None:
    assert isinstance(snapshot.get("meta"), dict)
    assert isinstance(snapshot.get("symbol_specs"), dict)


def test_snapshot_has_account_currency(snapshot: dict) -> None:
    account_currency = snapshot["meta"].get("account_currency")

    assert isinstance(account_currency, str)
    assert len(account_currency) == 3
    assert account_currency.isalpha()
    assert account_currency == account_currency.upper()


def test_snapshot_has_timezone_aware_as_of(snapshot: dict) -> None:
    as_of = snapshot["meta"].get("as_of")

    assert isinstance(as_of, str)
    assert as_of.strip()

    raw = as_of.strip()

    if raw.endswith("Z"):
        candidate = raw[:-1]

        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            parsed = None

        if (
            parsed is not None
            and parsed.tzinfo is not None
            and parsed.utcoffset() is not None
        ):
            pass
        else:
            parsed = datetime.fromisoformat(
                candidate + "+00:00"
            )
    else:
        parsed = datetime.fromisoformat(raw)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    assert parsed.astimezone(timezone.utc).tzinfo is not None


# =============================================================================
# Full snapshot -> canonical catalog
# =============================================================================

def test_full_snapshot_resolves_to_canonical_catalog(snapshot: dict) -> None:
    catalog = resolve_instrument_specs(snapshot)

    assert catalog
    assert all(
        isinstance(spec, InstrumentSpec)
        for spec in catalog.values()
    )

    assert set(catalog) == {
        str(symbol).strip().upper()
        for symbol in snapshot["symbol_specs"]
    }


def test_resolved_catalog_symbols_are_normalized(snapshot: dict) -> None:
    catalog = resolve_instrument_specs(snapshot)

    for symbol, spec in catalog.items():
        assert symbol == symbol.upper()
        assert spec.symbol == symbol


# =============================================================================
# Representative real instruments from the current snapshot
# =============================================================================

def test_eurusd_canonical_contract(snapshot: dict) -> None:
    spec = resolve_instrument_specs(
        snapshot,
        symbols=["EURUSD"],
    )["EURUSD"]

    assert spec.symbol == "EURUSD"

    assert spec.digits == 5
    assert math.isclose(spec.point, 0.00001)
    assert math.isclose(spec.tick_size, 0.00001)
    assert math.isclose(spec.tick_value, 1.0)

    # FX convention:
    # 5-digit EURUSD -> pip = 10 * point
    assert math.isclose(spec.pip_size, 0.0001)

    assert math.isclose(spec.contract_size, 100000.0)
    assert math.isclose(spec.volume_min, 0.01)
    assert math.isclose(spec.volume_step, 0.01)
    assert math.isclose(spec.volume_max, 100.0)

    assert spec.tick_value_currency == "USD"
    assert spec.currency_base == "EUR"
    assert spec.currency_profit == "USD"
    assert spec.currency_margin == "EUR"

    assert spec.trade_calc_mode == SYMBOL_CALC_MODE_FOREX
    assert spec.trade_stops_level == 0
    assert spec.trade_freeze_level == 0

    assert spec.account_currency == "USD"
    assert spec.source == "mt5"

    assert spec.tick_value_profit is not None
    assert spec.tick_value_loss is not None

    assert math.isclose(
        spec.value_per_price_unit_per_lot,
        100000.0,
    )

    assert math.isclose(
        spec.pip_value_per_lot,
        10.0,
    )


def test_usdjpy_preserves_directional_tick_values(snapshot: dict) -> None:
    spec = resolve_instrument_specs(
        snapshot,
        symbols=["USDJPY"],
    )["USDJPY"]

    assert spec.symbol == "USDJPY"

    assert spec.digits == 3
    assert math.isclose(spec.point, 0.001)
    assert math.isclose(spec.tick_size, 0.001)

    # 3-digit FX -> conventional pip = 10 * point
    assert math.isclose(spec.pip_size, 0.01)

    assert spec.currency_base == "USD"
    assert spec.currency_profit == "JPY"
    assert spec.currency_margin == "USD"

    assert spec.account_currency == "USD"

    assert spec.tick_value_profit is not None
    assert spec.tick_value_loss is not None

    assert math.isclose(
        spec.tick_value_profit,
        0.6421906407136022,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert math.isclose(
        spec.tick_value_loss,
        0.6422030132165379,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    # Important semantic check:
    # tick_value_currency is account/deposit currency, not currency_profit.
    assert spec.tick_value_currency == "USD"
    assert spec.tick_value_currency != spec.currency_profit

    assert math.isclose(
        spec.pip_value_per_lot_profit,
        spec.tick_value_profit
        / spec.tick_size
        * spec.pip_size,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert math.isclose(
        spec.pip_value_per_lot_loss,
        spec.tick_value_loss
        / spec.tick_size
        * spec.pip_size,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_xauusd_uses_point_as_project_cost_unit(snapshot: dict) -> None:
    spec = resolve_instrument_specs(
        snapshot,
        symbols=["XAUUSD"],
    )["XAUUSD"]

    assert spec.symbol == "XAUUSD"

    assert spec.digits == 2
    assert math.isclose(spec.point, 0.01)
    assert math.isclose(spec.tick_size, 0.01)
    assert math.isclose(spec.tick_value, 0.1)

    # Non-FX instrument:
    # project cost unit defaults to one MT5 point.
    assert math.isclose(spec.pip_size, 0.01)

    assert spec.trade_calc_mode == 4
    assert spec.is_forex is False

    assert spec.contract_size == 100.0
    assert spec.volume_min == 0.01
    assert spec.volume_step == 0.01
    assert spec.volume_max == 100.0

    assert spec.currency_base == "USD"
    assert spec.currency_profit == "USD"
    assert spec.currency_margin == "USD"
    assert spec.tick_value_currency == "USD"


# =============================================================================
# Derived values
# =============================================================================

@pytest.mark.parametrize(
    "symbol",
    [
        "XAUUSD",
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "USDCAD",
        "AUDUSD",
        "NZDUSD",
    ],
)
def test_value_per_price_unit_is_tick_value_over_tick_size(
    snapshot: dict,
    symbol: str,
) -> None:
    spec = resolve_instrument_specs(
        snapshot,
        symbols=[symbol],
    )[symbol]

    expected = spec.tick_value / spec.tick_size

    assert math.isclose(
        spec.value_per_price_unit_per_lot,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "symbol",
    [
        "XAUUSD",
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "USDCAD",
        "AUDUSD",
        "NZDUSD",
    ],
)
def test_pip_value_is_derived_from_canonical_unit(
    snapshot: dict,
    symbol: str,
) -> None:
    spec = resolve_instrument_specs(
        snapshot,
        symbols=[symbol],
    )[symbol]

    expected = (
        spec.value_per_price_unit_per_lot
        * spec.pip_size
    )

    assert math.isclose(
        spec.pip_value_per_lot,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# =============================================================================
# Provenance and normalization
# =============================================================================

def test_canonical_spec_preserves_snapshot_provenance(snapshot: dict) -> None:
    catalog = resolve_instrument_specs(
        snapshot,
        symbols=["EURUSD"],
    )

    spec = catalog["EURUSD"]

    assert spec.source == "mt5"
    assert spec.account_currency == (
        snapshot["meta"]["account_currency"].upper()
    )

    assert spec.as_of is not None
    assert spec.as_of.tzinfo is not None
    assert spec.as_of.utcoffset() is not None


def test_snapshot_raw_fields_are_promoted_into_canonical_contract(
    snapshot: dict,
) -> None:
    raw = snapshot["symbol_specs"]["EURUSD"]["raw"]

    spec = resolve_instrument_specs(
        snapshot,
        symbols=["EURUSD"],
    )["EURUSD"]

    assert spec.trade_calc_mode == raw["trade_calc_mode"]
    assert spec.trade_mode == raw["trade_mode"]
    assert spec.trade_execution_mode == raw["trade_exemode"]

    assert spec.trade_stops_level == raw["trade_stops_level"]
    assert spec.trade_freeze_level == raw["trade_freeze_level"]

    assert math.isclose(
        spec.tick_value_profit,
        raw["trade_tick_value_profit"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert math.isclose(
        spec.tick_value_loss,
        raw["trade_tick_value_loss"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert spec.currency_base == raw["currency_base"]
    assert spec.currency_profit == raw["currency_profit"]
    assert spec.currency_margin == raw["currency_margin"]


# =============================================================================
# Required-field enforcement
# =============================================================================

@pytest.mark.parametrize(
    "field",
    [
        "digits",
        "point",
        "trade_tick_value",
        "trade_tick_size",
        "contract_size",
        "volume_min",
        "volume_step",
        "volume_max",
    ],
)
def test_missing_required_mt5_field_is_rejected(
    snapshot: dict,
    field: str,
) -> None:
    broken = deepcopy(snapshot)
    broken["symbol_specs"]["EURUSD"][field] = None

    with pytest.raises(ValueError, match="missing required MT5 instrument fields"):
        resolve_instrument_specs(
            broken,
            symbols=["EURUSD"],
        )


def test_missing_account_currency_is_rejected(snapshot: dict) -> None:
    broken = deepcopy(snapshot)
    broken["meta"]["account_currency"] = None

    with pytest.raises(
        ValueError,
        match="snapshot.meta.account_currency is required",
    ):
        resolve_instrument_specs(
            broken,
            symbols=["EURUSD"],
        )


def test_missing_snapshot_as_of_is_rejected(snapshot: dict) -> None:
    broken = deepcopy(snapshot)
    broken["meta"]["as_of"] = None

    with pytest.raises(
        ValueError,
        match="snapshot.meta.as_of is required",
    ):
        resolve_instrument_specs(
            broken,
            symbols=["EURUSD"],
        )


def test_unknown_symbol_is_rejected(snapshot: dict) -> None:
    with pytest.raises(
        KeyError,
        match="instrument specifications missing from f03 snapshot",
    ):
        resolve_instrument_specs(
            snapshot,
            symbols=["THIS_SYMBOL_DOES_NOT_EXIST"],
        )


def test_duplicate_requested_symbols_are_rejected(snapshot: dict) -> None:
    with pytest.raises(
        ValueError,
        match="duplicate symbols requested",
    ):
        resolve_instrument_specs(
            snapshot,
            symbols=["EURUSD", "eurusd"],
        )


# =============================================================================
# No silent inference from missing raw directional fields
# =============================================================================

def test_directional_tick_values_are_not_invented_when_raw_fields_missing(
    snapshot: dict,
) -> None:
    broken = deepcopy(snapshot)

    raw = broken["symbol_specs"]["EURUSD"]["raw"]
    raw.pop("trade_tick_value_profit", None)
    raw.pop("trade_tick_value_loss", None)

    spec = resolve_instrument_specs(
        broken,
        symbols=["EURUSD"],
    )["EURUSD"]

    assert spec.tick_value_profit is None
    assert spec.tick_value_loss is None

    # Base tick_value remains valid because it comes from the explicit
    # trade_tick_value field.
    assert math.isclose(spec.tick_value, 1.0)


# =============================================================================
# Type / numeric validation
# =============================================================================

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("point", 0.0),
        ("trade_tick_size", 0.0),
        ("trade_tick_value", 0.0),
        ("contract_size", 0.0),
        ("volume_min", 0.0),
        ("volume_step", 0.0),
        ("volume_max", 0.0),
    ],
)
def test_invalid_non_positive_required_numeric_field_is_rejected(
    snapshot: dict,
    field: str,
    value: float,
) -> None:
    broken = deepcopy(snapshot)
    broken["symbol_specs"]["EURUSD"][field] = value

    with pytest.raises((ValueError, TypeError)):
        resolve_instrument_specs(
            broken,
            symbols=["EURUSD"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "point",
        "trade_tick_size",
        "trade_tick_value",
        "contract_size",
        "volume_min",
        "volume_step",
        "volume_max",
    ],
)
def test_nonfinite_numeric_field_is_rejected(
    snapshot: dict,
    field: str,
) -> None:
    broken = deepcopy(snapshot)
    broken["symbol_specs"]["EURUSD"][field] = float("nan")

    with pytest.raises((ValueError, TypeError)):
        resolve_instrument_specs(
            broken,
            symbols=["EURUSD"],
        )


# =============================================================================
# Broker constraint derivations
# =============================================================================

def test_stop_distance_is_derived_from_native_point_and_stop_level(
    snapshot: dict,
) -> None:
    broken = deepcopy(snapshot)

    broken["symbol_specs"]["EURUSD"]["raw"]["trade_stops_level"] = 25

    spec = resolve_instrument_specs(
        broken,
        symbols=["EURUSD"],
    )["EURUSD"]

    assert spec.trade_stops_level == 25
    assert math.isclose(
        spec.stop_distance_price,
        25 * spec.point,
    )


def test_freeze_distance_is_derived_from_native_point_and_freeze_level(
    snapshot: dict,
) -> None:
    broken = deepcopy(snapshot)

    broken["symbol_specs"]["EURUSD"]["raw"]["trade_freeze_level"] = 15

    spec = resolve_instrument_specs(
        broken,
        symbols=["EURUSD"],
    )["EURUSD"]

    assert spec.trade_freeze_level == 15
    assert math.isclose(
        spec.freeze_distance_price,
        15 * spec.point,
    )


# =============================================================================
# Instrument classification
# =============================================================================

def test_forex_classification_uses_native_calc_mode(
    snapshot: dict,
) -> None:
    eurusd = resolve_instrument_specs(
        snapshot,
        symbols=["EURUSD"],
    )["EURUSD"]

    xauusd = resolve_instrument_specs(
        snapshot,
        symbols=["XAUUSD"],
    )["XAUUSD"]

    assert eurusd.trade_calc_mode == SYMBOL_CALC_MODE_FOREX
    assert eurusd.is_forex is True

    assert xauusd.trade_calc_mode not in {
        SYMBOL_CALC_MODE_FOREX,
        SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE,
    }
    assert xauusd.is_forex is False


# =============================================================================
# Mode neutrality
# =============================================================================

@pytest.mark.parametrize(
    "execution_mode",
    [
        "live",
        "train",
        "backtest",
        "replay",
        "eval",
        "shadow",
        "paper",
        "optimize",
    ],
)
def test_canonical_instrument_contract_is_execution_mode_neutral(
    snapshot: dict,
    execution_mode: str,
) -> None:
    """
    The canonical instrument contract must be identical regardless of the
    execution/training mode consuming it.

    The mode is deliberately NOT passed to resolve_instrument_specs().
    """
    del execution_mode

    live_or_training_spec = resolve_instrument_specs(
        snapshot,
        symbols=["EURUSD"],
    )["EURUSD"]

    another_consumer_spec = resolve_instrument_specs(
        snapshot,
        symbols=["EURUSD"],
    )["EURUSD"]

    assert live_or_training_spec == another_consumer_spec


# =============================================================================
# Immutability
# =============================================================================

def test_canonical_instrument_is_immutable(snapshot: dict) -> None:
    spec = resolve_instrument_specs(
        snapshot,
        symbols=["EURUSD"],
    )["EURUSD"]

    with pytest.raises((AttributeError, TypeError)):
        spec.symbol = "GBPUSD"  # type: ignore[misc]


# =============================================================================
# Direct constructor validation
# =============================================================================

def test_direct_constructor_rejects_invalid_symbol() -> None:
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="",
            pip_size=0.0001,
            tick_size=0.00001,
            tick_value=1.0,
            tick_value_currency="USD",
        )


def test_direct_constructor_rejects_invalid_currency() -> None:
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="EURUSD",
            pip_size=0.0001,
            tick_size=0.00001,
            tick_value=1.0,
            tick_value_currency="USDX",
        )


def test_direct_constructor_rejects_invalid_tick_size() -> None:
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="EURUSD",
            pip_size=0.0001,
            tick_size=0.0,
            tick_value=1.0,
            tick_value_currency="USD",
        )


# =============================================================================
# Snapshot coverage sanity
# =============================================================================

def test_current_snapshot_contains_expected_core_instruments(
    snapshot: dict,
) -> None:
    expected = {
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "USDCAD",
        "AUDUSD",
        "NZDUSD",
        "XAUUSD",
    }

    actual = {
        str(symbol).strip().upper()
        for symbol in snapshot["symbol_specs"]
    }

    assert expected.issubset(actual)

# ============================================================================= END