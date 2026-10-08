# f08_risk/tests_ch4_i1_i2/risk_context_symbol_position_count_contract_tester_A.py (22)
#
# Run: pytest -v -s f08_risk/tests_ch4_i1_i2/risk_context_symbol_position_count_contract_tester_A.py

"""
Purpose:
    - Define and validate the missing per-symbol open-position-count contract
    - required before enforcing RiskLimits.max_positions_per_symbol.

Design rule:
    - This tester intentionally does NOT test RiskEngine yet.
    - The current RiskContext/SymbolRiskSnapshot contract exposes aggregate
      net-position state (current_lots/current_side), but not an explicit
      per-symbol position count.

Required future contract:
    - SymbolRiskSnapshot.position_count -> non-negative integer

Semantics:
    0 = symbol is flat / has no open positions
    1+ = actual number of open positions/tickets represented by the snapshot

This distinction is necessary because current_lots/current_side cannot
distinguish one 0.20-lot position from two 0.10-lot positions.
"""

from __future__ import annotations
from dataclasses import fields
from datetime import datetime, timezone
import pytest

from f08_risk.risk_context import SymbolRiskSnapshot


TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


def make_snapshot(position_count: int) -> SymbolRiskSnapshot:
    return SymbolRiskSnapshot(
        symbol="XAUUSD",
        exposure=0.10,
        notional=1_000.0,
        used_margin=250.0,
        current_lots=0.20,
        current_side=1,
        position_count=position_count,
    )


def test_symbol_snapshot_exposes_position_count_field() -> None:
    field_names = {field.name for field in fields(SymbolRiskSnapshot)}

    assert "position_count" in field_names


def test_zero_position_count_is_valid() -> None:
    snapshot = make_snapshot(0)

    assert snapshot.position_count == 0


def test_positive_position_count_is_preserved() -> None:
    snapshot = make_snapshot(2)

    assert snapshot.position_count == 2


def test_negative_position_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="position_count"):
        make_snapshot(-1)


def test_boolean_position_count_is_rejected() -> None:
    with pytest.raises(TypeError, match="position_count"):
        make_snapshot(True)


def test_position_count_is_integer_contract() -> None:
    snapshot = make_snapshot(3)

    assert isinstance(snapshot.position_count, int)
    assert not isinstance(snapshot.position_count, bool)
