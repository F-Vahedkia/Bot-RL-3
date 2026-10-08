# f08_risk/tests_ch4_i7/risk_audit_modification_contract_tester_A.py (47)
#
# Run: pytest -v -s f08_risk/tests_ch4_i7/risk_audit_modification_contract_tester_A.py

"""
Tester A - RiskAuditModification Contract

Purpose:
    Verify the immutable contract of RiskAuditModification.
"""

from __future__ import annotations
import pytest

from f08_risk.audit import RiskAuditModification


def test_valid_symbol_modification() -> None:
    modification = RiskAuditModification(
        constraint_code="MAX_SYMBOL_EXPOSURE",
        symbol="xauusd",
        field="target_exposure",
        requested_value=0.80,
        final_value=0.50,
    )

    assert modification.constraint_code == "MAX_SYMBOL_EXPOSURE"
    assert modification.symbol == "XAUUSD"
    assert modification.field == "target_exposure"
    assert modification.requested_value == 0.80
    assert modification.final_value == 0.50


def test_portfolio_level_modification_allows_no_symbol() -> None:
    modification = RiskAuditModification(
        constraint_code="MAX_TOTAL_EXPOSURE",
        symbol=None,
        field="target_exposure",
        requested_value=2.50,
        final_value=2.00,
    )

    assert modification.symbol is None
    assert modification.requested_value == 2.50
    assert modification.final_value == 2.00


def test_constraint_code_is_required() -> None:
    with pytest.raises(
        ValueError,
        match="constraint_code",
    ):
        RiskAuditModification(
            constraint_code="",
            symbol="XAUUSD",
            field="target_exposure",
            requested_value=0.80,
            final_value=0.50,
        )


def test_field_is_required() -> None:
    with pytest.raises(
        ValueError,
        match="field",
    ):
        RiskAuditModification(
            constraint_code="MAX_SYMBOL_EXPOSURE",
            symbol="XAUUSD",
            field="",
            requested_value=0.80,
            final_value=0.50,
        )


def test_empty_symbol_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="symbol",
    ):
        RiskAuditModification(
            constraint_code="MAX_SYMBOL_EXPOSURE",
            symbol="   ",
            field="target_exposure",
            requested_value=0.80,
            final_value=0.50,
        )


def test_non_finite_requested_value_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="requested_value",
    ):
        RiskAuditModification(
            constraint_code="MAX_SYMBOL_EXPOSURE",
            symbol="XAUUSD",
            field="target_exposure",
            requested_value=float("inf"),
            final_value=0.50,
        )


def test_non_finite_final_value_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="final_value",
    ):
        RiskAuditModification(
            constraint_code="MAX_SYMBOL_EXPOSURE",
            symbol="XAUUSD",
            field="target_exposure",
            requested_value=0.80,
            final_value=float("nan"),
        )


def test_modification_is_immutable() -> None:
    modification = RiskAuditModification(
        constraint_code="MAX_SYMBOL_EXPOSURE",
        symbol="XAUUSD",
        field="target_exposure",
        requested_value=0.80,
        final_value=0.50,
    )

    with pytest.raises(AttributeError):
        modification.final_value = 0.40  # type: ignore[misc]