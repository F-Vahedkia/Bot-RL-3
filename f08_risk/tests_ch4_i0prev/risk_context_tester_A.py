# f08_risk/tests_ch4_i0prev/risk_context_tester_A.py (8)
#
# Run: pytest -v -s f08_risk/tests_ch4_i0prev/risk_context_tester_A.py

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)


TS = datetime(
    2026,
    1,
    5,
    12,
    0,
    tzinfo=timezone.utc,
)


def account() -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        balance=10_000.0,
        equity=10_000.0,
        used_margin=1_000.0,
        free_margin=9_000.0,
        margin_level=1_000.0,
        leverage=100.0,
        peak_equity=10_500.0,
        day_start_equity=10_200.0,
        open_position_count=2,
    )


def symbol_snapshot(
    symbol: str = "XAUUSD",
) -> SymbolRiskSnapshot:
    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=0.20,
        notional=2_000.0,
        used_margin=500.0,
        current_lots=0.10,
        current_side=1,
    )


def context() -> RiskContext:
    return RiskContext(
        timestamp=TS,
        account=account(),
        symbols={
            "XAUUSD": symbol_snapshot("XAUUSD"),
            "EURUSD": symbol_snapshot("EURUSD"),
        },
        correlation={
            "XAUUSD": {
                "XAUUSD": 1.0,
                "EURUSD": 0.90,
            },
            "EURUSD": {
                "XAUUSD": 0.90,
                "EURUSD": 1.0,
            },
        },
        risk_blocked=False,
    )


def test_account_snapshot_basic_properties():
    a = account()

    assert a.margin_utilization == pytest.approx(0.10)
    assert a.drawdown == pytest.approx(
        1.0 - (10_000.0 / 10_500.0)
    )
    assert a.daily_drawdown == pytest.approx(
        1.0 - (10_000.0 / 10_200.0)
    )


def test_account_snapshot_rejects_invalid_leverage():
    with pytest.raises(ValueError, match="leverage"):
        AccountRiskSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
            leverage=0.0,
            peak_equity=10_000.0,
            day_start_equity=10_000.0,
            open_position_count=0,
        )


def test_account_snapshot_rejects_free_margin_above_equity():
    with pytest.raises(
        ValueError,
        match="free_margin cannot exceed equity",
    ):
        AccountRiskSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            used_margin=1_000.0,
            free_margin=10_001.0,
            margin_level=1_000.0,
            leverage=100.0,
            peak_equity=10_000.0,
            day_start_equity=10_000.0,
            open_position_count=0,
        )


def test_symbol_snapshot_normalizes_symbol():
    s = symbol_snapshot("xauusd")

    assert s.symbol == "XAUUSD"
    assert s.current_side == 1
    assert s.current_lots == pytest.approx(0.10)


def test_symbol_snapshot_rejects_invalid_side():
    with pytest.raises(
        ValueError,
        match="current_side must be one of",
    ):
        SymbolRiskSnapshot(
            symbol="XAUUSD",
            exposure=0.2,
            notional=2_000.0,
            used_margin=500.0,
            current_lots=0.1,
            current_side=2,
        )


def test_risk_context_normalizes_symbol_keys():
    c = RiskContext(
        timestamp=TS,
        account=account(),
        symbols={
            "xauusd": symbol_snapshot("XAUUSD"),
        },
        correlation={
            "xauusd": {
                "xauusd": 1.0,
            },
        },
    )

    assert "XAUUSD" in c.symbols
    assert c.symbol_snapshot("xauusd") is not None


def test_risk_context_total_current_exposure():
    c = context()

    assert c.total_current_exposure == pytest.approx(0.40)


def test_risk_context_total_current_margin():
    c = context()

    assert c.total_current_used_margin == pytest.approx(
        1_000.0
    )


def test_risk_context_rejects_invalid_correlation():
    with pytest.raises(
        ValueError,
        match=r"Correlation values must be in \[-1, 1\]",
    ):
        RiskContext(
            timestamp=TS,
            account=account(),
            symbols={
                "XAUUSD": symbol_snapshot("XAUUSD"),
            },
            correlation={
                "XAUUSD": {
                    "XAUUSD": 1.2,
                }
            },
        )


def test_risk_context_is_immutable():
    c = context()

    with pytest.raises(Exception):
        c.risk_blocked = True


def test_risk_context_lookup_missing_symbol():
    c = context()

    assert c.symbol_snapshot("GBPUSD") is None

