# f08_risk/tests_ch4_i9/risk_engine_multi_symbol_conflict_tester_A.py (51)
#
# Run: pytest -v -s f08_risk/tests_ch4_i9/risk_engine_multi_symbol_conflict_tester_A.py

"""
Bot-RL-3 v8
Chapter 4 - Item 9
Multi-Symbol / Conflict Scenarios Tester A
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioContext,
    PortfolioDecision,
)
from f08_risk.contracts import (
    RiskDecisionStatus,
    RiskRequest,
)
from f08_risk.limits import RiskLimits
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.risk_engine import RiskEngine


TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="multi-symbol-conflict",
)


def make_snapshot(
    *,
    symbol: str,
    exposure: float = 0.0,
) -> SymbolRiskSnapshot:

    if abs(exposure) <= 1e-12:
        return SymbolRiskSnapshot(
            symbol=symbol,
            exposure=0.0,
            notional=0.0,
            used_margin=0.0,
            current_lots=0.0,
            current_side=0,
        )

    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=exposure,
        notional=1_000.0,
        used_margin=250.0,
        current_lots=0.05,
        current_side=1 if exposure > 0.0 else -1,
    )


def make_context(
    *,
    symbols: dict[str, SymbolRiskSnapshot],
    correlation_value: float = 0.0,
) -> RiskContext:

    equity = 10_000.0
    used_margin = 1_000.0

    correlation = {
        symbol_a: {
            symbol_b: (
                correlation_value
                if symbol_a != symbol_b
                else 1.0
            )
            for symbol_b in symbols
        }
        for symbol_a in symbols
    }

    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=equity,
            equity=equity,
            used_margin=used_margin,
            free_margin=equity - used_margin,
            margin_level=1_000.0,
            leverage=100.0,
            peak_equity=equity,
            day_start_equity=equity,
            open_position_count=0,
        ),
        symbols=symbols,
        correlation=correlation,
    )


def make_request(
    *,
    context: RiskContext,
    target_exposure: dict[str, float],
    capital_allocation: dict[str, float] | None = None,
) -> RiskRequest:

    mode = DecisionMode.BACKTEST

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=mode,
        decision_id="multi-symbol-decision",
        capital_allocation=(
            capital_allocation or {}
        ),
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    portfolio = PortfolioContext(
        timestamp=TS,
        equity=context.account.equity,
        balance=context.account.balance,
        used_margin=context.account.used_margin,
        free_margin=context.account.free_margin,
        margin_level=context.account.margin_level,
        drawdown=0.0,
        daily_drawdown=0.0,
        exposure={
            symbol: snapshot.exposure
            for symbol, snapshot
            in context.symbols.items()
        },
        concentration={},
        correlation=context.correlation,
        risk_blocked=False,
        mode=mode,
    )

    return RiskRequest(
        decision=decision,
        portfolio=portfolio,
        risk_context=context,
    )


def test_two_symbols_can_be_evaluated_together() -> None:

    context = make_context(
        symbols={
            "XAUUSD": make_snapshot(
                symbol="XAUUSD",
                exposure=0.10,
            ),
            "EURUSD": make_snapshot(
                symbol="EURUSD",
                exposure=0.05,
            ),
        },
        correlation_value=0.0,
    )

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_total_exposure=1.0,
            max_symbol_concentration=1.0,
            max_margin_utilization=1.0,
        )
    )

    result = engine.evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
                "EURUSD": 0.05,
            },
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED

    assert result.target_exposure["XAUUSD"] == pytest.approx(0.10)
    assert result.target_exposure["EURUSD"] == pytest.approx(0.05)


def test_total_exposure_conflict_scales_all_symbols() -> None:

    context = make_context(
        symbols={
            "XAUUSD": make_snapshot(
                symbol="XAUUSD",
            ),
            "EURUSD": make_snapshot(
                symbol="EURUSD",
            ),
        },
        correlation_value=0.0,
    )

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_total_exposure=0.80,
            max_symbol_concentration=1.0,
            max_margin_utilization=1.0,
        )
    )

    result = engine.evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.60,
                "EURUSD": 0.60,
            },
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    total = sum(
        abs(value)
        for value in result.target_exposure.values()
    )

    assert total == pytest.approx(0.80)

    assert any(
        violation.code
        == "MAX_TOTAL_EXPOSURE"
        for violation in result.violations
    )


def test_high_correlation_conflict_is_scaled() -> None:

    context = make_context(
        symbols={
            "XAUUSD": make_snapshot(
                symbol="XAUUSD",
            ),
            "EURUSD": make_snapshot(
                symbol="EURUSD",
            ),
        },
        correlation_value=0.95,
    )

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_total_exposure=1.0,
            max_correlated_exposure=0.70,
            max_symbol_concentration=1.0,
            max_margin_utilization=1.0,
        )
    )

    result = engine.evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.50,
                "EURUSD": 0.50,
            },
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    combined = (
        abs(result.target_exposure["XAUUSD"])
        + abs(result.target_exposure["EURUSD"])
    )

    assert combined == pytest.approx(0.70)

    assert any(
        violation.code
        == "MAX_CORRELATED_EXPOSURE"
        for violation in result.violations
    )


def test_symbol_concentration_conflict_is_capped() -> None:

    context = make_context(
        symbols={
            "XAUUSD": make_snapshot(
                symbol="XAUUSD",
            ),
            "EURUSD": make_snapshot(
                symbol="EURUSD",
            ),
        },
        correlation_value=0.0,
    )

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_total_exposure=1.0,
            max_symbol_concentration=0.40,
            max_margin_utilization=1.0,
        )
    )

    result = engine.evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.20,
                "EURUSD": 0.20,
            },
            capital_allocation={
                "XAUUSD": 0.80,
                "EURUSD": 0.20,
            },
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    assert (
        result.capital_allocation["XAUUSD"]
        == pytest.approx(0.40)
    )

    assert any(
        violation.code
        == "MAX_SYMBOL_CONCENTRATION"
        for violation in result.violations
    )


def test_multiple_conflicts_are_deterministic() -> None:

    context = make_context(
        symbols={
            "XAUUSD": make_snapshot(
                symbol="XAUUSD",
            ),
            "EURUSD": make_snapshot(
                symbol="EURUSD",
            ),
        },
        correlation_value=0.95,
    )

    limits = RiskLimits(
        max_symbol_exposure=0.60,
        max_total_exposure=0.80,
        max_correlated_exposure=0.50,
        max_symbol_concentration=0.40,
        max_margin_utilization=1.0,
    )

    request = make_request(
        context=context,
        target_exposure={
            "XAUUSD": 0.90,
            "EURUSD": 0.70,
        },
        capital_allocation={
            "XAUUSD": 0.80,
            "EURUSD": 0.80,
        },
    )

    first = RiskEngine(
        limits=limits
    ).evaluate(request)

    second = RiskEngine(
        limits=limits
    ).evaluate(request)

    assert first.status == RiskDecisionStatus.MODIFIED
    assert second.status == RiskDecisionStatus.MODIFIED

    assert first.target_exposure == second.target_exposure
    assert first.capital_allocation == second.capital_allocation

    assert tuple(
        violation.code
        for violation in first.violations
    ) == tuple(
        violation.code
        for violation in second.violations
    )

