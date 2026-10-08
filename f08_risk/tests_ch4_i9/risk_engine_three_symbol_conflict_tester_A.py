# f08_risk/tests_ch4_i9/risk_engine_three_symbol_conflict_tester_A.py (52)
#
# Run: pytest -v -s f08_risk/tests_ch4_i9/risk_engine_three_symbol_conflict_tester_A.py

"""
Bot-RL-3 v8
Chapter 4 - Item 9
Three-Symbol Multi-Conflict Tester A
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
from f08_risk.contracts import RiskDecisionStatus, RiskRequest
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
    experiment_id="three-symbol-conflict",
)


def make_snapshot(
    symbol: str,
) -> SymbolRiskSnapshot:

    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=0.0,
        notional=0.0,
        used_margin=0.0,
        current_lots=0.0,
        current_side=0,
    )


def make_context(
    symbols: tuple[str, ...],
    correlations: dict[tuple[str, str], float],
) -> RiskContext:

    correlation = {
        symbol_a: {
            symbol_b: (
                1.0
                if symbol_a == symbol_b
                else correlations.get(
                    (symbol_a, symbol_b),
                    correlations.get(
                        (symbol_b, symbol_a),
                        0.0,
                    ),
                )
            )
            for symbol_b in symbols
        }
        for symbol_a in symbols
    }

    equity = 10_000.0

    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=equity,
            equity=equity,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
            leverage=100.0,
            peak_equity=equity,
            day_start_equity=equity,
            open_position_count=0,
        ),
        symbols={
            symbol: make_snapshot(symbol)
            for symbol in symbols
        },
        correlation=correlation,
    )


def make_request(
    context: RiskContext,
    target_exposure: dict[str, float],
) -> RiskRequest:

    mode = DecisionMode.BACKTEST

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=mode,
        decision_id="three-symbol-decision",
        capital_allocation={},
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


def make_engine(
    *,
    max_total_exposure: float = 1.0,
    max_correlated_exposure: float = 0.70,
) -> RiskEngine:

    return RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_total_exposure=max_total_exposure,
            max_correlated_exposure=max_correlated_exposure,
            high_correlation_threshold=0.80,
            max_symbol_concentration=1.0,
            max_margin_utilization=1.0,
        )
    )


def test_three_symbols_are_evaluated_together() -> None:

    symbols = (
        "EURUSD",
        "GBPUSD",
        "XAUUSD",
    )

    context = make_context(
        symbols=symbols,
        correlations={},
    )

    request = make_request(
        context=context,
        target_exposure={
            "EURUSD": 0.10,
            "GBPUSD": 0.20,
            "XAUUSD": 0.15,
        },
    )

    result = make_engine().evaluate(request)

    assert result.status == RiskDecisionStatus.APPROVED

    assert result.target_exposure == pytest.approx(
        {
            "EURUSD": 0.10,
            "GBPUSD": 0.20,
            "XAUUSD": 0.15,
        }
    )


def test_two_correlated_pairs_are_handled_in_one_portfolio() -> None:

    symbols = (
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
    )

    context = make_context(
        symbols=symbols,
        correlations={
            ("EURUSD", "GBPUSD"): 0.95,
            ("GBPUSD", "AUDUSD"): 0.90,
        },
    )

    request = make_request(
        context=context,
        target_exposure={
            "EURUSD": 0.25,
            "GBPUSD": 0.50,
            "AUDUSD": 0.25,
        },
    )

    result = make_engine(
        max_total_exposure=1.0,
        max_correlated_exposure=0.70,
    ).evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED

    assert any(
        violation.code
        == "MAX_CORRELATED_EXPOSURE"
        for violation in result.violations
    )

    for symbol_a, symbol_b in (
        ("EURUSD", "GBPUSD"),
        ("GBPUSD", "AUDUSD"),
    ):
        combined = (
            abs(result.target_exposure[symbol_a])
            + abs(result.target_exposure[symbol_b])
        )

        assert combined <= 0.70 + 1e-12


def test_correlation_and_total_exposure_both_remain_valid() -> None:

    symbols = (
        "EURUSD",
        "GBPUSD",
        "XAUUSD",
    )

    context = make_context(
        symbols=symbols,
        correlations={
            ("EURUSD", "GBPUSD"): 0.95,
            ("GBPUSD", "XAUUSD"): 0.90,
        },
    )

    request = make_request(
        context=context,
        target_exposure={
            "EURUSD": 0.60,
            "GBPUSD": 0.60,
            "XAUUSD": 0.60,
        },
    )

    result = make_engine(
        max_total_exposure=0.90,
        max_correlated_exposure=0.70,
    ).evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED

    total = sum(
        abs(value)
        for value in result.target_exposure.values()
    )

    assert total <= 0.90 + 1e-12

    for symbol_a, symbol_b in (
        ("EURUSD", "GBPUSD"),
        ("GBPUSD", "XAUUSD"),
    ):
        combined = (
            abs(result.target_exposure[symbol_a])
            + abs(result.target_exposure[symbol_b])
        )

        assert combined <= 0.70 + 1e-12


def test_input_symbol_order_does_not_change_result() -> None:

    correlations = {
        ("EURUSD", "GBPUSD"): 0.95,
        ("GBPUSD", "XAUUSD"): 0.90,
    }

    context_a = make_context(
        symbols=(
            "EURUSD",
            "GBPUSD",
            "XAUUSD",
        ),
        correlations=correlations,
    )

    context_b = make_context(
        symbols=(
            "XAUUSD",
            "EURUSD",
            "GBPUSD",
        ),
        correlations=correlations,
    )

    request_a = make_request(
        context=context_a,
        target_exposure={
            "EURUSD": 0.60,
            "GBPUSD": 0.60,
            "XAUUSD": 0.60,
        },
    )

    request_b = make_request(
        context=context_b,
        target_exposure={
            "XAUUSD": 0.60,
            "GBPUSD": 0.60,
            "EURUSD": 0.60,
        },
    )

    result_a = make_engine(
        max_total_exposure=0.90,
        max_correlated_exposure=0.70,
    ).evaluate(request_a)

    result_b = make_engine(
        max_total_exposure=0.90,
        max_correlated_exposure=0.70,
    ).evaluate(request_b)

    assert result_a.target_exposure == result_b.target_exposure

    assert tuple(
        violation.code
        for violation in result_a.violations
    ) == tuple(
        violation.code
        for violation in result_b.violations
    )


def test_three_symbol_result_is_deterministic() -> None:

    symbols = (
        "EURUSD",
        "GBPUSD",
        "XAUUSD",
    )

    context = make_context(
        symbols=symbols,
        correlations={
            ("EURUSD", "GBPUSD"): 0.95,
            ("EURUSD", "XAUUSD"): 0.85,
            ("GBPUSD", "XAUUSD"): 0.90,
        },
    )

    request = make_request(
        context=context,
        target_exposure={
            "EURUSD": 0.70,
            "GBPUSD": 0.70,
            "XAUUSD": 0.70,
        },
    )

    engine_a = make_engine(
        max_total_exposure=0.90,
        max_correlated_exposure=0.70,
    )

    engine_b = make_engine(
        max_total_exposure=0.90,
        max_correlated_exposure=0.70,
    )

    result_a = engine_a.evaluate(request)
    result_b = engine_b.evaluate(request)

    assert result_a.status == result_b.status

    assert result_a.target_exposure == result_b.target_exposure

    assert result_a.capital_allocation == (
        result_b.capital_allocation
    )

    assert tuple(
        violation.code
        for violation in result_a.violations
    ) == tuple(
        violation.code
        for violation in result_b.violations
    )

