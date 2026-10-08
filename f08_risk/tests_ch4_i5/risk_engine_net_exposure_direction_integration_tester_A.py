# f08_risk/tests_ch4_i5/risk_engine_net_exposure_direction_integration_tester_A.py (31)
#
# Run: pytest -v -s f08_risk/tests_ch4_i5/risk_engine_net_exposure_direction_integration_tester_A.py

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
    experiment_id="exp-net-direction-integration",
)


def _direction(exposure: float) -> str:
    if exposure > 0.0:
        return "LONG"
    if exposure < 0.0:
        return "SHORT"
    return "FLAT"


def _make_snapshot(
    symbol: str,
    exposure: float,
) -> SymbolRiskSnapshot:
    if exposure == 0.0:
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


def _make_context(
    *,
    symbols: dict[str, SymbolRiskSnapshot],
    used_margin: float = 1_000.0,
) -> RiskContext:
    equity = 10_000.0

    correlation = {
        symbol: {
            other: (
                1.0
                if symbol == other
                else 0.0
            )
            for other in symbols
        }
        for symbol in symbols
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
            open_position_count=sum(
                snapshot.position_count
                if snapshot.position_count is not None
                else (
                    0
                    if snapshot.exposure == 0.0
                    else 1
                )
                for snapshot in symbols.values()
            ),
        ),
        symbols=symbols,
        correlation=correlation,
    )


def _make_request(
    *,
    context: RiskContext,
    target_exposure: dict[str, float],
    portfolio_risk: float = 0.10,
) -> RiskRequest:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="net-direction-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=portfolio_risk,
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
            for symbol, snapshot in context.symbols.items()
        },
        concentration={},
        correlation=context.correlation,
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )

    return RiskRequest(
        decision=decision,
        portfolio=portfolio,
        risk_context=context,
    )


def test_positive_target_exposure_is_long() -> None:
    exposure = 0.25

    assert _direction(exposure) == "LONG"


def test_negative_target_exposure_is_short() -> None:
    exposure = -0.25

    assert _direction(exposure) == "SHORT"


def test_zero_target_exposure_is_flat() -> None:
    exposure = 0.0

    assert _direction(exposure) == "FLAT"


def test_symbol_exposure_cap_preserves_long_direction() -> None:
    context = _make_context(
        symbols={
            "XAUUSD": _make_snapshot(
                "XAUUSD",
                0.10,
            ),
        }
    )

    limits = RiskLimits(
        max_symbol_exposure=0.50,
        max_total_exposure=1.00,
    )

    result = RiskEngine(
        limits=limits
    ).evaluate(
        _make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.80,
            },
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.50)
    assert _direction(
        result.target_exposure["XAUUSD"]
    ) == "LONG"


def test_symbol_exposure_cap_preserves_short_direction() -> None:
    context = _make_context(
        symbols={
            "XAUUSD": _make_snapshot(
                "XAUUSD",
                -0.10,
            ),
        }
    )

    limits = RiskLimits(
        max_symbol_exposure=0.50,
        max_total_exposure=1.00,
    )

    result = RiskEngine(
        limits=limits
    ).evaluate(
        _make_request(
            context=context,
            target_exposure={
                "XAUUSD": -0.80,
            },
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED
    assert result.target_exposure["XAUUSD"] == pytest.approx(-0.50)
    assert _direction(
        result.target_exposure["XAUUSD"]
    ) == "SHORT"


def test_total_exposure_scaling_preserves_all_directions() -> None:
    context = _make_context(
        symbols={
            "EURUSD": _make_snapshot(
                "EURUSD",
                0.10,
            ),
            "GBPUSD": _make_snapshot(
                "GBPUSD",
                -0.10,
            ),
        }
    )

    limits = RiskLimits(
        max_symbol_exposure=1.00,
        max_total_exposure=0.50,
    )

    result = RiskEngine(
        limits=limits
    ).evaluate(
        _make_request(
            context=context,
            target_exposure={
                "EURUSD": 0.40,
                "GBPUSD": -0.40,
            },
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    assert result.target_exposure["EURUSD"] > 0.0
    assert result.target_exposure["GBPUSD"] < 0.0

    assert _direction(
        result.target_exposure["EURUSD"]
    ) == "LONG"

    assert _direction(
        result.target_exposure["GBPUSD"]
    ) == "SHORT"


def test_correlation_scaling_preserves_all_directions() -> None:
    context = RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
            leverage=100.0,
            peak_equity=10_000.0,
            day_start_equity=10_000.0,
            open_position_count=2,
        ),
        symbols={
            "EURUSD": _make_snapshot(
                "EURUSD",
                0.10,
            ),
            "GBPUSD": _make_snapshot(
                "GBPUSD",
                -0.10,
            ),
        },
        correlation={
            "EURUSD": {
                "EURUSD": 1.0,
                "GBPUSD": 0.90,
            },
            "GBPUSD": {
                "EURUSD": 0.90,
                "GBPUSD": 1.0,
            },
        },
    )

    limits = RiskLimits(
        max_symbol_exposure=1.00,
        max_total_exposure=1.00,
        high_correlation_threshold=0.85,
        max_correlated_exposure=0.50,
    )

    result = RiskEngine(
        limits=limits
    ).evaluate(
        _make_request(
            context=context,
            target_exposure={
                "EURUSD": 0.40,
                "GBPUSD": -0.40,
            },
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    assert result.target_exposure["EURUSD"] > 0.0
    assert result.target_exposure["GBPUSD"] < 0.0

    assert _direction(
        result.target_exposure["EURUSD"]
    ) == "LONG"

    assert _direction(
        result.target_exposure["GBPUSD"]
    ) == "SHORT"


def test_margin_scaling_preserves_direction() -> None:
    context = _make_context(
        symbols={
            "XAUUSD": _make_snapshot(
                "XAUUSD",
                -0.10,
            ),
        },
        used_margin=2_000.0,
    )

    limits = RiskLimits(
        max_symbol_exposure=1.00,
        max_total_exposure=1.00,
        max_margin_utilization=0.50,
    )

    result = RiskEngine(
        limits=limits
    ).evaluate(
        _make_request(
            context=context,
            target_exposure={
                "XAUUSD": -0.80,
            },
        )
    )

    assert result.status in {
        RiskDecisionStatus.APPROVED,
        RiskDecisionStatus.MODIFIED,
    }

    assert result.target_exposure["XAUUSD"] < 0.0
    assert _direction(
        result.target_exposure["XAUUSD"]
    ) == "SHORT"


def test_flat_to_long_transition_is_derived_from_signed_exposure() -> None:
    current_exposure = 0.0
    target_exposure = 0.30

    assert _direction(current_exposure) == "FLAT"
    assert _direction(target_exposure) == "LONG"


def test_flat_to_short_transition_is_derived_from_signed_exposure() -> None:
    current_exposure = 0.0
    target_exposure = -0.30

    assert _direction(current_exposure) == "FLAT"
    assert _direction(target_exposure) == "SHORT"


def test_long_to_flat_transition_is_derived_from_signed_exposure() -> None:
    current_exposure = 0.30
    target_exposure = 0.0

    assert _direction(current_exposure) == "LONG"
    assert _direction(target_exposure) == "FLAT"


def test_short_to_flat_transition_is_derived_from_signed_exposure() -> None:
    current_exposure = -0.30
    target_exposure = 0.0

    assert _direction(current_exposure) == "SHORT"
    assert _direction(target_exposure) == "FLAT"


def test_long_to_short_transition_is_derived_from_signed_exposure() -> None:
    current_exposure = 0.30
    target_exposure = -0.30

    assert _direction(current_exposure) == "LONG"
    assert _direction(target_exposure) == "SHORT"


def test_short_to_long_transition_is_derived_from_signed_exposure() -> None:
    current_exposure = -0.30
    target_exposure = 0.30

    assert _direction(current_exposure) == "SHORT"
    assert _direction(target_exposure) == "LONG"
