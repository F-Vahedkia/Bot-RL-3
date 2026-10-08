# f08_risk/tests_ch4_i0prev/risk_engine_instrument_margin_tester_A.py (19)
#
# Run: pytest -v -s f08_risk/tests_ch4_i0prev/risk_engine_instrument_margin_tester_A.py

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
    RiskRequest,
)
from f08_risk.limits import (
    RiskLimits,
)
from f08_risk.margin_calculator import (
    InstrumentMarginSpec,
)
from f08_risk.projection_margin import (
    ProjectionMarginRequest,
)
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.risk_engine import (
    RiskEngine,
)

TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-test",
)


def context() -> RiskContext:
    return RiskContext(
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
            "XAUUSD": SymbolRiskSnapshot(
                symbol="XAUUSD",
                exposure=0.20,
                notional=2_000.0,
                used_margin=500.0,
                current_lots=0.10,
                current_side=1,
            ),
            "EURUSD": SymbolRiskSnapshot(
                symbol="EURUSD",
                exposure=-0.10,
                notional=1_000.0,
                used_margin=500.0,
                current_lots=0.20,
                current_side=-1,
            ),
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
    )


def portfolio_request_old1(
    target_exposure: dict[str, float],
) -> RiskRequest:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    return RiskRequest(
        decision=decision,
        portfolio=decision,
        risk_context=context(),
    )


def portfolio_request(
    target_exposure: dict[str, float],
) -> RiskRequest:
    risk_context = context()

    portfolio = PortfolioContext(
        timestamp=TS,
        equity=risk_context.account.equity,
        balance=risk_context.account.balance,
        used_margin=risk_context.account.used_margin,
        free_margin=risk_context.account.free_margin,
        margin_level=risk_context.account.margin_level,
        drawdown=0.0,
        daily_drawdown=0.0,
        exposure={
            symbol: snapshot.exposure
            for symbol, snapshot in risk_context.symbols.items()
        },
        concentration={},
        correlation=risk_context.correlation,
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    return RiskRequest(
        decision=decision,
        portfolio=portfolio,
        risk_context=risk_context,
    )


def margin_requests(
    target_exposure: dict[str, float],
) -> dict[str, ProjectionMarginRequest]:

    requests = {}

    if "XAUUSD" in target_exposure:
        requests["XAUUSD"] = ProjectionMarginRequest(
            symbol="XAUUSD",
            target_exposure=target_exposure["XAUUSD"],
            equity=10_000.0,
            price=2_000.0,
            instrument=InstrumentMarginSpec(
                symbol="XAUUSD",
                contract_size=100.0,
                leverage=100.0,
            ),
        )

    if "EURUSD" in target_exposure:
        requests["EURUSD"] = ProjectionMarginRequest(
            symbol="EURUSD",
            target_exposure=target_exposure["EURUSD"],
            equity=10_000.0,
            price=1.10,
            instrument=InstrumentMarginSpec(
                symbol="EURUSD",
                contract_size=100_000.0,
                leverage=100.0,
            ),
        )

    return requests


def test_engine_uses_instrument_aware_margin() -> None:
    target = {
        "XAUUSD": 0.20,
        "EURUSD": -0.10,
    }

    engine = RiskEngine()

    result = engine.evaluate(
        portfolio_request(target),
        margin_requests=margin_requests(target),
    )

    assert result.status.value == "approved"

    projected = result.metadata["projected"]

    assert projected["projected_used_margin"] == pytest.approx(
        30.0
    )

    assert projected["projected_free_margin"] == pytest.approx(
        9_970.0
    )

    assert projected["projected_margin_utilization"] == pytest.approx(
        0.003
    )


def test_engine_realigns_margin_requests_after_symbol_cap() -> None:
    target = {
        "XAUUSD": 0.80,
        "EURUSD": -0.10,
    }

    engine = RiskEngine()

    result = engine.evaluate(
        portfolio_request(target),
        margin_requests=margin_requests(target),
    )

    assert result.status.value == "modified"

    assert result.target_exposure["XAUUSD"] == pytest.approx(
        0.4166666666666667
    )
    assert result.target_exposure["EURUSD"] == pytest.approx(
        -0.08333333333333334
    )

    projected = result.metadata["projected"]

    assert projected["projected_used_margin"] == pytest.approx(
        50.0
    )


def test_engine_realigns_margin_after_total_exposure_scaling() -> None:
    target = {
        "XAUUSD": 0.60,
        "EURUSD": -0.60,
    }

    limits = RiskLimits(
        max_total_exposure=0.50,
    )

    engine = RiskEngine(
        limits=limits,
    )

    result = engine.evaluate(
        portfolio_request(target),
        margin_requests=margin_requests(target),
    )

    assert result.status.value == "modified"

    assert (
        abs(result.target_exposure["XAUUSD"])
        + abs(result.target_exposure["EURUSD"])
    ) == pytest.approx(0.50)

    projected = result.metadata["projected"]

    assert projected["projected_used_margin"] == pytest.approx(
        50.0
    )


def test_engine_realigns_margin_during_margin_scaling() -> None:
    target = {
        "XAUUSD": 0.40,
        "EURUSD": -0.20,
    }

    limits = RiskLimits(
        max_total_exposure=1.0,
        max_margin_utilization=0.001,
    )

    engine = RiskEngine(
        limits=limits,
    )

    result = engine.evaluate(
        portfolio_request(target),
        margin_requests=margin_requests(target),
    )

    assert result.status.value == "modified"

    projected = result.metadata["projected"]

    assert projected["projected_used_margin"] == pytest.approx(
        10.0
    )

    assert projected["projected_margin_utilization"] == pytest.approx(
        0.001
    )


def test_engine_rejects_missing_margin_request() -> None:
    target = {
        "XAUUSD": 0.20,
        "EURUSD": -0.10,
    }

    requests = {
        "XAUUSD": ProjectionMarginRequest(
            symbol="XAUUSD",
            target_exposure=0.20,
            equity=10_000.0,
            price=2_000.0,
            instrument=InstrumentMarginSpec(
                symbol="XAUUSD",
                contract_size=100.0,
                leverage=100.0,
            ),
        ),
    }

    engine = RiskEngine()

    with pytest.raises(
        ValueError,
        match="Missing margin request",
    ):
        engine.evaluate(
            portfolio_request(target),
            margin_requests=requests,
        )


def test_engine_instrument_margin_path_is_deterministic() -> None:
    target = {
        "XAUUSD": 0.40,
        "EURUSD": -0.20,
    }

    requests = margin_requests(target)

    engine_a = RiskEngine()
    engine_b = RiskEngine()

    result_a = engine_a.evaluate(
        portfolio_request(target),
        margin_requests=requests,
    )

    result_b = engine_b.evaluate(
        portfolio_request(target),
        margin_requests=requests,
    )

    assert result_a == result_b

