# f08_risk/tests_ch4_i0prev/risk_projection_margin_integration_tester_A.py (18)
#
# Run:  pytest -v -s f08_risk/tests_ch4_i0prev/risk_projection_margin_integration_tester_A.py

from __future__ import annotations

import pytest

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioDecision,
)

from f08_risk.margin_calculator import (
    InstrumentMarginSpec,
)

# from f08_risk.position_sizing import (
#     PositionSizingCalculator,
# )

from f08_risk.projection_margin import (
    ProjectionMarginCalculator,
    ProjectionMarginRequest,
)

from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)

from f08_risk.risk_projection import (
    RiskProjection,
)


TS = __import__("datetime").datetime(
    2026,
    1,
    5,
    12,
    0,
    tzinfo=__import__("datetime").timezone.utc,
)


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


def decision(
    *,
    target_exposure: dict[str, float],
) -> PortfolioDecision:
    return PortfolioDecision(
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


def margin_requests(
    target_exposure: dict[str, float],
) -> dict[str, ProjectionMarginRequest]:

    return {
        "EURUSD": ProjectionMarginRequest(
            symbol="EURUSD",
            target_exposure=target_exposure["EURUSD"],
            equity=10_000.0,
            price=1.10,
            instrument=InstrumentMarginSpec(
                symbol="EURUSD",
                contract_size=100_000.0,
                leverage=100.0,
            ),
        ),
        "XAUUSD": ProjectionMarginRequest(
            symbol="XAUUSD",
            target_exposure=target_exposure["XAUUSD"],
            equity=10_000.0,
            price=2_000.0,
            instrument=InstrumentMarginSpec(
                symbol="XAUUSD",
                contract_size=100.0,
                leverage=100.0,
            ),
        ),
    }


# =============================================================================
# Integration
# =============================================================================

def test_projection_uses_instrument_aware_margin() -> None:
    target = {
        "XAUUSD": 0.20,
        "EURUSD": -0.10,
    }

    result = RiskProjection().project(
        context(),
        decision(target_exposure=target),
        margin_requests=margin_requests(target),
    )

    # XAUUSD:
    #
    # target notional = 10,000 * 0.20 = 2,000
    # margin = 2,000 / 100 = 20
    #
    # EURUSD:
    #
    # target notional = 10,000 * 0.10 = 1,000
    # margin = 1,000 / 100 = 10

    assert result.symbols["XAUUSD"].projected_notional == pytest.approx(
        2_000.0
    )

    assert result.symbols["XAUUSD"].projected_used_margin == pytest.approx(
        20.0
    )

    assert result.symbols["EURUSD"].projected_notional == pytest.approx(
        1_000.0
    )

    assert result.symbols["EURUSD"].projected_used_margin == pytest.approx(
        10.0
    )

    assert result.projected_used_margin == pytest.approx(30.0)


def test_projection_calculates_new_symbol_margin() -> None:
    target = {
        "XAUUSD": 0.20,
        "EURUSD": -0.10,
        "GBPUSD": 0.30,
    }

    requests = margin_requests(
        {
            "XAUUSD": 0.20,
            "EURUSD": -0.10,
        }
    )

    requests["GBPUSD"] = ProjectionMarginRequest(
        symbol="GBPUSD",
        target_exposure=0.30,
        equity=10_000.0,
        price=1.25,
        instrument=InstrumentMarginSpec(
            symbol="GBPUSD",
            contract_size=100_000.0,
            leverage=50.0,
        ),
    )

    result = RiskProjection().project(
        context(),
        decision(target_exposure=target),
        margin_requests=requests,
    )

    gbp = result.symbols["GBPUSD"]

    assert gbp.projected_notional == pytest.approx(3_000.0)

    assert gbp.projected_used_margin == pytest.approx(60.0)


def test_projection_requires_margin_request_for_active_symbol() -> None:
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

    with pytest.raises(
        ValueError,
        match="Missing margin request",
    ):
        RiskProjection().project(
            context(),
            decision(target_exposure=target),
            margin_requests=requests,
        )


def test_projection_rejects_target_exposure_mismatch() -> None:
    target = {
        "XAUUSD": 0.20,
        "EURUSD": -0.10,
    }

    requests = margin_requests(target)

    requests["XAUUSD"] = ProjectionMarginRequest(
        symbol="XAUUSD",
        target_exposure=0.30,
        equity=10_000.0,
        price=2_000.0,
        instrument=InstrumentMarginSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            leverage=100.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="target_exposure does not match",
    ):
        RiskProjection().project(
            context(),
            decision(target_exposure=target),
            margin_requests=requests,
        )


def test_zero_target_does_not_require_margin_request() -> None:
    target = {
        "XAUUSD": 0.0,
        "EURUSD": -0.10,
    }

    requests = {
        "EURUSD": ProjectionMarginRequest(
            symbol="EURUSD",
            target_exposure=-0.10,
            equity=10_000.0,
            price=1.10,
            instrument=InstrumentMarginSpec(
                symbol="EURUSD",
                contract_size=100_000.0,
                leverage=100.0,
            ),
        ),
    }

    result = RiskProjection().project(
        context(),
        decision(target_exposure=target),
        margin_requests=requests,
    )

    assert result.symbols["XAUUSD"].projected_notional == pytest.approx(
        0.0
    )

    assert result.symbols["XAUUSD"].projected_used_margin == pytest.approx(
        0.0
    )


def test_projection_remains_deterministic() -> None:
    target = {
        "XAUUSD": 0.20,
        "EURUSD": -0.10,
    }

    requests = margin_requests(target)

    projector = RiskProjection()

    a = projector.project(
        context(),
        decision(target_exposure=target),
        margin_requests=requests,
    )

    b = projector.project(
        context(),
        decision(target_exposure=target),
        margin_requests=requests,
    )

    assert a == b

