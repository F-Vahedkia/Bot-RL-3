# f08_risk/tests_ch4_i0prev/risk_projection_tester_A.py (10)
#
# Run: pytest -v -s f08_risk/tests_ch4_i0prev/risk_projection_tester_A.py


from __future__ import annotations

from datetime import datetime, timezone

import pytest

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioDecision,
)

from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)

from f08_risk.risk_projection import (
    RiskProjection,
)


TS = datetime(
    2026,
    1,
    5,
    12,
    0,
    tzinfo=timezone.utc,
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


def test_projection_preserves_timestamp_and_decision_id():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.20,
                "EURUSD": -0.10,
            }
        ),
    )

    assert result.timestamp == TS
    assert result.decision_id == "decision-001"


def test_projection_preserves_current_state_for_missing_targets():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={}
        ),
    )

    assert result.symbols["XAUUSD"].target_exposure == pytest.approx(
        0.20
    )

    assert result.symbols["EURUSD"].target_exposure == pytest.approx(
        -0.10
    )


def test_projection_reduces_single_symbol_exposure():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.10,
                "EURUSD": -0.10,
            }
        ),
    )

    xau = result.symbols["XAUUSD"]

    assert xau.projected_exposure_delta == pytest.approx(
        -0.10
    )

    assert xau.projected_notional == pytest.approx(
        1_000.0
    )

    assert xau.projected_used_margin == pytest.approx(
        250.0
    )


def test_projection_increases_single_symbol_exposure():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.40,
                "EURUSD": -0.10,
            }
        ),
    )

    xau = result.symbols["XAUUSD"]

    assert xau.projected_exposure_delta == pytest.approx(
        0.20
    )

    assert xau.projected_notional == pytest.approx(
        4_000.0
    )

    assert xau.projected_used_margin == pytest.approx(
        1_000.0
    )


def test_projection_closes_symbol():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.0,
                "EURUSD": -0.10,
            }
        ),
    )

    xau = result.symbols["XAUUSD"]

    assert xau.target_exposure == pytest.approx(0.0)
    assert xau.projected_notional == pytest.approx(0.0)
    assert xau.projected_used_margin == pytest.approx(0.0)


def test_projection_can_introduce_new_symbol():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.20,
                "EURUSD": -0.10,
                "GBPUSD": 0.30,
            }
        ),
    )

    assert "GBPUSD" in result.symbols

    gbp = result.symbols["GBPUSD"]

    assert gbp.current_exposure == pytest.approx(0.0)
    assert gbp.target_exposure == pytest.approx(0.30)
    assert gbp.current_notional == pytest.approx(0.0)
    assert gbp.current_used_margin == pytest.approx(0.0)


def test_projection_total_exposure_is_based_on_targets():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.40,
                "EURUSD": -0.30,
            }
        ),
    )

    assert result.current_total_exposure == pytest.approx(
        0.30
    )

    assert result.projected_total_exposure == pytest.approx(
        0.70
    )


def test_projection_total_used_margin_is_recomputed():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.40,
                "EURUSD": -0.20,
            }
        ),
    )

    assert result.current_used_margin == pytest.approx(
        1_000.0
    )

    assert result.projected_used_margin == pytest.approx(
        2_000.0
    )

    assert result.projected_free_margin == pytest.approx(
        8_000.0
    )


def test_projection_position_count():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.40,
                "EURUSD": 0.0,
                "GBPUSD": 0.20,
            }
        ),
    )

    assert result.projected_position_count == 2


def test_projection_margin_utilization():
    result = RiskProjection().project(
        context(),
        decision(
            target_exposure={
                "XAUUSD": 0.40,
                "EURUSD": -0.20,
            }
        ),
    )

    assert result.projected_margin_utilization == pytest.approx(
        0.20
    )


def test_projection_is_deterministic():
    projector = RiskProjection()

    d = decision(
        target_exposure={
            "XAUUSD": 0.40,
            "EURUSD": -0.20,
            "GBPUSD": 0.10,
        }
    )

    a = projector.project(context(), d)
    b = projector.project(context(), d)

    assert a == b


def test_projection_rejects_timestamp_mismatch():
    bad_decision = PortfolioDecision(
        approved=True,
        timestamp=datetime(
            2026,
            1,
            5,
            12,
            1,
            tzinfo=timezone.utc,
        ),
        mode=DecisionMode.BACKTEST,
        decision_id="decision-002",
        capital_allocation={},
        margin_allocation={},
        target_exposure={},
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    with pytest.raises(
        ValueError,
        match="Decision timestamp must match",
    ):
        RiskProjection().project(
            context(),
            bad_decision,
        )

