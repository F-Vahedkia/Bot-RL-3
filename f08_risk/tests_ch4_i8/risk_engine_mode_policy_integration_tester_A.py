# f08_risk/tests_ch4_i8/risk_engine_mode_policy_integration_tester_A.py (50)
#
# Run: pytest -v -s f08_risk/tests_ch4_i8/risk_engine_mode_policy_integration_tester_A.py

"""
Bot-RL-3 v8
Chapter 4 - Item 8
RiskEngine <-> RiskModePolicy Integration Tester A
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
from f08_risk.risk_mode_policy import (
    RISK_MODES,
    get_risk_mode_policy,
)
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.risk_engine import RiskEngine


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
    experiment_id="risk-mode-integration",
)


def get_decision_mode(mode_name: str) -> DecisionMode:
    """
    Resolve the project's DecisionMode member by its actual value.
    """

    for member in DecisionMode:
        raw_value = getattr(
            member,
            "value",
            member,
        )

        if (
            str(raw_value).strip().lower()
            == mode_name
        ):
            return member

    raise AssertionError(
        f"DecisionMode does not expose official mode "
        f"{mode_name!r}."
    )


def make_context() -> RiskContext:
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
            open_position_count=0,
        ),
        symbols={
            "XAUUSD": SymbolRiskSnapshot(
                symbol="XAUUSD",
                exposure=0.10,
                notional=1_000.0,
                used_margin=250.0,
                current_lots=0.05,
                current_side=1,
            ),
        },
        correlation={
            "XAUUSD": {
                "XAUUSD": 1.0,
            },
        },
    )


def make_request(
    mode_name: str,
    *,
    target_exposure: float = 0.10,
) -> RiskRequest:

    context = make_context()

    mode = get_decision_mode(
        mode_name
    )

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=mode,
        decision_id=f"mode-{mode_name}",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": target_exposure,
        },
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


@pytest.mark.parametrize(
    "mode_name",
    sorted(RISK_MODES),
)
def test_every_official_mode_is_accepted(
    mode_name: str,
) -> None:

    engine = RiskEngine()

    request = make_request(
        mode_name
    )

    result = engine.evaluate(
        request
    )

    assert result.mode == request.decision.mode


@pytest.mark.parametrize(
    "mode_name",
    sorted(RISK_MODES),
)
def test_mode_identity_is_preserved_in_audit(
    mode_name: str,
) -> None:

    engine = RiskEngine()

    request = make_request(
        mode_name
    )

    result = engine.evaluate(
        request
    )

    records = engine.audit_trail.records()

    assert len(records) == 1
    assert records[0].mode == request.decision.mode
    assert records[0].mode == result.mode


@pytest.mark.parametrize(
    "mode_name",
    sorted(RISK_MODES),
)
def test_all_official_modes_use_full_risk_policy(
    mode_name: str,
) -> None:

    policy = get_risk_mode_policy(
        mode_name
    )

    assert policy.enforce_hard_constraints is True
    assert policy.enforce_soft_constraints is True
    assert policy.enforce_stop_loss is True
    assert policy.record_audit is True


def test_risk_mode_policy_is_exposed_in_result_metadata() -> None:
    engine = RiskEngine()

    request = make_request(
        "backtest"
    )

    result = engine.evaluate(
        request
    )

    policy_metadata = result.metadata[
        "risk_mode_policy"
    ]

    assert policy_metadata["mode"] == "backtest"
    assert policy_metadata[
        "enforce_hard_constraints"
    ] is True
    assert policy_metadata[
        "enforce_soft_constraints"
    ] is True
    assert policy_metadata[
        "enforce_stop_loss"
    ] is True
    assert policy_metadata[
        "record_audit"
    ] is True


@pytest.mark.parametrize(
    "mode_name",
    sorted(RISK_MODES),
)
def test_hard_constraint_remains_active_across_modes(
    mode_name: str,
) -> None:

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=0.20,
        )
    )

    request = make_request(
        mode_name,
        target_exposure=0.80,
    )

    result = engine.evaluate(
        request
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    assert abs(
        result.target_exposure["XAUUSD"]
    ) <= 0.20 + 1e-12

    assert any(
        violation.code
        == "MAX_SYMBOL_EXPOSURE"
        for violation
        in result.violations
    )


def test_audit_is_created_once_per_completed_evaluation() -> None:
    engine = RiskEngine()

    request = make_request(
        "backtest"
    )

    engine.evaluate(
        request
    )

    assert engine.audit_trail.count == 1
