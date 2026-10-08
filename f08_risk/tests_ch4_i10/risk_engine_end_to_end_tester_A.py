# f08_risk/tests_ch4_i10/risk_engine_end_to_end_tester_A.py (53)
#
# Run: pytest -v -s f08_risk/tests_ch4_i10/risk_engine_end_to_end_tester_A.py

"""
Bot-RL-3 v8 - Chapter 4 / Item 10
RiskEngine End-to-End Tester A

Purpose
-------
Validate one integrated Chapter-4 risk flow through the PUBLIC API:

    PortfolioDecision
        -> RiskRequest
        -> RiskEngine.evaluate()
        -> projected risk evaluation
        -> risk modifications / re-projection
        -> final hard safety checks
        -> RiskDecision
        -> RiskEngineState
        -> RiskAuditTrail

This tester is intentionally different from the earlier component/boundary
and audit integration testers: it validates the complete chain in a compact,
realistic multi-symbol scenario and checks cross-layer invariants.
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
    model_name="item10-e2e-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="chapter4-item10-e2e",
)


SYMBOLS = ("XAUUSD", "EURUSD", "AUDUSD")


def make_symbol_snapshot(
    symbol: str,
    *,
    exposure: float,
    notional: float = 1_000.0,
    used_margin: float = 250.0,
    current_lots: float = 0.05,
    current_side: int | None = None,
) -> SymbolRiskSnapshot:
    """Build the current per-symbol snapshot used by the E2E scenario."""

    if current_side is None:
        if exposure > 0.0:
            current_side = 1
        elif exposure < 0.0:
            current_side = -1
        else:
            current_side = 0

    if exposure == 0.0:
        notional = 0.0
        used_margin = 0.0
        current_lots = 0.0
        current_side = 0

    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=exposure,
        notional=notional,
        used_margin=used_margin,
        current_lots=current_lots,
        current_side=current_side,
    )


def make_context(
    *,
    open_position_count: int = 1,
    equity: float = 10_000.0,
    used_margin: float = 1_000.0,
    symbols: dict[str, SymbolRiskSnapshot] | None = None,
    correlation: dict[str, dict[str, float]] | None = None,
) -> RiskContext:
    """Create a deterministic three-symbol portfolio context."""

    if symbols is None:
        symbols = {
            "XAUUSD": make_symbol_snapshot(
                "XAUUSD",
                exposure=0.10,
                notional=1_000.0,
                used_margin=250.0,
                current_lots=0.05,
                current_side=1,
            ),
            "EURUSD": make_symbol_snapshot(
                "EURUSD",
                exposure=0.0,
            ),
            "AUDUSD": make_symbol_snapshot(
                "AUDUSD",
                exposure=0.0,
            ),
        }

    if correlation is None:
        correlation = {
            "AUDUSD": {
                "AUDUSD": 1.0,
                "EURUSD": 0.80,
                "XAUUSD": 0.0,
            },
            "EURUSD": {
                "AUDUSD": 0.80,
                "EURUSD": 1.0,
                "XAUUSD": 0.80,
            },
            "XAUUSD": {
                "AUDUSD": 0.0,
                "EURUSD": 0.80,
                "XAUUSD": 1.0,
            },
        }

    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=equity,
            equity=equity,
            used_margin=used_margin,
            free_margin=equity - used_margin,
            margin_level=1000.0,
            leverage=100.0,
            peak_equity=equity,
            day_start_equity=equity,
            open_position_count=open_position_count,
        ),
        symbols=symbols,
        correlation=correlation,
    )


def make_portfolio(
    context: RiskContext,
    *,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
    risk_blocked: bool = False,
) -> PortfolioContext:
    """Create the PortfolioContext paired with the same RiskContext."""

    return PortfolioContext(
        timestamp=TS,
        equity=context.account.equity,
        balance=context.account.balance,
        used_margin=context.account.used_margin,
        free_margin=context.account.free_margin,
        margin_level=context.account.margin_level,
        drawdown=drawdown,
        daily_drawdown=daily_drawdown,
        exposure={
            symbol: snapshot.exposure
            for symbol, snapshot in context.symbols.items()
        },
        concentration={},
        correlation=context.correlation,
        risk_blocked=risk_blocked,
        mode=DecisionMode.BACKTEST,
    )


def make_request(
    *,
    decision_id: str,
    target_exposure: dict[str, float],
    capital_allocation: dict[str, float],
    margin_allocation: dict[str, float],
    portfolio_risk: float = 0.10,
    context: RiskContext | None = None,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
    risk_blocked: bool = False,
) -> RiskRequest:
    """Create the complete public RiskRequest used by Item 10."""

    if context is None:
        context = make_context()

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id=decision_id,
        capital_allocation=capital_allocation,
        margin_allocation=margin_allocation,
        target_exposure=target_exposure,
        target_signals={
            symbol: 1
            for symbol, exposure in target_exposure.items()
            if exposure > 0.0
        },
        portfolio_risk=portfolio_risk,
        reason_codes=(),
        model=MODEL,
    )

    return RiskRequest(
        decision=decision,
        portfolio=make_portfolio(
            context,
            drawdown=drawdown,
            daily_drawdown=daily_drawdown,
            risk_blocked=risk_blocked,
        ),
        risk_context=context,
    )


def make_stress_request(decision_id: str = "item10-stress") -> RiskRequest:
    """Stress several completed Chapter-4 risk mechanisms at once."""

    context = make_context(open_position_count=1)

    return make_request(
        decision_id=decision_id,
        target_exposure={
            "XAUUSD": 0.80,
            "EURUSD": 0.60,
            "AUDUSD": 0.40,
        },
        capital_allocation={
            "XAUUSD": 0.80,
            "EURUSD": 0.20,
            "AUDUSD": 0.20,
        },
        margin_allocation={
            "XAUUSD": 3_000.0,
            "EURUSD": 2_000.0,
            "AUDUSD": 1_500.0,
        },
        portfolio_risk=0.40,
        context=context,
    )


def test_e2e_clean_three_symbol_request_is_approved_and_fully_recorded() -> None:
    """
    Baseline end-to-end path:

    - multi-symbol request reaches projected evaluation;
    - no risk limit is exceeded;
    - RiskDecision is approved;
    - projected summary is attached;
    - runtime state and audit trail are both updated exactly once.
    """

    context = make_context(open_position_count=1)
    request = make_request(
        decision_id="item10-clean",
        target_exposure={
            "XAUUSD": 0.20,
            "EURUSD": 0.15,
            "AUDUSD": 0.10,
        },
        capital_allocation={
            "XAUUSD": 0.20,
            "EURUSD": 0.15,
            "AUDUSD": 0.10,
        },
        margin_allocation={
            "XAUUSD": 100.0,
            "EURUSD": 100.0,
            "AUDUSD": 100.0,
        },
        portfolio_risk=0.20,
        context=context,
    )

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=0.50,
            max_total_exposure=1.00,
            max_symbol_concentration=0.50,
            high_correlation_threshold=0.90,
            max_correlated_exposure=1.00,
            max_margin_utilization=0.80,
            max_portfolio_risk=0.50,
            max_open_positions=4,
        )
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.metadata["evaluation_path"] == "projected"
    assert tuple(result.metadata["risk_context_symbols"]) == tuple(sorted(SYMBOLS))

    projected = result.metadata["projected"]
    assert projected["projected_free_margin"] > 0.0
    assert projected["projected_total_exposure"] <= 1.00 + 1e-12
    assert projected["projected_margin_utilization"] <= 0.80 + 1e-12

    assert engine.state.evaluation_count == 1
    assert engine.state.last_decision_id == result.decision_id

    assert engine.audit_trail.count == 1
    record = engine.audit_trail.latest
    assert record is not None
    assert record.source_decision_id == request.decision.decision_id
    assert record.risk_decision_id == result.decision_id
    assert record.status == result.status
    assert record.evaluation_path == "projected"
    assert record.requested_target_exposure == request.decision.target_exposure
    assert record.final_target_exposure == result.target_exposure
    assert record.violations == result.violations


def test_e2e_stress_request_modifies_multiple_risk_dimensions_and_reprojects() -> None:
    """
    Integrated modification path.

    The exact final exposure values are deliberately not hard-coded here:
    multiple constraints may scale the same candidate sequentially. The E2E
    contract is that the public result is deterministic, bounded, and audited.
    """

    request = make_stress_request()

    limits = RiskLimits(
        max_symbol_exposure=0.40,
        max_total_exposure=0.65,
        max_symbol_concentration=0.30,
        high_correlation_threshold=0.70,
        max_correlated_exposure=0.40,
        max_margin_utilization=0.80,
        max_portfolio_risk=0.50,
        max_open_positions=5,
    )

    engine = RiskEngine(limits=limits)
    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED

    codes = {violation.code for violation in result.violations}
    assert "MAX_SYMBOL_EXPOSURE" in codes
    assert "MAX_SYMBOL_CONCENTRATION" in codes
    assert "MAX_CORRELATED_EXPOSURE" in codes
    assert "MAX_TOTAL_EXPOSURE" in codes
    assert set(result.target_exposure) == set(request.decision.target_exposure)
    assert all(
        abs(exposure) <= limits.max_symbol_exposure + 1e-12
        for exposure in result.target_exposure.values()
    )
    assert (
        sum(abs(exposure) for exposure in result.target_exposure.values())
        <= limits.max_total_exposure + 1e-12
    )
    projected = result.metadata["projected"]
    assert projected["projected_total_exposure"] <= (
        limits.max_total_exposure + 1e-12
    )
    assert projected["projected_free_margin"] > 0.0

    assert engine.state.evaluation_count == 1
    assert engine.audit_trail.count == 1
    record = engine.audit_trail.latest
    assert record is not None
    assert record.requested_target_exposure == request.decision.target_exposure
    assert record.final_target_exposure == result.target_exposure
    assert record.violations == result.violations


def test_e2e_hard_position_limit_rejects_after_the_modification_chain() -> None:
    """
    Validate the final hard-safety stage after projection/modification logic.

    One existing open position plus two new active target symbols would produce
    three projected positions, while the hard portfolio limit is two.
    """

    context = make_context(open_position_count=1)
    request = make_request(
        decision_id="item10-position-hard-guard",
        target_exposure={
            "XAUUSD": 0.10,
            "EURUSD": 0.10,
            "AUDUSD": 0.10,
        },
        capital_allocation={
            "XAUUSD": 0.10,
            "EURUSD": 0.10,
            "AUDUSD": 0.10,
        },
        margin_allocation={
            "XAUUSD": 100.0,
            "EURUSD": 100.0,
            "AUDUSD": 100.0,
        },
        portfolio_risk=0.10,
        context=context,
    )

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.00,
            max_total_exposure=1.00,
            max_symbol_concentration=1.00,
            high_correlation_threshold=0.95,
            max_correlated_exposure=1.00,
            max_margin_utilization=0.80,
            max_portfolio_risk=0.50,
            max_open_positions=2,
        )
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.REJECTED
    assert any(
        violation.code == "MAX_OPEN_POSITIONS"
        for violation in result.violations
    )
    assert result.target_exposure == {}
    assert result.capital_allocation == {}
    assert result.margin_allocation == {}

    assert engine.state.evaluation_count == 1
    assert engine.state.last_decision_id == result.decision_id
    assert engine.audit_trail.count == 1

    record = engine.audit_trail.latest
    assert record is not None
    assert record.status == RiskDecisionStatus.REJECTED
    assert record.final_target_exposure == {}
    assert record.violations == result.violations


def test_e2e_identical_request_is_deterministic() -> None:
    """Repeated evaluation of identical inputs must produce identical decisions."""

    request_a = make_stress_request("item10-deterministic-a")
    request_b = make_stress_request("item10-deterministic-b")

    limits = RiskLimits(
        max_symbol_exposure=0.40,
        max_total_exposure=0.65,
        max_symbol_concentration=0.30,
        high_correlation_threshold=0.70,
        max_correlated_exposure=0.50,
        max_margin_utilization=0.80,
        max_portfolio_risk=0.50,
        max_open_positions=5,
    )

    result_a = RiskEngine(limits=limits).evaluate(request_a)
    result_b = RiskEngine(limits=limits).evaluate(request_b)

    assert result_a.status == result_b.status
    assert result_a.target_exposure == pytest.approx(result_b.target_exposure)
    assert result_a.capital_allocation == pytest.approx(result_b.capital_allocation)
    assert result_a.margin_allocation == pytest.approx(result_b.margin_allocation)
    assert result_a.violations == result_b.violations
    assert result_a.metadata == result_b.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
