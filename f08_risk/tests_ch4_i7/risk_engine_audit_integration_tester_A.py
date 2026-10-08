# f08_risk/tests_ch4_i7/risk_engine_audit_integration_tester_A.py (46)
#
# Run: pytest -v -s f08_risk/tests_ch4_i7/risk_engine_audit_integration_tester_A.py

"""
Tester A - RiskEngine Audit Integration

Purpose:
    Verify that every completed RiskEngine.evaluate() call creates exactly
    one RiskAuditRecord in RiskAuditTrail, while RiskEngineState remains
    the runtime-statistics object.

این تستر 8 موضوع را بررسی می‌کند:
    ثبت evaluation در مسیر projected
    ثبت حالت MODIFIED
    ثبت حالت REJECTED
    حفظ ترتیب چند رکورد
    استقلال RiskAuditTrail از RiskEngineState
    بازیابی رکورد با decision_id
    یکتا بودن audit_id
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
from f08_risk.limits import (
    RiskLimits,
)
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingRequest,
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
    experiment_id="risk-audit-integration",
)


def make_context(
    *,
    equity: float = 10_000.0,
    used_margin: float = 1_000.0,
    open_position_count: int = 0,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
    risk_blocked: bool = False,
) -> RiskContext:
    if drawdown > 0.0:
        peak_equity = equity / (1.0 - drawdown) * (1.0 + 1e-12)
    else:
        peak_equity = equity

    if daily_drawdown > 0.0:
        day_start_equity = equity / (1.0 - daily_drawdown) * (1.0 + 1e-12)
    else:
        day_start_equity = equity

    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=equity,
            equity=equity,
            used_margin=used_margin,
            free_margin=equity - used_margin,
            margin_level=1000.0,
            leverage=100.0,
            peak_equity=peak_equity,
            day_start_equity=day_start_equity,
            open_position_count=open_position_count,
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
        risk_blocked=risk_blocked,
    )


def make_portfolio(
    *,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
    risk_blocked: bool = False,
) -> PortfolioContext:
    context = make_context()

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
    decision_id: str = "decision-001",
    portfolio_risk: float = 0.10,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
    risk_blocked: bool = False,
    target_exposure: dict[str, float] | None = None,
    stop_loss_requests: dict[
        str,
        StopLossPositionSizingRequest,
    ] | None = None,
) -> RiskRequest:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id=decision_id,
        capital_allocation={
            "XAUUSD": 0.20,
        },
        margin_allocation={
            "XAUUSD": 0.10,
        },
        target_exposure=target_exposure or {
            "XAUUSD": 0.10,
        },
        target_signals={
            "XAUUSD": 1,
        },
        portfolio_risk=portfolio_risk,
        reason_codes=(),
        model=MODEL,
    )

    context = make_context(
        drawdown=drawdown,
        daily_drawdown=daily_drawdown,
        risk_blocked=risk_blocked,
    )

    return RiskRequest(
        decision=decision,
        portfolio=make_portfolio(
            drawdown=drawdown,
            daily_drawdown=daily_drawdown,
            risk_blocked=risk_blocked,
        ),
        risk_context=context,
        stop_loss_requests=stop_loss_requests or {},
    )


def test_projected_evaluation_creates_one_audit_record() -> None:
    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_total_exposure=1.0,
            max_margin_utilization=1.0,
        )
    )

    request = make_request(
        decision_id="decision-projected",
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.APPROVED
    assert engine.audit_trail.count == 1

    record = engine.audit_trail.latest
    assert record is not None

    assert record.source_decision_id == "decision-projected"
    assert record.risk_decision_id == result.decision_id
    assert record.status == result.status
    assert record.evaluation_path == "projected"
    assert record.requested_target_exposure == {
        "XAUUSD": 0.10,
    }
    assert record.final_target_exposure == {
        "XAUUSD": 0.10,
    }


def test_projected_stop_loss_is_preserved_in_audit() -> None:
    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_total_exposure=1.0,
            max_margin_utilization=1.0,
        )
    )

    stop_loss_request = StopLossPositionSizingRequest(
        symbol="XAUUSD",
        equity=10_000.0,
        risk_per_trade=0.01,
        entry_price=2000.0,
        stop_price=1990.0,
        contract_size=100.0,
        currency_conversion_rate=1.0,
    )

    request = make_request(
        decision_id="decision-stop-loss-audit",
        target_exposure={
            "XAUUSD": 0.10,
        },
        stop_loss_requests={
            "XAUUSD": stop_loss_request,
        },
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.APPROVED
    assert engine.audit_trail.count == 1

    record = engine.audit_trail.latest
    assert record is not None

    assert record.stop_loss_requests == {
        "XAUUSD": stop_loss_request,
    }

    assert record.stop_loss_results == {
        "XAUUSD": result.stop_loss_results["XAUUSD"],
    }

    assert record.stop_loss_results["XAUUSD"] is result.stop_loss_results["XAUUSD"]


def test_modified_decision_audit_contains_requested_and_final_state() -> None:
    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=0.50,
            max_total_exposure=1.0,
            max_margin_utilization=1.0,
        )
    )

    request = make_request(
        decision_id="decision-modified",
        target_exposure={
            "XAUUSD": 0.80,
        },
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED
    assert engine.audit_trail.count == 1

    record = engine.audit_trail.latest
    assert record is not None

    assert record.requested_target_exposure == {
        "XAUUSD": 0.80,
    }
    assert record.final_target_exposure == {
        "XAUUSD": 0.50,
    }
    assert record.violations == result.violations


def test_rejected_decision_is_also_audited() -> None:
    engine = RiskEngine(
        limits=RiskLimits(
            max_drawdown=0.20,
        )
    )

    request = make_request(
        decision_id="decision-rejected",
        drawdown=0.20,
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.REJECTED
    assert engine.audit_trail.count == 1

    record = engine.audit_trail.latest
    assert record is not None

    assert record.source_decision_id == "decision-rejected"
    assert record.status == RiskDecisionStatus.REJECTED
    assert record.final_target_exposure == {}
    assert record.violations == result.violations


def test_rejected_stop_loss_request_is_audited_without_result() -> None:
    engine = RiskEngine(
        limits=RiskLimits(
            max_drawdown=0.20,
        )
    )

    stop_loss_request = StopLossPositionSizingRequest(
        symbol="XAUUSD",
        equity=10_000.0,
        risk_per_trade=0.01,
        entry_price=2000.0,
        stop_price=1990.0,
        contract_size=100.0,
        currency_conversion_rate=1.0,
    )

    request = make_request(
        decision_id="decision-rejected-stop-loss",
        drawdown=0.20,
        target_exposure={
            "XAUUSD": 0.10,
        },
        stop_loss_requests={
            "XAUUSD": stop_loss_request,
        },
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.REJECTED
    assert engine.audit_trail.count == 1

    record = engine.audit_trail.latest
    assert record is not None

    assert record.status == RiskDecisionStatus.REJECTED

    assert record.stop_loss_requests == {
        "XAUUSD": stop_loss_request,
    }

    assert record.stop_loss_results == {}

    assert result.stop_loss_results == {}

    assert record.final_target_exposure == {}
    

def test_multiple_evaluations_create_ordered_audit_records() -> None:
    engine = RiskEngine()

    first = engine.evaluate(
        make_request(
            decision_id="decision-001",
        )
    )
    second = engine.evaluate(
        make_request(
            decision_id="decision-002",
        )
    )

    assert engine.audit_trail.count == 2

    records = engine.audit_trail.records()

    assert records[0].risk_decision_id == first.decision_id
    assert records[1].risk_decision_id == second.decision_id


def test_audit_trail_is_separate_from_runtime_state() -> None:
    engine = RiskEngine()

    result = engine.evaluate(
        make_request(
            decision_id="decision-state-separation",
        )
    )

    assert engine.state.evaluation_count == 1
    assert engine.audit_trail.count == 1

    assert engine.state.last_decision_id == result.decision_id

    engine.state.reset()

    assert engine.state.evaluation_count == 0
    assert engine.audit_trail.count == 1
    assert engine.audit_trail.latest is not None


def test_audit_record_can_be_retrieved_by_decision_id() -> None:
    engine = RiskEngine()

    result = engine.evaluate(
        make_request(
            decision_id="decision-lookup",
        )
    )

    matches = engine.audit_trail.for_decision(
        result.decision_id
    )

    assert len(matches) == 1
    assert matches[0].risk_decision_id == result.decision_id


def test_audit_id_is_unique_across_repeated_evaluations() -> None:
    engine = RiskEngine()

    engine.evaluate(
        make_request(
            decision_id="decision-repeat",
        )
    )
    engine.evaluate(
        make_request(
            decision_id="decision-repeat",
        )
    )

    records = engine.audit_trail.records()

    assert len(records) == 2
    assert records[0].audit_id != records[1].audit_id


def test_constraint_modification_is_recorded_in_audit() -> None:
    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=0.50,
            max_total_exposure=1.0,
            max_margin_utilization=1.0,
        )
    )

    request = make_request(
        decision_id="decision-modification-lineage",
        target_exposure={
            "XAUUSD": 0.80,
        },
    )

    result = engine.evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED

    record = engine.audit_trail.latest
    assert record is not None

    assert len(record.modifications) == 1

    modification = record.modifications[0]

    assert modification.constraint_code == "MAX_SYMBOL_EXPOSURE"
    assert modification.symbol == "XAUUSD"
    assert modification.field == "target_exposure"
    assert modification.requested_value == 0.80
    assert modification.final_value == 0.50

