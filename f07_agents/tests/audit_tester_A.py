# f07_agents/tests/audit_tester_A.py (t9)
#
# Run: pytest -v -s f07_agents/tests/audit_tester_A.py
#
# Purpose:
#     Contract validation for audit.py
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from f07_agents.audit import (
    AgentAuditTrail,
    MetaDecisionAuditRecord,
    SymbolDecisionAuditRecord,
)
from f07_agents.contracts import (
    DecisionMode,
    MetaPolicyOutput,
    ModelIdentity,
    PolicyOutput,
    PortfolioDecision,
    SymbolAgentOutput,
)

TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1.0.0",
    policy_version="policy-1",
    config_version="cfg-1",
    experiment_id="audit-test",
)


def make_symbol_output(symbol: str = "XAUUSD") -> SymbolAgentOutput:
    return SymbolAgentOutput(
        symbol=symbol,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        signal=1,
        confidence=0.90,
        expected_return=0.02,
        risk_score=0.10,
        desired_exposure=0.20,
        stop_price=1900.0,
        model=MODEL,
        decision_id="decision-001",
    )


def make_decision(
    *,
    approved: bool = True,
    decision_id: str = "decision-001",
) -> PortfolioDecision:
    return PortfolioDecision(
        approved=approved,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        decision_id=decision_id,
        capital_allocation={"XAUUSD": 0.20, "EURUSD": 0.10},
        margin_allocation={"XAUUSD": 2_000.0, "EURUSD": 1_000.0},
        target_exposure={"XAUUSD": 0.20, "EURUSD": -0.10},
        target_signals={"XAUUSD": 1, "EURUSD": -1},
        portfolio_risk=0.25,
        reason_codes=("approved",),
        model=MODEL,
        target_stop_price={"XAUUSD": 1900.0, "EURUSD": 1.10},
    )


def test_symbol_audit_record_validates_identity_and_timezone():
    record = SymbolDecisionAuditRecord(
        audit_id="audit-001",
        decision_id="decision-001",
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        symbol="XAUUSD",
        signal=1,
        confidence=0.90,
        expected_return=0.02,
        risk_score=0.10,
        desired_exposure=0.20,
        stop_price=1900.0,
        model=MODEL,
        metadata={"source": "test"},
    )

    assert record.audit_id == "audit-001"
    assert record.decision_id == "decision-001"
    assert record.symbol == "XAUUSD"
    assert record.stop_price == pytest.approx(1900.0)
    assert record.model == MODEL
    assert record.metadata == {"source": "test"}

    with pytest.raises(ValueError, match="timezone-aware"):
        SymbolDecisionAuditRecord(
            audit_id="audit-002",
            decision_id="decision-002",
            timestamp=datetime(2026, 1, 1, 12, 0),
            mode=DecisionMode.BACKTEST,
            symbol="XAUUSD",
            signal=1,
            confidence=0.90,
            expected_return=0.02,
            risk_score=0.10,
            desired_exposure=0.20,
            stop_price=1900.0,
            model=MODEL,
        )


def test_meta_audit_record_copies_mutable_mappings():
    capital = {"XAUUSD": 0.20}
    target_stop = {"XAUUSD": 1900.0}
    record = MetaDecisionAuditRecord(
        audit_id="audit-001",
        decision_id="decision-001",
        timestamp=TIMESTAMP,
        mode=DecisionMode.LIVE,
        approved=True,
        capital_allocation=capital,
        margin_allocation={"XAUUSD": 2_000.0},
        target_exposure={"XAUUSD": 0.20},
        target_signals={"XAUUSD": 1},
        portfolio_risk=0.25,
        reason_codes=("approved",),
        model=MODEL,
        target_stop_price=target_stop,
        metadata={"source": "test"},
    )

    capital["XAUUSD"] = 0.90
    target_stop["XAUUSD"] = 9999.0

    assert record.capital_allocation == {"XAUUSD": 0.20}
    assert record.target_stop_price == {"XAUUSD": 1900.0}
    assert record.reason_codes == ("approved",)


def test_audit_trail_appends_symbol_output_and_exposes_latest():
    trail = AgentAuditTrail(max_records=10)
    metadata = {"stage": "symbol"}
    output = make_symbol_output()

    record = trail.append_symbol_output(
        output,
        audit_id="audit-symbol-001",
        metadata=metadata,
    )

    assert isinstance(record, SymbolDecisionAuditRecord)
    assert trail.symbol_count == 1
    assert trail.meta_count == 0
    assert trail.count == 1
    assert trail.latest_symbol == record
    assert trail.symbol_records() == (record,)
    assert record.metadata == metadata


def test_audit_trail_appends_final_portfolio_decision():
    trail = AgentAuditTrail(max_records=10)
    decision = make_decision()

    record = trail.append_portfolio_decision(
        decision,
        audit_id="audit-meta-001",
        metadata={"stage": "final-decision"},
    )

    assert isinstance(record, MetaDecisionAuditRecord)
    assert trail.meta_count == 1
    assert trail.latest_meta == record
    assert record.approved is True
    assert record.decision_id == decision.decision_id
    assert record.target_exposure == dict(decision.target_exposure)
    assert record.target_stop_price == dict(decision.target_stop_price)
    assert record.model == MODEL


def test_audit_trail_preserves_order_and_clear():
    trail = AgentAuditTrail(max_records=10)
    symbol_record = trail.append_symbol_output(
        make_symbol_output(),
        audit_id="symbol-001",
    )
    meta_record = trail.append_portfolio_decision(
        make_decision(decision_id="meta-001"),
        audit_id="meta-001",
    )

    assert trail.symbol_records() == (symbol_record,)
    assert trail.meta_records() == (meta_record,)
    assert trail.count == 2

    trail.clear()

    assert trail.count == 0
    assert trail.symbol_count == 0
    assert trail.meta_count == 0
    assert trail.latest_symbol is None
    assert trail.latest_meta is None


def test_audit_trail_is_bounded():
    trail = AgentAuditTrail(max_records=2)

    trail.append_symbol_output(make_symbol_output("XAUUSD"), audit_id="s-1")
    trail.append_symbol_output(make_symbol_output("EURUSD"), audit_id="s-2")
    trail.append_symbol_output(make_symbol_output("GBPUSD"), audit_id="s-3")

    assert trail.symbol_count == 2
    assert [record.audit_id for record in trail.symbol_records()] == ["s-2", "s-3"]


def test_audit_trail_rejects_invalid_constructor_and_missing_objects():
    with pytest.raises(ValueError, match="max_records"):
        AgentAuditTrail(max_records=0)

    trail = AgentAuditTrail()

    with pytest.raises(ValueError, match="output is required"):
        trail.append_symbol_output(None, audit_id="x")

    with pytest.raises(ValueError, match="decision is required"):
        trail.append_portfolio_decision(None, audit_id="x")
