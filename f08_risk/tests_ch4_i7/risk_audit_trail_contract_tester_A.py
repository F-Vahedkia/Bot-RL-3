# f08_risk/tests_ch4_i7/risk_audit_trail_contract_tester_A.py (45)
# Created at 1405/06/22

# Run: pytest -v -s f08_risk/tests_ch4_i7/risk_audit_trail_contract_tester_A.py

from __future__ import annotations
from datetime import datetime, timezone
import pytest

from f07_agents.contracts import DecisionMode, ModelIdentity
from f08_risk.audit import RiskAuditRecord, RiskAuditTrail
from f08_risk.contracts import RiskDecisionStatus, RiskViolation


TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="audit-trail-contract",
)

VIOLATION = RiskViolation(
    code="MAX_SYMBOL_EXPOSURE",
    message="Exposure was capped.",
    symbol="XAUUSD",
    severity="warning",
)


def make_record(audit_id: str = "audit-001", decision_id: str = "decision-001") -> RiskAuditRecord:
    return RiskAuditRecord(
        audit_id=audit_id,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        source_decision_id=decision_id,
        risk_decision_id=decision_id,
        status=RiskDecisionStatus.MODIFIED,
        model=MODEL,
        requested_target_exposure={"XAUUSD": 0.80},
        final_target_exposure={"XAUUSD": 0.50},
        requested_capital_allocation={"XAUUSD": 0.80},
        final_capital_allocation={"XAUUSD": 0.50},
        requested_margin_allocation={"XAUUSD": 0.40},
        final_margin_allocation={"XAUUSD": 0.30},
        requested_portfolio_risk=0.20,
        final_portfolio_risk=0.20,
        violations=(VIOLATION,),
        evaluation_path="projected",
    )


def test_append_and_count() -> None:
    trail = RiskAuditTrail()
    record = make_record()

    trail.append(record)

    assert trail.count == 1
    assert trail.latest == record


def test_records_preserve_insertion_order() -> None:
    trail = RiskAuditTrail()
    first = make_record("audit-001", "decision-001")
    second = make_record("audit-002", "decision-002")

    trail.append(first)
    trail.append(second)

    assert trail.records() == (first, second)


def test_get_returns_exact_record() -> None:
    trail = RiskAuditTrail()
    record = make_record()
    trail.append(record)

    assert trail.get("audit-001") is record


def test_get_missing_record_raises_key_error() -> None:
    trail = RiskAuditTrail()

    with pytest.raises(KeyError):
        trail.get("missing")


def test_for_decision_matches_source_or_risk_decision_id() -> None:
    trail = RiskAuditTrail()

    source_match = make_record("audit-001", "decision-001")
    other = make_record("audit-002", "decision-002")
    trail.append(source_match)
    trail.append(other)

    assert trail.for_decision("decision-001") == (source_match,)


def test_duplicate_audit_id_is_rejected() -> None:
    trail = RiskAuditTrail()
    trail.append(make_record("audit-001"))

    with pytest.raises(ValueError, match="Duplicate audit_id"):
        trail.append(make_record("audit-001", "decision-002"))


def test_non_record_is_rejected() -> None:
    trail = RiskAuditTrail()

    with pytest.raises(TypeError, match="RiskAuditRecord"):
        trail.append("not-a-record")  # type: ignore[arg-type]


def test_clear_resets_trail() -> None:
    trail = RiskAuditTrail()
    trail.append(make_record())

    trail.clear()

    assert trail.count == 0
    assert trail.latest is None
    assert trail.records() == ()

