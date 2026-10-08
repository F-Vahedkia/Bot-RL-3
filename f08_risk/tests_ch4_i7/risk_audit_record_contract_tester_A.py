# f08_risk/tests_ch4_i7/risk_audit_record_contract_tester_A.py (44)
# Created at 1405/06/22
#
# Run: pytest -v -s f08_risk/tests_ch4_i7/risk_audit_record_contract_tester_A.py

from __future__ import annotations
from datetime import datetime, timezone
import pytest

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity
)
from f08_risk.audit import (
    RiskAuditModification,
    RiskAuditRecord,
)
from f08_risk.contracts import (
    RiskDecisionStatus,
    RiskViolation,
)
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingRequest,
    StopLossPositionSizingResult,
)

TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="audit-contract",
)

VIOLATION = RiskViolation(
    code="MAX_SYMBOL_EXPOSURE",
    message="Exposure was capped.",
    symbol="XAUUSD",
    severity="warning",
)

STOP_LOSS_REQUEST = StopLossPositionSizingRequest(
    symbol="XAUUSD",
    equity=10_000.0,
    risk_per_trade=0.01,
    entry_price=2000.0,
    stop_price=1990.0,
    contract_size=100.0,
    currency_conversion_rate=1.0,
)

STOP_LOSS_RESULT = StopLossPositionSizingResult(
    symbol="XAUUSD",
    equity=10_000.0,
    risk_per_trade=0.01,
    risk_budget=100.0,
    entry_price=2000.0,
    stop_price=1990.0,
    stop_distance=10.0,
    contract_size=100.0,
    currency_conversion_rate=1.0,
    loss_per_lot_at_stop=1000.0,
    volume_lots=0.1,
)

def make_record(**overrides: object) -> RiskAuditRecord:
    values: dict[str, object] = {
        "audit_id": "audit-001",
        "timestamp": TS,
        "mode": DecisionMode.BACKTEST,
        "source_decision_id": "decision-001",
        "risk_decision_id": "decision-001",
        "status": RiskDecisionStatus.MODIFIED,
        "model": MODEL,
        "requested_target_exposure": {"XAUUSD": 0.80},
        "final_target_exposure": {"XAUUSD": 0.50},
        "requested_capital_allocation": {"XAUUSD": 0.80},
        "final_capital_allocation": {"XAUUSD": 0.50},
        "requested_margin_allocation": {"XAUUSD": 0.40},
        "final_margin_allocation": {"XAUUSD": 0.30},
        "requested_portfolio_risk": 0.20,
        "final_portfolio_risk": 0.20,
        "violations": (VIOLATION,),
        "evaluation_path": "projected",
    }
    values.update(overrides)
    return RiskAuditRecord(**values)  # type: ignore[arg-type]


def test_valid_record_is_constructed() -> None:
    record = make_record()
    assert record.audit_id == "audit-001"
    assert record.source_decision_id == "decision-001"
    assert record.status == RiskDecisionStatus.MODIFIED
    assert record.violations == (VIOLATION,)


def test_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_record(timestamp=datetime(2026, 1, 5, 12, 0))


def test_required_ids_are_non_empty() -> None:
    with pytest.raises(ValueError, match="audit_id"):
        make_record(audit_id="")
    with pytest.raises(ValueError, match="source_decision_id"):
        make_record(source_decision_id="")
    with pytest.raises(ValueError, match="risk_decision_id"):
        make_record(risk_decision_id="")


def test_signed_exposure_is_allowed() -> None:
    record = make_record(
        requested_target_exposure={"EURUSD": -0.40},
        final_target_exposure={"EURUSD": -0.20},
    )
    assert record.requested_target_exposure["EURUSD"] == -0.40
    assert record.final_target_exposure["EURUSD"] == -0.20


def test_allocation_mappings_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="requested_capital_allocation"):
        make_record(requested_capital_allocation={"XAUUSD": -0.10})

    with pytest.raises(ValueError, match="final_margin_allocation"):
        make_record(final_margin_allocation={"XAUUSD": -0.10})


def test_portfolio_risk_must_be_finite_and_non_negative() -> None:
    with pytest.raises(ValueError, match="requested_portfolio_risk"):
        make_record(requested_portfolio_risk=-0.01)

    with pytest.raises(ValueError, match="final_portfolio_risk"):
        make_record(final_portfolio_risk=float("inf"))


def test_violations_must_be_risk_violations() -> None:
    with pytest.raises(TypeError, match="RiskViolation"):
        make_record(violations=("MAX_SYMBOL_EXPOSURE",))


def test_record_is_immutable() -> None:
    record = make_record()
    with pytest.raises(AttributeError):
        record.audit_id = "changed"  # type: ignore[misc]


def test_stop_loss_requests_are_preserved() -> None:
    record = make_record(
        stop_loss_requests={
            "XAUUSD": STOP_LOSS_REQUEST,
        }
    )

    assert record.stop_loss_requests["XAUUSD"] is STOP_LOSS_REQUEST


def test_stop_loss_results_are_preserved() -> None:
    record = make_record(
        stop_loss_results={
            "XAUUSD": STOP_LOSS_RESULT,
        }
    )

    assert record.stop_loss_results["XAUUSD"] is STOP_LOSS_RESULT


def test_stop_loss_mappings_are_read_only() -> None:
    record = make_record(
        stop_loss_requests={
            "XAUUSD": STOP_LOSS_REQUEST,
        },
        stop_loss_results={
            "XAUUSD": STOP_LOSS_RESULT,
        },
    )

    with pytest.raises(TypeError):
        record.stop_loss_requests["EURUSD"] = STOP_LOSS_REQUEST  # type: ignore[index]

    with pytest.raises(TypeError):
        record.stop_loss_results["EURUSD"] = STOP_LOSS_RESULT  # type: ignore[index]


def test_stop_loss_mapping_rejects_wrong_value_type() -> None:
    with pytest.raises(
        TypeError,
        match="StopLossPositionSizingRequest",
    ):
        make_record(
            stop_loss_requests={
                "XAUUSD": STOP_LOSS_RESULT,  # type: ignore[dict-item]
            }
        )

    with pytest.raises(
        TypeError,
        match="StopLossPositionSizingResult",
    ):
        make_record(
            stop_loss_results={
                "XAUUSD": STOP_LOSS_REQUEST,  # type: ignore[dict-item]
            }
        )


def test_modifications_are_preserved_and_read_only() -> None:
    modification = RiskAuditModification(
        constraint_code="MAX_SYMBOL_EXPOSURE",
        symbol="XAUUSD",
        field="target_exposure",
        requested_value=0.80,
        final_value=0.50,
    )

    record = make_record(
        modifications=(modification,),
    )

    assert record.modifications == (modification,)

    with pytest.raises(TypeError):
        record.modifications[0] = modification  # type: ignore[index]


def test_modifications_are_preserved_and_immutable() -> None:
    modification = RiskAuditModification(
        constraint_code="MAX_SYMBOL_EXPOSURE",
        symbol="XAUUSD",
        field="target_exposure",
        requested_value=0.80,
        final_value=0.50,
    )

    record = make_record(
        modifications=(modification,),
    )

    assert record.modifications == (modification,)

    with pytest.raises(TypeError):
        record.modifications[0] = modification  # type: ignore[index]

