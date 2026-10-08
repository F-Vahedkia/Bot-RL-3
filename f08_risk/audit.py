# f08_risk/audit.py (43)
# Created: 1405/06/22

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Tuple

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity
)
from f08_risk.contracts import (
    RiskDecisionStatus,
    RiskViolation
)
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingRequest,
    StopLossPositionSizingResult,
)

# =============================================================================
# Class 1
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskAuditModification:
    """
    Immutable record describing one concrete Risk constraint modification.
    """

    constraint_code: str
    symbol: str | None
    field: str
    requested_value: float
    final_value: float

    def __post_init__(self) -> None:
        if not str(self.constraint_code).strip():
            raise ValueError("constraint_code is required")

        if self.symbol is not None:
            normalized_symbol = str(self.symbol).strip().upper()

            if not normalized_symbol:
                raise ValueError("symbol must be non-empty when provided")

            object.__setattr__(self, "symbol", normalized_symbol)

        if not str(self.field).strip():
            raise ValueError("field is required")

        requested = float(self.requested_value)
        final = float(self.final_value)

        if not isfinite(requested):
            raise ValueError("requested_value must be finite")

        if not isfinite(final):
            raise ValueError("final_value must be finite")

        object.__setattr__(self, "requested_value", requested)
        object.__setattr__(self, "final_value", final)

# =============================================================================
# Class 2
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskAuditRecord:
    """
    Immutable audit/lineage snapshot for one Risk evaluation.

    The record captures the lineage from the source PortfolioDecision to the
    final RiskDecision. Mutable mappings are defensively copied and exposed as
    read-only mapping proxies.
    """

    audit_id: str
    timestamp: datetime
    mode: DecisionMode
    source_decision_id: str
    risk_decision_id: str
    status: RiskDecisionStatus
    model: ModelIdentity

    requested_target_exposure: Mapping[str, float]
    final_target_exposure: Mapping[str, float]

    requested_capital_allocation: Mapping[str, float]
    final_capital_allocation: Mapping[str, float]

    requested_margin_allocation: Mapping[str, float]
    final_margin_allocation: Mapping[str, float]

    requested_portfolio_risk: float
    final_portfolio_risk: float

    violations: Tuple[RiskViolation, ...]
    evaluation_path: str

    modifications: Tuple[RiskAuditModification, ...] = ()

    stop_loss_requests: Mapping[
        str,
        StopLossPositionSizingRequest,
    ] = dataclass_field(default_factory=dict)

    stop_loss_results: Mapping[
        str,
        StopLossPositionSizingResult,
    ] = dataclass_field(default_factory=dict)


    # =========================================================================
    def __post_init__(self) -> None:
        if not str(self.audit_id).strip():
            raise ValueError("audit_id is required")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if not str(self.source_decision_id).strip():
            raise ValueError("source_decision_id is required")

        if not str(self.risk_decision_id).strip():
            raise ValueError("risk_decision_id is required")

        if not isinstance(self.status, RiskDecisionStatus):
            raise TypeError("status must be RiskDecisionStatus")

        if not str(self.evaluation_path).strip():
            raise ValueError("evaluation_path is required")

        if not isinstance(self.violations, tuple):
            raise TypeError("violations must be a tuple")

        for violation in self.violations:
            if not isinstance(violation, RiskViolation):
                raise TypeError("violations must contain RiskViolation objects")
            
        #-----new block-1 start
        if not isinstance(self.modifications, tuple):
            raise TypeError("modifications must be a tuple")

        for modification in self.modifications:
            if not isinstance(modification, RiskAuditModification):
                raise TypeError("modifications must contain RiskAuditModification objects")
            
        #-----new block-1 end

        for name, value in (
            ("requested_portfolio_risk", self.requested_portfolio_risk),
            ("final_portfolio_risk", self.final_portfolio_risk),
        ):
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")

        for field_name, allow_negative in (
            ("requested_target_exposure", True),
            ("final_target_exposure", True),
            ("requested_capital_allocation", False),
            ("final_capital_allocation", False),
            ("requested_margin_allocation", False),
            ("final_margin_allocation", False),
        ):
            original = getattr(self, field_name)
            normalized = self._validate_mapping(
                original,
                allow_negative=allow_negative,
                name=field_name,
            )
            object.__setattr__(self, field_name, MappingProxyType(normalized))


        for field_name in (
            "stop_loss_requests",
            "stop_loss_results",
        ):
            original = getattr(self, field_name)
            if not isinstance(original, Mapping):
                raise TypeError(
                    f"{field_name} must be a mapping"
                )

            normalized = {}

            for symbol, value in original.items():
                key = str(symbol).strip().upper()

                if not key:
                    raise ValueError(
                        f"{field_name} contains an empty symbol"
                    )

                if field_name == "stop_loss_requests":
                    if not isinstance(
                        value,
                        StopLossPositionSizingRequest,
                    ):
                        raise TypeError(
                            f"{field_name} must contain "
                            "StopLossPositionSizingRequest objects"
                        )
                else:
                    if not isinstance(
                        value,
                        StopLossPositionSizingResult,
                    ):
                        raise TypeError(
                            f"{field_name} must contain "
                            "StopLossPositionSizingResult objects"
                        )

                normalized[key] = value

            object.__setattr__(
                self,
                field_name,
                MappingProxyType(normalized),
            )

    # =========================================================================
    @staticmethod
    def _validate_mapping(
        mapping: Mapping[str, float],
        *,
        allow_negative: bool,
        name: str,
    ) -> dict[str, float]:
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a mapping")

        normalized: dict[str, float] = {}
        for symbol, value in mapping.items():
            key = str(symbol).strip().upper()
            if not key:
                raise ValueError(f"{name} contains an empty symbol")

            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"Invalid {name} value for {symbol}")
            if not allow_negative and numeric < 0.0:
                raise ValueError(f"Invalid {name} value for {symbol}")

            normalized[key] = numeric

        return normalized

# =============================================================================
# Class 3
# =============================================================================

class RiskAuditTrail:
    """
    In-memory ordered audit trail for completed Risk evaluations.

    This is intentionally separate from RiskEngineState. RiskEngineState is
    runtime statistics; RiskAuditTrail preserves individual audit records.
    """

    # =========================================================================
    def __init__(self) -> None:
        self._records: list[RiskAuditRecord] = []

    # =========================================================================
    def append(self, record: RiskAuditRecord) -> None:
        if not isinstance(record, RiskAuditRecord):
            raise TypeError(
                "record must be RiskAuditRecord"
            )

        if any(
            existing.audit_id == record.audit_id
            for existing in self._records
        ):
            raise ValueError(
                f"Duplicate audit_id: {record.audit_id}"
            )

        self._records.append(record)

    # =========================================================================
    def get(self, audit_id: str) -> RiskAuditRecord:
        normalized = str(audit_id).strip()
        if not normalized:
            raise ValueError("audit_id is required")

        for record in self._records:
            if record.audit_id == normalized:
                return record

        raise KeyError(normalized)

    # =========================================================================
    def for_decision(self, decision_id: str) -> Tuple[RiskAuditRecord, ...]:
        normalized = str(decision_id).strip()
        if not normalized:
            raise ValueError("decision_id is required")

        return tuple(
            record
            for record in self._records
            if (
                record.source_decision_id == normalized
                or record.risk_decision_id == normalized
            )
        )

    # =========================================================================
    @property
    def latest(self) -> RiskAuditRecord | None:
        if not self._records:
            return None
        return self._records[-1]

    # =========================================================================
    @property
    def count(self) -> int:
        return len(self._records)

    # =========================================================================
    def records(self) -> Tuple[RiskAuditRecord, ...]:
        return tuple(self._records)

    # =========================================================================
    def clear(self) -> None:
        self._records.clear()

# ============================================================================= END
