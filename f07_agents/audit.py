# f07_agents/tests/audit.py (10)
#
# Created: 1405/06/24

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Mapping

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioDecision,
    SymbolAgentOutput,
)


def _validate_timestamp(timestamp: datetime) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware.")


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


@dataclass(frozen=True)
class SymbolDecisionAuditRecord:
    audit_id: str
    decision_id: str
    timestamp: datetime
    mode: DecisionMode
    symbol: str
    signal: int
    confidence: float
    expected_return: float
    risk_score: float
    desired_exposure: float
    stop_price: float | None
    model: ModelIdentity | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.audit_id).strip():
            raise ValueError("audit_id must be non-empty.")
        if not str(self.decision_id).strip():
            raise ValueError("decision_id must be non-empty.")
        if not str(self.symbol).strip():
            raise ValueError("symbol must be non-empty.")
        if not isinstance(self.mode, DecisionMode):
            raise TypeError("mode must be DecisionMode.")
        _validate_timestamp(self.timestamp)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True)
class MetaDecisionAuditRecord:
    audit_id: str
    decision_id: str
    timestamp: datetime
    mode: DecisionMode
    approved: bool
    capital_allocation: Mapping[str, float]
    margin_allocation: Mapping[str, float]
    target_exposure: Mapping[str, float]
    target_signals: Mapping[str, int]
    portfolio_risk: float
    reason_codes: tuple[str, ...]
    model: ModelIdentity | None
    target_stop_price: Mapping[str, float] | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.audit_id).strip():
            raise ValueError("audit_id must be non-empty.")
        if not str(self.decision_id).strip():
            raise ValueError("decision_id must be non-empty.")
        if not isinstance(self.mode, DecisionMode):
            raise TypeError("mode must be DecisionMode.")
        _validate_timestamp(self.timestamp)
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))
        object.__setattr__(
            self,
            "capital_allocation",
            dict(self.capital_allocation),
        )
        object.__setattr__(
            self,
            "margin_allocation",
            dict(self.margin_allocation),
        )
        object.__setattr__(self, "target_exposure", dict(self.target_exposure))
        object.__setattr__(self, "target_signals", dict(self.target_signals))
        object.__setattr__(
            self,
            "target_stop_price",
            None if self.target_stop_price is None else dict(self.target_stop_price),
        )


class AgentAuditTrail:
    """Thread-safe, bounded in-memory audit trail for the Decision Layer."""

    def __init__(self, *, max_records: int = 100_000) -> None:
        if max_records < 1:
            raise ValueError("max_records must be >= 1.")
        self.max_records = int(max_records)
        self._symbol: deque[SymbolDecisionAuditRecord] = deque(maxlen=self.max_records)
        self._meta: deque[MetaDecisionAuditRecord] = deque(maxlen=self.max_records)
        self._lock = RLock()

    @property
    def symbol_count(self) -> int:
        with self._lock:
            return len(self._symbol)

    @property
    def meta_count(self) -> int:
        with self._lock:
            return len(self._meta)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._symbol) + len(self._meta)

    @property
    def latest_symbol(self) -> SymbolDecisionAuditRecord | None:
        with self._lock:
            return self._symbol[-1] if self._symbol else None

    @property
    def latest_meta(self) -> MetaDecisionAuditRecord | None:
        with self._lock:
            return self._meta[-1] if self._meta else None

    def append_symbol_output(
        self,
        output: SymbolAgentOutput,
        *,
        audit_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> SymbolDecisionAuditRecord:
        if output is None:
            raise ValueError("output is required.")

        record = SymbolDecisionAuditRecord(
            audit_id=audit_id,
            decision_id=output.decision_id,
            timestamp=output.timestamp,
            mode=output.mode,
            symbol=output.symbol,
            signal=output.signal,
            confidence=output.confidence,
            expected_return=output.expected_return,
            risk_score=output.risk_score,
            desired_exposure=output.desired_exposure,
            stop_price=output.stop_price,
            model=output.model,
            metadata=metadata,
        )
        with self._lock:
            self._symbol.append(record)
        return record

    def append_portfolio_decision(
        self,
        decision: PortfolioDecision,
        *,
        audit_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> MetaDecisionAuditRecord:
        if decision is None:
            raise ValueError("decision is required.")

        record = MetaDecisionAuditRecord(
            audit_id=audit_id,
            decision_id=decision.decision_id,
            timestamp=decision.timestamp,
            mode=decision.mode,
            approved=decision.approved,
            capital_allocation=dict(decision.capital_allocation),
            margin_allocation=dict(decision.margin_allocation),
            target_exposure=dict(decision.target_exposure),
            target_signals=dict(decision.target_signals),
            portfolio_risk=decision.portfolio_risk,
            reason_codes=tuple(decision.reason_codes),
            model=decision.model,
            target_stop_price=(
                None
                if decision.target_stop_price is None
                else dict(decision.target_stop_price)
            ),
            metadata=metadata,
        )
        with self._lock:
            self._meta.append(record)
        return record

    def symbol_records(self) -> tuple[SymbolDecisionAuditRecord, ...]:
        with self._lock:
            return tuple(self._symbol)

    def meta_records(self) -> tuple[MetaDecisionAuditRecord, ...]:
        with self._lock:
            return tuple(self._meta)

    def clear(self) -> None:
        with self._lock:
            self._symbol.clear()
            self._meta.clear()


__all__ = [
    "SymbolDecisionAuditRecord",
    "MetaDecisionAuditRecord",
    "AgentAuditTrail",
]
