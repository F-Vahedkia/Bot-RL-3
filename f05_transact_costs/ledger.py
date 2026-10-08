from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
from threading import RLock

from .contracts import CostReconciliation, ExecutionFill


@dataclass
class TransactionCostLedger:
    """Thread-safe in-memory fill ledger; persistence is an external responsibility."""

    _fills: dict[str, ExecutionFill] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def record(self, fill: ExecutionFill) -> None:
        if not isinstance(fill, ExecutionFill):
            raise TypeError("fill must be ExecutionFill")
        with self._lock:
            if fill.fill_id in self._fills:
                raise ValueError(f"duplicate fill_id: {fill.fill_id}")
            self._fills[fill.fill_id] = fill

    def get(self, fill_id: str) -> ExecutionFill:
        with self._lock:
            try:
                return self._fills[str(fill_id)]
            except KeyError as exc:
                raise KeyError(f"unknown fill_id: {fill_id}") from exc

    def all(self) -> tuple[ExecutionFill, ...]:
        with self._lock:
            return tuple(self._fills.values())

    def count(self) -> int:
        with self._lock:
            return len(self._fills)

    @staticmethod
    def reconcile_execution(expected: Iterable[ExecutionFill], actual: Iterable[ExecutionFill]) -> CostReconciliation:
        exp = tuple(expected)
        act = tuple(actual)
        if not exp or not act:
            raise ValueError("expected and actual fill collections must be non-empty")
        if any(not isinstance(x, ExecutionFill) for x in (*exp, *act)):
            raise TypeError("all fills must be ExecutionFill")
        eids = {x.execution_id for x in exp}
        aids = {x.execution_id for x in act}
        if len(eids) != 1 or eids != aids:
            raise ValueError("expected and actual fills must share one execution_id")
        exp_key = {(x.symbol, x.side, x.role, x.cost.account_currency) for x in exp}
        act_key = {(x.symbol, x.side, x.role, x.cost.account_currency) for x in act}
        if len(exp_key) != 1 or len(act_key) != 1 or exp_key != act_key:
            raise ValueError("expected and actual fills must share symbol, side, role, and currency")
        if any(x.cost.realization.value != "expected" or x.source.value != "simulation" for x in exp):
            raise ValueError("expected fills must be simulation/expected")
        if any(x.cost.realization.value != "actual" or x.source.value != "broker" for x in act):
            raise ValueError("actual fills must be broker/actual")
        expected_lots = sum(x.lots for x in exp)
        actual_lots = sum(x.lots for x in act)
        expected_total = sum(x.cost.total_cost for x in exp)
        actual_total = sum(x.cost.total_cost for x in act)
        return CostReconciliation(
            execution_id=exp[0].execution_id,
            expected_lots=expected_lots,
            actual_lots=actual_lots,
            expected_total=expected_total,
            actual_total=actual_total,
            variance=actual_total - expected_total,
            account_currency=exp[0].cost.account_currency,
            slippage_cap_exceeded=any(x.slippage_cap_exceeded for x in act),
        )
