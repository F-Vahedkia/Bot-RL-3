from __future__ import annotations

from dataclasses import dataclass

from .calculator import TransactionCostCalculator
from .contracts import CostRealization, TransactionCostBreakdown, TransactionCostRequest


@dataclass(frozen=True, slots=True)
class RoundTripCostEstimate:
    """Two-fill expected cost: entry plus exit/stop leg."""

    entry: TransactionCostBreakdown
    exit: TransactionCostBreakdown

    @property
    def total_cost(self) -> float:
        return self.entry.total_cost + self.exit.total_cost

    @property
    def account_currency(self) -> str:
        if self.entry.account_currency != self.exit.account_currency:
            raise ValueError("entry and exit account currencies must match")
        return self.entry.account_currency


@dataclass(frozen=True, slots=True)
class RoundTripCostEstimator:
    calculator: TransactionCostCalculator

    def estimate(self, entry: TransactionCostRequest, exit: TransactionCostRequest) -> RoundTripCostEstimate:
        if entry.lots != exit.lots:
            raise ValueError("entry and exit lots must match for round-trip estimate")
        if entry.symbol != exit.symbol:
            raise ValueError("entry and exit symbols must match")
        if entry.account_currency != exit.account_currency:
            raise ValueError("entry and exit account currencies must match")
        return RoundTripCostEstimate(
            entry=self.calculator.calculate(entry, realization=CostRealization.EXPECTED, price_costs_embedded_in_fill=True),
            exit=self.calculator.calculate(exit, realization=CostRealization.EXPECTED, price_costs_embedded_in_fill=True),
        )
