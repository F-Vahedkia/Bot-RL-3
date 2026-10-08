from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional


@dataclass
class PositionState:
    symbol: str
    side: int = 0
    lots: float = 0.0
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    used_margin: float = 0.0
    notional: float = 0.0
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).upper().strip()
        self.side = int(self.side)
        self.lots = float(self.lots)
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.side not in (-1, 0, 1):
            raise ValueError("side must be -1, 0, or 1")
        if not math.isfinite(self.lots) or self.lots < 0.0:
            raise ValueError("lots must be finite and >= 0")
        if self.side == 0 and self.lots != 0.0:
            raise ValueError("flat position must have lots=0")


@dataclass
class PortfolioState:
    initial_balance: float = 10_000.0
    balance: float = 10_000.0
    equity: float = 10_000.0
    peak_equity: float = 10_000.0
    day_start_equity: float = 10_000.0
    used_margin: float = 0.0
    free_margin: float = 10_000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    step_count: int = 0
    positions: Dict[str, PositionState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial_balance = float(self.initial_balance)
        self.balance = float(self.balance)
        self.equity = float(self.equity)
        self.peak_equity = float(self.peak_equity)
        self.day_start_equity = float(self.day_start_equity)
        self.used_margin = float(self.used_margin)
        self.free_margin = float(self.free_margin)
        self.realized_pnl = float(self.realized_pnl)
        self.unrealized_pnl = float(self.unrealized_pnl)
        self.step_count = int(self.step_count)
        self._validate_numeric_state()

    def _validate_numeric_state(self) -> None:
        values = (
            self.initial_balance, self.balance, self.equity,
            self.peak_equity, self.day_start_equity,
            self.used_margin, self.free_margin,
            self.realized_pnl, self.unrealized_pnl,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("PortfolioState contains non-finite values")
        if self.initial_balance <= 0.0:
            raise ValueError("initial_balance must be > 0")
        if self.used_margin < 0.0:
            raise ValueError("used_margin must be >= 0")
        if self.free_margin < 0.0:
            raise ValueError("free_margin must be >= 0")

    @property
    def total_drawdown(self) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def daily_drawdown(self) -> float:
        if self.day_start_equity <= 0.0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity)

    @property
    def margin_level(self) -> Optional[float]:
        if self.used_margin <= 0.0:
            return None
        return self.equity / self.used_margin

    @property
    def open_position_symbols(self) -> tuple[str, ...]:
        return tuple(
            symbol for symbol, pos in self.positions.items()
            if pos.side != 0 and pos.lots > 0.0
        )

    @property
    def open_position_count(self) -> int:
        return len(self.open_position_symbols)

    def reset(self, balance: Optional[float] = None) -> None:
        value = self.initial_balance if balance is None else float(balance)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("reset balance must be > 0 and finite")
        self.initial_balance = value
        self.balance = value
        self.equity = value
        self.peak_equity = value
        self.day_start_equity = value
        self.used_margin = 0.0
        self.free_margin = value
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.step_count = 0
        self.positions.clear()

    def set_position(self, position: PositionState) -> None:
        if not isinstance(position, PositionState):
            raise TypeError("position must be PositionState")
        self.positions[position.symbol] = position

    def get_position(self, symbol: str) -> PositionState:
        key = str(symbol).upper().strip()
        if not key:
            raise ValueError("symbol is required")
        if key not in self.positions:
            self.positions[key] = PositionState(symbol=key)
        return self.positions[key]

    def remove_position(self, symbol: str) -> None:
        self.positions.pop(str(symbol).upper().strip(), None)

    def update_accounting(
        self,
        *,
        realized_delta: float = 0.0,
        unrealized_total: Optional[float] = None,
        used_margin: Optional[float] = None,
    ) -> None:
        self.realized_pnl += float(realized_delta)
        if unrealized_total is not None:
            self.unrealized_pnl = float(unrealized_total)
        self.balance = self.initial_balance + self.realized_pnl
        self.equity = self.balance + self.unrealized_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        if used_margin is not None:
            self.used_margin = max(0.0, float(used_margin))
        self.free_margin = max(0.0, self.equity - self.used_margin)
        self._validate_numeric_state()

    def advance_step(self) -> None:
        self.step_count += 1
