from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class ObservationContract:
    symbol: str
    base_tf: str
    columns: Tuple[str, ...]
    row_count: int

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()
        base_tf = str(self.base_tf).upper().strip()
        if not symbol:
            raise ValueError("symbol is required")
        if not base_tf:
            raise ValueError("base_tf is required")
        row_count = int(self.row_count)
        if row_count < 0:
            raise ValueError("row_count must be >= 0")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "base_tf", base_tf)
        object.__setattr__(self, "columns", tuple(str(c) for c in self.columns))
        object.__setattr__(self, "row_count", row_count)

    @property
    def feature_count(self) -> int:
        return len(self.columns)


@dataclass(frozen=True, slots=True)
class PositionIntent:
    """Final semantic position target delivered to f06_env by the decision/risk path."""

    symbol: str
    target_side: int = 0
    target_lots: float = 0.0
    stop_price: Optional[float] = None

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()
        side = int(self.target_side)
        lots = float(self.target_lots)
        if not symbol:
            raise ValueError("symbol is required")
        if side not in (-1, 0, 1):
            raise ValueError("target_side must be -1, 0, or 1")
        if not np.isfinite(lots) or lots < 0.0:
            raise ValueError("target_lots must be finite and >= 0")
        stop_price = None if self.stop_price is None else float(self.stop_price)
        if stop_price is not None and (not np.isfinite(stop_price) or stop_price <= 0.0):
            raise ValueError("stop_price must be positive and finite")
        if side == 0:
            if lots != 0.0:
                raise ValueError("flat intent must have target_lots=0")
            if stop_price is not None:
                raise ValueError("flat intent must have stop_price=None")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "target_side", side)
        object.__setattr__(self, "target_lots", lots)
        object.__setattr__(self, "stop_price", stop_price)


@dataclass(frozen=True, slots=True)
class PortfolioAction:
    intents: Tuple[PositionIntent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        intents = tuple(self.intents)
        seen: set[str] = set()
        for intent in intents:
            if not isinstance(intent, PositionIntent):
                raise TypeError("PortfolioAction.intents must contain PositionIntent objects")
            if intent.symbol in seen:
                raise ValueError(f"Duplicate symbol in PortfolioAction: {intent.symbol}")
            seen.add(intent.symbol)
        object.__setattr__(self, "intents", intents)


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation)
        if not np.issubdtype(observation.dtype, np.number):
            raise TypeError("StepResult.observation must be numeric")
        reward = float(self.reward)
        if not np.isfinite(reward):
            raise ValueError("StepResult.reward must be finite")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "info", dict(self.info))


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Simulation-only configuration. Broker runtime account settings do not belong here."""

    base_tf: str = "M1"
    window_size: int = 128
    initial_balance: float = 10_000.0
    max_episode_steps: Optional[int] = None
    max_drawdown_pct: Optional[float] = None
    stop_out_level: Optional[float] = None
    leverage: float = 1.0

    def __post_init__(self) -> None:
        base_tf = str(self.base_tf).upper().strip()
        window_size = int(self.window_size)
        initial_balance = float(self.initial_balance)
        max_steps = None if self.max_episode_steps is None else int(self.max_episode_steps)
        max_drawdown = None if self.max_drawdown_pct is None else float(self.max_drawdown_pct)
        stop_out = None if self.stop_out_level is None else float(self.stop_out_level)
        leverage = float(self.leverage)

        if not base_tf:
            raise ValueError("base_tf is required")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if not np.isfinite(initial_balance) or initial_balance <= 0.0:
            raise ValueError("initial_balance must be > 0 and finite")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        if max_drawdown is not None and not (0.0 < max_drawdown < 1.0):
            raise ValueError("max_drawdown_pct must be between 0 and 1")
        if stop_out is not None and (not np.isfinite(stop_out) or stop_out <= 0.0):
            raise ValueError("stop_out_level must be > 0 and finite")
        if not np.isfinite(leverage) or leverage <= 0.0:
            raise ValueError("leverage must be > 0 and finite")

        object.__setattr__(self, "base_tf", base_tf)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "initial_balance", initial_balance)
        object.__setattr__(self, "max_episode_steps", max_steps)
        object.__setattr__(self, "max_drawdown_pct", max_drawdown)
        object.__setattr__(self, "stop_out_level", stop_out)
        object.__setattr__(self, "leverage", leverage)
