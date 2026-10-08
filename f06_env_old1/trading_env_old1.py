# f06_env/trading_env.py (4)

"""

"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from typing import Dict, Mapping, Optional
import numpy as np
from f06_env.contracts import (
    EnvironmentConfig,
    PortfolioAction,
    StepResult,
)
from f06_env.portfolio_state import (
    PortfolioState,
    PositionState,
)
from f06_env.execution_simulator import (
    ExecutionCost,
    ExecutionSimulator,
)

# =============================================================================
# Class-1
# =============================================================================
class TradingEnvironment:
    """
    Minimal v8 portfolio-aware trading environment.

    This class is deliberately independent of Gym/Gymnasium
    and independent of any RL algorithm.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str, np.ndarray],
        prices: Mapping[str, np.ndarray],
        config: EnvironmentConfig,
        execution: Optional[ExecutionSimulator] = None,
        execution_costs: Optional[Mapping[str, ExecutionCost]] = None,
    ) -> None:
        if not observations:
            raise ValueError("observations must not be empty")

        if not prices:
            raise ValueError("prices must not be empty")

        self.config = config

        self.observations = {
            str(symbol).upper(): np.asarray(obs, dtype=np.float32)
            for symbol, obs in observations.items()
        }

        self.prices = {
            str(symbol).upper(): np.asarray(values, dtype=np.float64)
            for symbol, values in prices.items()
        }

        self.symbols = tuple(self.observations.keys())

        if set(self.symbols) != set(self.prices.keys()):
            raise ValueError(
                "observations and prices must contain the same symbols"
            )

        lengths = {
            symbol: len(values)
            for symbol, values in self.prices.items()
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"All symbols must have identical step counts: {lengths}"
            )

        self.n_steps = next(iter(lengths.values()))

        if self.n_steps < 2:
            raise ValueError("Environment requires at least 2 price steps")

        for symbol, obs in self.observations.items():
            if len(obs) != self.n_steps:
                raise ValueError(
                    f"Observation length mismatch for {symbol}: "
                    f"{len(obs)} != {self.n_steps}"
                )

        self.execution = execution or ExecutionSimulator()

        self.execution_costs = {
            str(symbol).upper(): cost
            for symbol, cost in (execution_costs or {}).items()
        }

        self.portfolio = PortfolioState(
            initial_balance=config.initial_balance
        )

        self._t = 0
        self._terminated = False
        self._truncated = False

        self._leverage = float(config.leverage) # این هوش مصنوعی احمق دوباره اشتباه کرد. باید از مشخصات اکانت در بروکر این را بخواند.

    # ------------------------------------------------------------------
    def _current_observation(self) -> np.ndarray:
        """
        Temporary v8 core representation.

        For the multi-symbol core, observations are concatenated
        in deterministic symbol order.
        """
        arrays = [
            self.observations[symbol][self._t].reshape(-1)
            for symbol in self.symbols
        ]

        return np.concatenate(arrays).astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    def _current_prices(self) -> Dict[str, float]:
        return {
            symbol: float(self.prices[symbol][self._t])
            for symbol in self.symbols
        }

    # ------------------------------------------------------------------
    def _calculate_unrealized(self) -> float:
        return self.execution.mark_to_market(
            positions=self.portfolio.positions,
            prices=self._current_prices(),
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        self.portfolio.reset()
        self._t = 0
        self._terminated = False
        self._truncated = False

        for symbol in self.symbols:
            self.portfolio.set_position(
                PositionState(symbol=symbol)
            )

        obs = self._current_observation()

        info = {
            "step": 0,
            "symbols": self.symbols,
            "equity": self.portfolio.equity,
            "free_margin": self.portfolio.free_margin,
        }

        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: PortfolioAction) -> StepResult:
        equity_before = float(self.portfolio.equity)

        if self._terminated or self._truncated:
            raise RuntimeError(
                "Environment is done. Call reset() before step()."
            )

        if not isinstance(action, PortfolioAction):
            raise TypeError(
                "action must be PortfolioAction"
            )

        action_symbols = {
            intent.symbol for intent in action.intents
        }

        unknown = action_symbols - set(self.symbols)

        if unknown:
            raise ValueError(
                f"Action contains unknown symbols: {sorted(unknown)}"
            )

        # Missing symbols mean HOLD/current position.
        intents = {
            symbol: None
            for symbol in self.symbols
        }

        for intent in action.intents:
            intents[intent.symbol] = intent

        current_prices = self._current_prices()

        realized_delta = 0.0
        execution_info = {}

        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)

            intent = intents[symbol]

            if intent is None:
                continue

            result = self.execution.execute(
                position=position,
                intent=intent,
                market_price=current_prices[symbol],
                cost=self.execution_costs.get(
                    symbol,
                    ExecutionCost(),
                ),
            )

            realized_delta += result["realized_pnl"]
            execution_info[symbol] = result

        # Advance one market step.
        next_t = self._t + 1

        terminated = next_t >= self.n_steps - 1

        if self.config.max_episode_steps is not None:
            truncated = (
                self.portfolio.step_count + 1
                >= self.config.max_episode_steps
                and not terminated
            )
        else:
            truncated = False

        if next_t < self.n_steps:
            self._t = next_t

        unrealized_total = self._calculate_unrealized()

        used_margin = self._calculate_used_margin()

        self.portfolio.update_mark_to_market(
            realized_delta=realized_delta,
            unrealized_total=unrealized_total,
            used_margin=used_margin,
        )

        self.portfolio.advance_step()

        # ======================================= new added- start
        risk_terminated = False
        termination_reason = None

        if (
            self.config.max_drawdown_pct is not None
            and self.portfolio.total_drawdown
            >= self.config.max_drawdown_pct
        ):
            risk_terminated = True
            termination_reason = "max_drawdown"

        margin_level = self.portfolio.margin_level

        if (
            not risk_terminated
            and self.config.stop_out_level is not None
            and margin_level is not None
            and margin_level <= self.config.stop_out_level
        ):
            risk_terminated = True
            termination_reason = "margin_stop_out"
        # ======================================= new added- end

        reward = float(self.portfolio.equity - equity_before)

        self._terminated = bool(terminated or risk_terminated)
        self._truncated = bool(truncated)

        info = {
            "step": self.portfolio.step_count,
            "t": self._t,
            "symbols": self.symbols,
            "balance": self.portfolio.balance,
            "equity": self.portfolio.equity,
            "used_margin": self.portfolio.used_margin,
            "free_margin": self.portfolio.free_margin,
            "realized_pnl": self.portfolio.realized_pnl,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "drawdown": self.portfolio.total_drawdown,
            "margin_level": self.portfolio.margin_level,
            "termination_reason": termination_reason,
            "execution": execution_info,
        }

        return StepResult(
            observation=self._current_observation(),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        )

    # ------------------------------------------------------------------
    def _calculate_used_margin(self) -> float:
        total = 0.0

        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)

            if position.side == 0 or position.lots <= 0.0:
                continue

            price = float(self.prices[symbol][self._t])

            total += (
                price
                * position.lots
                * self.execution.contract_size
                / self._leverage
            )

        return float(total)

# ============================================================================= END
