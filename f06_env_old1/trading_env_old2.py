# f06_env/trading_env.py

"""V8 portfolio-aware trading environment with f07 transaction-cost integration.

f06_env owns environment progression, portfolio state mutation, reward, and
state-transition orchestration. Transaction-cost calculation is delegated to
f05_transact_costs through an already-composed ExecutionSimulator.

The environment consumes resolved simulation MarketQuote timelines. A scalar
price series is no longer an execution source because f07 requires historical
Bid/Ask for simulation transaction costs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np

from f06_env.contracts import EnvironmentConfig, PortfolioAction, StepResult
from f06_env.portfolio_state import PortfolioState, PositionState
from f06_env.execution_simulator import ExecutionSimulator
from f05_transact_costs import CostSource, ExecutionFill, MarketQuote


class TradingEnvironment:
    """Minimal V8 portfolio-aware trading environment.

    The environment is deliberately independent of Gym/Gymnasium and RL
    algorithms. Execution is delegated to f06_env.execution_simulator, which
    in turn delegates transaction-cost calculation to f05_transact_costs.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str, np.ndarray],
        quotes: Mapping[str, Sequence[MarketQuote]],
        config: EnvironmentConfig,
        execution: ExecutionSimulator | Mapping[str, ExecutionSimulator]
    ) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        if not quotes:
            raise ValueError("quotes must not be empty")
        if not isinstance(execution, (ExecutionSimulator, Mapping)):
            raise TypeError("execution must be ExecutionSimulator or a symbol mapping")

        self.config = config
        self.observations = {
            str(symbol).upper(): np.asarray(obs, dtype=np.float32)
            for symbol, obs in observations.items()
        }
        self.quotes = {
            str(symbol).upper(): tuple(quote_series)
            for symbol, quote_series in quotes.items()
        }
        self.symbols = tuple(self.observations.keys())

        if set(self.symbols) != set(self.quotes.keys()):
            raise ValueError("observations and quotes must contain the same symbols")

        for symbol, quote_series in self.quotes.items():
            if not quote_series:
                raise ValueError(f"quotes[{symbol}] must not be empty")
            for quote in quote_series:
                if not isinstance(quote, MarketQuote):
                    raise TypeError(f"quotes[{symbol}] must contain MarketQuote objects")
                if quote.symbol != symbol:
                    raise ValueError(
                        f"quote symbol mismatch for {symbol}: {quote.symbol}"
                    )
                if quote.source is not CostSource.SIMULATION:
                    raise ValueError(
                        f"TradingEnvironment requires simulation quotes for {symbol}"
                    )

        lengths = {symbol: len(series) for symbol, series in self.quotes.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"All symbols must have identical quote step counts: {lengths}"
            )
        self.n_steps = next(iter(lengths.values()))
        if self.n_steps < 2:
            raise ValueError("Environment requires at least 2 quote steps")

        for symbol, obs in self.observations.items():
            if len(obs) != self.n_steps:
                raise ValueError(
                    f"Observation length mismatch for {symbol}: "
                    f"{len(obs)} != {self.n_steps}"
                )

        self.execution = self._resolve_execution(execution)

        self.portfolio = PortfolioState(initial_balance=config.initial_balance)
        self._t = 0
        self._terminated = False
        self._truncated = False
        self._episode_index = -1

        # EnvironmentConfig.leverage remains the simulation-account leverage
        # used by this environment. Broker/account leverage belongs to the
        # live account/broker boundary and is not fabricated here.
        self._leverage = float(config.leverage)
        if not np.isfinite(self._leverage) or self._leverage <= 0.0:
            raise ValueError("config.leverage must be positive and finite")

    @staticmethod
    def _execution_symbols(
        execution: Mapping[str, ExecutionSimulator],
    ) -> dict[str, ExecutionSimulator]:
        result: dict[str, ExecutionSimulator] = {}
        for symbol, simulator in execution.items():
            key = str(symbol).upper()
            if not isinstance(simulator, ExecutionSimulator):
                raise TypeError(f"execution[{key}] must be ExecutionSimulator")
            result[key] = simulator
        return result

    def _resolve_execution(
        self,
        execution: ExecutionSimulator | Mapping[str, ExecutionSimulator],
    ) -> dict[str, ExecutionSimulator]:
        if isinstance(execution, Mapping):
            resolved = self._execution_symbols(execution)
            if set(resolved) != set(self.symbols):
                raise ValueError(
                    "execution mapping must contain exactly the environment symbols"
                )
        else:
            resolved = {symbol: execution for symbol in self.symbols}

        for symbol, simulator in resolved.items():
            instrument_symbol = simulator.transaction_cost_context.instrument.symbol
            if instrument_symbol != symbol:
                raise ValueError(
                    f"ExecutionSimulator instrument symbol {instrument_symbol!r} "
                    f"does not match environment symbol {symbol!r}"
                )
        return resolved

    def _current_observation(self) -> np.ndarray:
        arrays = [
            self.observations[symbol][self._t].reshape(-1)
            for symbol in self.symbols
        ]
        return np.concatenate(arrays).astype(np.float32, copy=False)

    def _current_quote(self, symbol: str) -> MarketQuote:
        return self.quotes[symbol][self._t]

    def _current_mark_prices(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)
            quote = self._current_quote(symbol)
            # Mark with the executable exit side: long positions are marked at
            # bid and shorts at ask. Flat symbols use the mid price.
            if position.side > 0:
                result[symbol] = quote.bid
            elif position.side < 0:
                result[symbol] = quote.ask
            else:
                result[symbol] = quote.mid
        return result

    def _calculate_unrealized(self) -> float:
        mark_prices = self._current_mark_prices()
        total = 0.0
        for symbol in self.symbols:
            total += self.execution[symbol].mark_to_market(
                positions={symbol: self.portfolio.get_position(symbol)},
                prices={symbol: mark_prices[symbol]},
            )
        return float(total)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError("seed must be >= 0")

        self._episode_index += 1
        self.portfolio.reset()
        self._t = 0
        self._terminated = False
        self._truncated = False
        for symbol in self.symbols:
            self.portfolio.set_position(PositionState(symbol=symbol))

        obs = self._current_observation()
        info = {
            "step": 0,
            "t": 0,
            "episode": self._episode_index,
            "symbols": self.symbols,
            "equity": self.portfolio.equity,
            "free_margin": self.portfolio.free_margin,
        }
        return obs, info

    def step(self, action: PortfolioAction) -> StepResult:
        equity_before = float(self.portfolio.equity)
        if self._terminated or self._truncated:
            raise RuntimeError("Environment is done. Call reset() before step().")
        if not isinstance(action, PortfolioAction):
            raise TypeError("action must be PortfolioAction")

        action_symbols = {intent.symbol for intent in action.intents}
        unknown = action_symbols - set(self.symbols)
        if unknown:
            raise ValueError(f"Action contains unknown symbols: {sorted(unknown)}")

        intents = {symbol: None for symbol in self.symbols}
        for intent in action.intents:
            intents[intent.symbol] = intent

        realized_delta = 0.0
        execution_info: dict[str, dict[str, object]] = {}

        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)
            intent = intents[symbol]
            if intent is None:
                continue

            quote = self._current_quote(symbol)
            execution_id = (
                f"ep{self._episode_index}:t{self._t}:{symbol}"
            )
            event_key = f"{execution_id}:simulation"
            execution_result = self.execution[symbol].execute(
                position,
                intent,
                quote=quote,
                execution_id=execution_id,
                execution_timestamp=quote.timestamp,
                event_key=event_key,
            )
            realized_delta += float(execution_result["realized_pnl"])

            fills = tuple(execution_result.get("execution_fills", ()))
            if not all(isinstance(fill, ExecutionFill) for fill in fills):
                raise TypeError("ExecutionSimulator returned a non-ExecutionFill result")

            execution_info[symbol] = execution_result

        next_t = self._t + 1
        terminated = next_t >= self.n_steps - 1
        if self.config.max_episode_steps is not None:
            truncated = (
                self.portfolio.step_count + 1 >= self.config.max_episode_steps
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

        risk_terminated = False
        termination_reason = None
        if (
            self.config.max_drawdown_pct is not None
            and self.portfolio.total_drawdown >= self.config.max_drawdown_pct
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

        reward = float(self.portfolio.equity - equity_before)
        self._terminated = bool(terminated or risk_terminated)
        self._truncated = bool(truncated)

        cash_cost_total = sum(
            float(result.get("cash_cost_total", 0.0))
            for result in execution_info.values()
        )
        price_cost_total = sum(
            float(result.get("price_cost_total", 0.0))
            for result in execution_info.values()
        )

        info = {
            "step": self.portfolio.step_count,
            "t": self._t,
            "episode": self._episode_index,
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
            "transaction_costs": {
                "cash_cost_total": float(cash_cost_total),
                "price_cost_total": float(price_cost_total),
            },
            "execution": execution_info,
        }
        return StepResult(
            observation=self._current_observation(),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        )

    def _calculate_used_margin(self) -> float:
        total = 0.0
        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)
            if position.side == 0 or position.lots <= 0.0:
                continue
            quote = self._current_quote(symbol)
            simulator = self.execution[symbol]
            contract_size = simulator.contract_size
            price = quote.executable_price(position.side)
            total += (
                price
                * position.lots
                * contract_size
                / self._leverage
            )
        return float(total)
