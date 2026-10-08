# f06_env/trading_env.py

"""V8 portfolio-aware trading environment.

The environment consumes the Observation produced upstream. The same market
OHLC+spread data used to build the Observation may be supplied as
``market_data``; when the Observation itself is a DataFrame with the standard
market columns, it is used directly and no second market-data input is needed.

For the frozen f03_data contract used by Bot-RL-3 V8:
    close  = historical Bid close
    spread = MT5 historical spread in points

The environment reconstructs a historical executable quote as:
    bid   = close
    ask   = bid + spread * instrument.tick_size

No broker/API access is performed here. Transaction costs are delegated to
f05_transact_costs through the already-composed ExecutionSimulator.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from f06_env.contracts import EnvironmentConfig, PortfolioAction, StepResult
from f06_env.portfolio_state import PortfolioState, PositionState
from f06_env.execution_simulator import ExecutionSimulator
from f05_transact_costs import CostSource, ExecutionFill, InstrumentSpec, MarketQuote


_REQUIRED_MARKET_COLUMNS = ("open", "high", "low", "close", "spread")


class TradingEnvironment:
    """Minimal V8 portfolio-aware simulation environment.

    ``observations`` is keyed by symbol. Each value may be either:

    * a pandas DataFrame (preferred): the final Observation, carrying the
      market columns required for historical quote reconstruction; or
    * a numeric NumPy array, when ``market_data`` is supplied separately.

    ``market_data`` is optional. When omitted, each observation must be a
    DataFrame containing ``open/high/low/close/spread``. The environment never
    reads f03_data directly and never connects to a broker.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str, pd.DataFrame | np.ndarray],
        config: EnvironmentConfig,
        execution: ExecutionSimulator | Mapping[str, ExecutionSimulator],
        market_data: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        if not isinstance(execution, (ExecutionSimulator, Mapping)):
            raise TypeError("execution must be ExecutionSimulator or a symbol mapping")
        if market_data is not None and not isinstance(market_data, Mapping):
            raise TypeError("market_data must be a symbol mapping or None")

        self.config = config
        self.symbols = tuple(str(symbol).upper().strip() for symbol in observations)
        if any(not symbol for symbol in self.symbols):
            raise ValueError("observation symbols must be non-empty")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("duplicate observation symbols are not allowed")

        normalized_observations: dict[str, np.ndarray] = {}
        observation_index: dict[str, pd.DatetimeIndex | None] = {}
        observation_frames: dict[str, pd.DataFrame | None] = {}

        for raw_symbol, value in observations.items():
            symbol = str(raw_symbol).upper().strip()
            if isinstance(value, pd.DataFrame):
                frame = self._normalize_frame(value, f"observations[{symbol}]")
                if frame.empty:
                    raise ValueError(f"observations[{symbol}] must not be empty")
                normalized_observations[symbol] = frame.to_numpy(dtype=np.float32, copy=True)
                observation_index[symbol] = frame.index
                observation_frames[symbol] = frame
            else:
                array = np.asarray(value, dtype=np.float32)
                if array.ndim != 2:
                    raise ValueError(f"observations[{symbol}] must be a 2-D array")
                if len(array) == 0:
                    raise ValueError(f"observations[{symbol}] must not be empty")
                normalized_observations[symbol] = np.array(array, dtype=np.float32, copy=True)
                observation_index[symbol] = None
                observation_frames[symbol] = None

        self.observations = normalized_observations

        if market_data is None:
            market_source = {
                symbol: frame
                for symbol, frame in observation_frames.items()
                if frame is not None
            }
            if len(market_source) != len(self.symbols):
                raise ValueError(
                    "market_data is required when observations are NumPy arrays"
                )
        else:
            market_source = {}
            for raw_symbol, frame in market_data.items():
                symbol = str(raw_symbol).upper().strip()
                market_source[symbol] = frame
            if set(market_source) != set(self.symbols):
                raise ValueError(
                    "market_data must contain exactly the environment symbols"
                )

        self.execution = self._resolve_execution(execution)
        self.market_data = {
            symbol: self._normalize_frame(
                market_source[symbol], f"market_data[{symbol}]"
            )
            for symbol in self.symbols
        }

        lengths = {symbol: len(self.market_data[symbol]) for symbol in self.symbols}
        if any(length < 2 for length in lengths.values()):
            raise ValueError("Environment requires at least 2 market-data steps")
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"All symbols must have identical market-data step counts: {lengths}"
            )
        self.n_steps = next(iter(lengths.values()))

        for symbol in self.symbols:
            obs = self.observations[symbol]
            if len(obs) != self.n_steps:
                raise ValueError(
                    f"Observation length mismatch for {symbol}: "
                    f"{len(obs)} != {self.n_steps}"
                )
            obs_index = observation_index[symbol]
            if obs_index is not None and not obs_index.equals(self.market_data[symbol].index):
                raise ValueError(
                    f"Observation and market-data timestamps must match exactly for {symbol}"
                )

        self.quotes = self._build_all_quotes()
        self._validate_quote_timelines()

        self.portfolio = PortfolioState(initial_balance=config.initial_balance)
        self._t = 0
        self._terminated = False
        self._truncated = False
        self._episode_index = -1

        self._leverage = float(config.leverage)
        if not np.isfinite(self._leverage) or self._leverage <= 0.0:
            raise ValueError("config.leverage must be positive and finite")

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame, field_name: str) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{field_name} must be a pandas DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{field_name} index must be a pandas DatetimeIndex")
        if frame.index.tz is None:
            raise ValueError(f"{field_name} index must be timezone-aware")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{field_name} index must be monotonic increasing")
        if frame.index.has_duplicates:
            raise ValueError(f"{field_name} index must not contain duplicate timestamps")
        return frame.copy(deep=True).astype(float).set_axis(
            frame.index.tz_convert("UTC"), axis="index"
        )

    @staticmethod
    def _execution_symbols(
        execution: Mapping[str, ExecutionSimulator],
    ) -> dict[str, ExecutionSimulator]:
        result: dict[str, ExecutionSimulator] = {}
        for raw_symbol, simulator in execution.items():
            symbol = str(raw_symbol).upper().strip()
            if not isinstance(simulator, ExecutionSimulator):
                raise TypeError(f"execution[{symbol}] must be ExecutionSimulator")
            result[symbol] = simulator
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
            instrument = simulator.transaction_cost_context.instrument
            if not isinstance(instrument, InstrumentSpec):
                raise TypeError("ExecutionSimulator instrument must be InstrumentSpec")
            if instrument.symbol != symbol:
                raise ValueError(
                    f"ExecutionSimulator instrument symbol {instrument.symbol!r} "
                    f"does not match environment symbol {symbol!r}"
                )
        return resolved

    @staticmethod
    def _market_columns(frame: pd.DataFrame, symbol: str) -> tuple[str, str]:
        missing = [column for column in _REQUIRED_MARKET_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                f"market data for {symbol} must contain "
                f"open/high/low/close/spread; missing: {missing}"
            )
        return "close", "spread"

    def _build_simulation_quotes(
        self,
        *,
        symbol: str,
        frame: pd.DataFrame,
        instrument: InstrumentSpec,
    ) -> tuple[MarketQuote, ...]:
        close_column, spread_column = self._market_columns(frame, symbol)
        tick_size = float(instrument.tick_size)
        if not np.isfinite(tick_size) or tick_size <= 0.0:
            raise ValueError(f"instrument.tick_size for {symbol} must be positive and finite")

        quotes: list[MarketQuote] = []
        for timestamp, row in frame.iterrows():
            bid = float(row[close_column])
            spread_points = float(row[spread_column])
            if not np.isfinite(bid) or bid <= 0.0:
                raise ValueError(f"invalid Bid close at {symbol} {timestamp}")
            if not np.isfinite(spread_points) or spread_points < 0.0:
                raise ValueError(f"invalid spread at {symbol} {timestamp}")
            ask = bid + spread_points * tick_size
            quotes.append(
                MarketQuote(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    timestamp=timestamp.to_pydatetime(),
                    source=CostSource.SIMULATION,
                )
            )
        return tuple(quotes)

    def _build_all_quotes(self) -> dict[str, tuple[MarketQuote, ...]]:
        result: dict[str, tuple[MarketQuote, ...]] = {}
        for symbol in self.symbols:
            instrument = self.execution[symbol].transaction_cost_context.instrument
            result[symbol] = self._build_simulation_quotes(
                symbol=symbol,
                frame=self.market_data[symbol],
                instrument=instrument,
            )
        return result

    def _validate_quote_timelines(self) -> None:
        first_symbol = self.symbols[0]
        reference = tuple(quote.timestamp for quote in self.quotes[first_symbol])
        for symbol in self.symbols[1:]:
            current = tuple(quote.timestamp for quote in self.quotes[symbol])
            if current != reference:
                raise ValueError("All symbols must share the same quote timestamps")

    def _current_observation(self) -> np.ndarray:
        arrays = [self.observations[symbol][self._t].reshape(-1) for symbol in self.symbols]
        return np.concatenate(arrays).astype(np.float32, copy=False)

    def _current_quote(self, symbol: str) -> MarketQuote:
        return self.quotes[symbol][self._t]

    def _current_mark_prices(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)
            quote = self._current_quote(symbol)
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

    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
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

        return self._current_observation(), {
            "step": 0,
            "t": 0,
            "episode": self._episode_index,
            "symbols": self.symbols,
            "timestamp": self.quotes[self.symbols[0]][0].timestamp,
            "equity": self.portfolio.equity,
            "free_margin": self.portfolio.free_margin,
        }

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
            execution_id = f"ep{self._episode_index}:t{self._t}:{symbol}"
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
            "timestamp": self.quotes[self.symbols[0]][self._t].timestamp,
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
