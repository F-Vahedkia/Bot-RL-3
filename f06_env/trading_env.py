from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


from f03_data.instrument_specs import InstrumentSpec
from f05_transact_costs.contracts import MarketQuote

from f06_env.contracts import EnvironmentConfig, PortfolioAction, StepResult
from f06_env.execution_simulator import ExecutionCost, ExecutionSimulator
from f06_env.historical_quote_resolver import HistoricalQuoteResolver
from f06_env.portfolio_state import PortfolioState, PositionState

ObservationLike = np.ndarray | pd.DataFrame


class TradingEnvironment:
    """Deterministic candle-based portfolio simulation environment.

    Core boundary:
        observations + already-resolved historical executable quotes
        + resolved InstrumentSpec -> simulation state transitions.

    This class performs no MT5 calls, no file I/O, no config parsing and no
    transaction-cost policy construction. Those responsibilities remain in
    f02/f07/composition-root components.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str, ObservationLike],
        market_quotes: Mapping[str, Sequence[MarketQuote]],
        instruments: Mapping[str, InstrumentSpec],
        config: EnvironmentConfig,
        execution: Optional[ExecutionSimulator] = None,
        execution_costs: Optional[Mapping[str, ExecutionCost]] = None,
        margin_rates_to_account: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        if not market_quotes:
            raise ValueError("market_quotes must not be empty")
        if not instruments:
            raise ValueError("instruments must not be empty")

        self.config = config
        self.observations, self.observation_indices = self._normalize_observations(observations)
        self.quotes = self._normalize_quotes(market_quotes)
        self.instruments = self._normalize_instruments(instruments)
        self.symbols = tuple(self.observations.keys())

        if set(self.symbols) != set(self.quotes) or set(self.symbols) != set(self.instruments):
            raise ValueError("observations, market_quotes and instruments must contain exactly the same symbols")

        self.n_steps = self._validate_timelines()
        self.execution = execution or ExecutionSimulator()
        self.execution_costs = {
            str(k).upper().strip(): v for k, v in (execution_costs or {}).items()
        }
        self.margin_rates_to_account = {
            str(k).upper().strip(): float(v)
            for k, v in (margin_rates_to_account or {}).items()
        }
        for symbol, rate in self.margin_rates_to_account.items():
            if not np.isfinite(rate) or rate <= 0.0:
                raise ValueError(f"margin_rates_to_account[{symbol}] must be positive and finite")

        for symbol, instrument in self.instruments.items():
            if instrument.contract_size is None:
                raise ValueError(f"InstrumentSpec.contract_size is required for simulation margin: {symbol}")

        self.portfolio = PortfolioState(initial_balance=config.initial_balance)
        self._t = 0
        self._terminated = False
        self._truncated = False
        self._episode_index = -1


    @staticmethod
    def _normalize_observations(
        observations: Mapping[str, ObservationLike],
    ) -> tuple[dict[str, np.ndarray], dict[str, pd.DatetimeIndex | None]]:
        arrays: dict[str, np.ndarray] = {}
        indices: dict[str, pd.DatetimeIndex | None] = {}
        for raw_symbol, raw_obs in observations.items():
            symbol = str(raw_symbol).upper().strip()
            if not symbol:
                raise ValueError("observation symbol is empty")
            if isinstance(raw_obs, pd.DataFrame):
                frame = raw_obs
                if not isinstance(frame.index, pd.DatetimeIndex):
                    raise TypeError(f"observations[{symbol}] index must be DatetimeIndex")
                if frame.index.tz is None:
                    raise ValueError(f"observations[{symbol}] index must be timezone-aware")
                index = frame.index.tz_convert("UTC")
                if index.has_duplicates or not index.is_monotonic_increasing:
                    raise ValueError(f"observations[{symbol}] index must be unique and increasing")
                values = frame.to_numpy(dtype=np.float32, copy=True)
                indices[symbol] = index
            else:
                values = np.asarray(raw_obs, dtype=np.float32)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                if values.ndim != 2:
                    raise ValueError(f"observations[{symbol}] must be 2-dimensional")
                if not np.isfinite(values).all():
                    raise ValueError(f"observations[{symbol}] contains non-finite values")
                indices[symbol] = None
            if len(values) == 0:
                raise ValueError(f"observations[{symbol}] must not be empty")
            arrays[symbol] = values
        return arrays, indices


    @staticmethod
    def _normalize_quotes(
        market_quotes: Mapping[str, Sequence[MarketQuote]],
    ) -> dict[str, tuple[MarketQuote, ...]]:
        result: dict[str, tuple[MarketQuote, ...]] = {}
        for raw_symbol, raw_quotes in market_quotes.items():
            symbol = str(raw_symbol).upper().strip()
            normalized: list[MarketQuote] = []
            previous = None
            for quote in raw_quotes:
                if not isinstance(quote, MarketQuote):
                    raise TypeError(f"market_quotes[{symbol}] contains non-MarketQuote value")
                if quote.symbol != symbol:
                    raise ValueError(f"quote symbol {quote.symbol!r} != mapping key {symbol!r}")
                if previous is not None and quote.timestamp <= previous:
                    raise ValueError(f"market_quotes[{symbol}] timestamps must be strictly increasing")
                normalized.append(quote)
                previous = quote.timestamp
            if not normalized:
                raise ValueError(f"market_quotes[{symbol}] must not be empty")
            result[symbol] = tuple(normalized)
        return result


    @staticmethod
    def _normalize_instruments(
        instruments: Mapping[str, InstrumentSpec],
    ) -> dict[str, InstrumentSpec]:
        result = {}
        for raw_symbol, instrument in instruments.items():
            symbol = str(raw_symbol).upper().strip()
            if not isinstance(instrument, InstrumentSpec):
                raise TypeError(f"instruments[{symbol}] must be InstrumentSpec")
            if instrument.symbol != symbol:
                raise ValueError(f"instrument symbol {instrument.symbol!r} != mapping key {symbol!r}")
            result[symbol] = instrument
        return result


    def _validate_timelines(self) -> int:
        lengths = {symbol: len(quotes) for symbol, quotes in self.quotes.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all symbols must have identical quote lengths: {lengths}")
        n_steps = next(iter(lengths.values()))
        if n_steps < 2:
            raise ValueError("environment requires at least 2 market steps")
        reference = tuple(q.timestamp for q in self.quotes[self.symbols[0]])
        for symbol in self.symbols[1:]:
            current = tuple(q.timestamp for q in self.quotes[symbol])
            if current != reference:
                raise ValueError("all symbols must share identical quote timestamps")
        for symbol, obs in self.observations.items():
            if len(obs) != n_steps:
                raise ValueError(f"observation length mismatch for {symbol}")
            index = self.observation_indices[symbol]
            if index is not None:
                quote_index = pd.DatetimeIndex([q.timestamp for q in self.quotes[symbol]])
                if not index.equals(quote_index):
                    raise ValueError(f"observation and quote timestamps must match exactly for {symbol}")
        return n_steps


    @classmethod
    def from_mtf_datasets(
        cls,
        *,
        observations: Mapping[str, ObservationLike],
        datasets: Mapping[str, object],
        instruments: Mapping[str, InstrumentSpec],
        config: EnvironmentConfig,
        quote_timeframes: Optional[Mapping[str, str] | str] = None,
        execution: Optional[ExecutionSimulator] = None,
        execution_costs: Optional[Mapping[str, ExecutionCost]] = None,
        margin_rates_to_account: Optional[Mapping[str, float]] = None,
    ) -> "TradingEnvironment":
        if not observations:
            raise ValueError("observations must not be empty")

        if not datasets:
            raise ValueError("datasets must not be empty")

        if not instruments:
            raise ValueError("instruments must not be empty")

        # --------------------------------------------------------------
        # Normalize and validate the canonical instrument catalog.
        # --------------------------------------------------------------
        canonical_instruments: dict[str, InstrumentSpec] = {}

        for raw_symbol, instrument in instruments.items():
            key = str(raw_symbol).upper().strip()

            if not key:
                raise ValueError("instrument catalog contains an empty symbol")

            if key in canonical_instruments:
                raise ValueError(
                    f"duplicate instrument symbol after normalization: {key}"
                )

            if not isinstance(instrument, InstrumentSpec):
                raise TypeError(
                    f"instruments[{key}] must be canonical InstrumentSpec"
                )

            if instrument.symbol != key:
                raise ValueError(
                    f"instrument symbol {instrument.symbol!r} "
                    f"does not match catalog key {key!r}"
                )

            canonical_instruments[key] = instrument

        observation_symbols = {
            str(symbol).upper().strip()
            for symbol in observations
        }

        if set(canonical_instruments) != observation_symbols:
            raise ValueError(
                "observations and instruments must contain exactly the same symbols"
            )

        # --------------------------------------------------------------
        # Normalize the dataset catalog without changing the datasets.
        # --------------------------------------------------------------
        datasets_by_symbol: dict[str, object] = {}

        for raw_symbol, dataset in datasets.items():
            key = str(raw_symbol).upper().strip()

            if not key:
                raise ValueError("dataset catalog contains an empty symbol")

            if key in datasets_by_symbol:
                raise ValueError(
                    f"duplicate dataset symbol after normalization: {key}"
                )

            datasets_by_symbol[key] = dataset

        if not observation_symbols.issubset(datasets_by_symbol):
            missing = sorted(
                observation_symbols - set(datasets_by_symbol)
            )
            raise KeyError(
                f"datasets missing required symbols: {missing}"
            )

        if isinstance(quote_timeframes, Mapping):
            quote_timeframes_by_symbol = {
                str(symbol).upper().strip(): timeframe
                for symbol, timeframe in quote_timeframes.items()
            }
        else:
            quote_timeframes_by_symbol = None

        quotes: dict[str, tuple[MarketQuote, ...]] = {}
        resolved_instruments: dict[str, InstrumentSpec] = {}

        for symbol, obs in observations.items():
            key = str(symbol).upper().strip()

            dataset = datasets_by_symbol[key]
            instrument = canonical_instruments[key]

            if quote_timeframes_by_symbol is not None:
                quote_tf = quote_timeframes_by_symbol.get(key)
            else:
                quote_tf = quote_timeframes

            index = cls._extract_observation_index(obs)

            if index is None:
                raise ValueError(
                    "from_mtf_datasets requires pandas observation DataFrames "
                    "so the candle observation clock is explicit"
                )

            resolver = HistoricalQuoteResolver(
                dataset=dataset,
                instrument=instrument,
                quote_timeframe=quote_tf,
            )

            bundle = resolver.resolve(index)

            quotes[key] = bundle.quotes
            resolved_instruments[key] = bundle.instrument

        return cls(
            observations=observations,
            market_quotes=quotes,
            instruments=resolved_instruments,
            config=config,
            execution=execution,
            execution_costs=execution_costs,
            margin_rates_to_account=margin_rates_to_account,
        )


    @staticmethod
    def _extract_observation_index(observation: ObservationLike) -> pd.DatetimeIndex | None:
        if not isinstance(observation, pd.DataFrame):
            return None
        if not isinstance(observation.index, pd.DatetimeIndex):
            raise TypeError("observation DataFrame index must be DatetimeIndex")
        return observation.index


    def _current_observation(self) -> np.ndarray:
        arrays = [self.observations[symbol][self._t].reshape(-1) for symbol in self.symbols]
        return np.concatenate(arrays).astype(np.float32, copy=False)


    def _current_quotes(self) -> dict[str, MarketQuote]:
        return {symbol: self.quotes[symbol][self._t] for symbol in self.symbols}


    def _current_mark_quotes(self) -> dict[str, MarketQuote]:
        result: dict[str, MarketQuote] = {}
        for symbol in self.symbols:
            result[symbol] = self.quotes[symbol][self._t]
        return result


    def _calculate_unrealized(self) -> float:
        return self.execution.mark_to_market(
            positions=self.portfolio.positions,
            quotes=self._current_mark_quotes(),
            instruments=self.instruments,
        )


    def _calculate_used_margin(self) -> float:
        total = 0.0
        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)
            if position.side == 0 or position.lots <= 0.0:
                continue
            instrument = self.instruments[symbol]
            quote = self.quotes[symbol][self._t]
            conversion = self.margin_rates_to_account.get(symbol, 1.0)
            contract_size = float(instrument.contract_size)
            total += (
                abs(quote.mid)
                * position.lots
                * contract_size
                * conversion
                / self.config.leverage
            )
        return float(total)


    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError("seed must be >= 0")
            # Reserved deterministic seed surface; f04 itself has no stochastic model.
            np.random.default_rng(seed)
        self.portfolio.reset()
        self._t = 0
        self._terminated = False
        self._truncated = False
        self._episode_index += 1
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
        if self._terminated or self._truncated:
            raise RuntimeError("Environment is done. Call reset() before step().")
        if not isinstance(action, PortfolioAction):
            raise TypeError("action must be PortfolioAction")

        unknown = {intent.symbol for intent in action.intents} - set(self.symbols)
        if unknown:
            raise ValueError(f"Action contains unknown symbols: {sorted(unknown)}")

        equity_before = float(self.portfolio.equity)
        intents = {symbol: None for symbol in self.symbols}
        for intent in action.intents:
            intents[intent.symbol] = intent

        realized_delta = 0.0
        execution_info: dict[str, dict] = {}
        quotes_now = self._current_quotes()

        for symbol in self.symbols:
            intent = intents[symbol]
            if intent is None:
                continue
            result = self.execution.execute(
                position=self.portfolio.get_position(symbol),
                intent=intent,
                quote=quotes_now[symbol],
                instrument=self.instruments[symbol],
                cost=self.execution_costs.get(symbol),
            )
            realized_delta += float(result["realized_pnl"])
            execution_info[symbol] = result

        next_t = self._t + 1
        natural_termination = next_t >= self.n_steps - 1
        truncated = False
        if self.config.max_episode_steps is not None:
            truncated = (
                self.portfolio.step_count + 1 >= self.config.max_episode_steps
                and not natural_termination
            )

        self._t = min(next_t, self.n_steps - 1)
        unrealized_total = self._calculate_unrealized()
        used_margin = self._calculate_used_margin()
        self.portfolio.update_accounting(
            realized_delta=realized_delta,
            unrealized_total=unrealized_total,
            used_margin=used_margin,
        )
        self.portfolio.advance_step()

        termination_reason = None
        risk_terminated = False
        if self.config.max_drawdown_pct is not None and self.portfolio.total_drawdown >= self.config.max_drawdown_pct:
            risk_terminated = True
            termination_reason = "max_drawdown"
        elif (
            self.config.stop_out_level is not None
            and self.portfolio.margin_level is not None
            and self.portfolio.margin_level <= self.config.stop_out_level
        ):
            risk_terminated = True
            termination_reason = "margin_stop_out"

        self._terminated = bool(natural_termination or risk_terminated)
        self._truncated = bool(truncated)
        reward = float(self.portfolio.equity - equity_before)

        info = {
            "step": self.portfolio.step_count,
            "t": self._t,
            "symbols": self.symbols,
            "timestamp": self.quotes[self.symbols[0]][self._t].timestamp,
            "balance": self.portfolio.balance,
            "equity": self.portfolio.equity,
            "used_margin": self.portfolio.used_margin,
            "free_margin": self.portfolio.free_margin,
            "realized_pnl": self.portfolio.realized_pnl,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "drawdown": self.portfolio.total_drawdown,
            "daily_drawdown": self.portfolio.daily_drawdown,
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

