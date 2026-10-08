from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd

from f03_data.mtf_dataset import MTFDataset
from f05_transact_costs.contracts import CostSource, InstrumentSpec, MarketQuote
from f05_transact_costs.providers import MappingInstrumentResolver


@dataclass(frozen=True, slots=True)
class HistoricalQuoteBundle:
    quotes: tuple[MarketQuote, ...]
    instrument: InstrumentSpec


class HistoricalQuoteResolver:
    """Resolve candle-based executable quotes from frozen f02 MTFDataset data.

    - OHLC is treated as BID OHLC (project contract).
    - historical ``spread`` is interpreted as MT5 points.
    - Ask = bid + spread_points * point.
    - no resampling, interpolation, broker I/O, or hidden cache is performed.
    """

    def __init__(
        self,
        *,
        dataset: MTFDataset,
        symbol_spec: Mapping[str, object],
        quote_timeframe: str | None = None,
    ) -> None:
        if not isinstance(dataset, MTFDataset):
            raise TypeError("dataset must be MTFDataset")
        self.dataset = dataset
        self.symbol = str(dataset.symbol).upper().strip()
        self.quote_timeframe = str(quote_timeframe or dataset.base_tf).upper().strip()
        if not self.quote_timeframe:
            raise ValueError("quote_timeframe is required")

        raw = dict(symbol_spec)
        self.point = self._resolve_point(raw)
        normalized = dict(raw)
        if "tick_value_currency" not in normalized:
            currency = normalized.get("currency_profit") or normalized.get("currency_base")
            if currency is not None:
                normalized["tick_value_currency"] = currency
        self.instrument = MappingInstrumentResolver({self.symbol: normalized}).resolve(symbol=self.symbol)

    @staticmethod
    def _resolve_point(raw: Mapping[str, object]) -> float:
        point = raw.get("point")
        if point is None:
            raise KeyError("historical symbol spec requires point")
        point = float(point)
        if not np.isfinite(point) or point <= 0.0:
            raise ValueError("symbol point must be positive and finite")
        return point

    @staticmethod
    def _to_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("observation_index must be pandas.DatetimeIndex")
        if index.tz is None:
            raise ValueError("observation_index must be timezone-aware")
        out = index.tz_convert("UTC")
        if not out.is_monotonic_increasing or out.has_duplicates:
            raise ValueError("observation_index must be unique and increasing")
        return out

    def resolve(self, observation_index: pd.DatetimeIndex) -> HistoricalQuoteBundle:
        obs_index = self._to_utc_index(observation_index)
        frame = self.dataset.get(self.quote_timeframe)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"dataset[{self.quote_timeframe}] must be DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("historical market frame index must be DatetimeIndex")
        if frame.index.tz is None:
            raise ValueError("historical market frame index must be timezone-aware")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("historical market frame index must be unique and increasing")

        frame = frame.copy()
        frame.index = frame.index.tz_convert("UTC")
        required = {"close", "spread"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"historical market frame is missing columns: {missing}")

        selected = frame.reindex(obs_index)
        if selected[["close", "spread"]].isna().any().any():
            missing_times = selected.index[
                selected[["close", "spread"]].isna().any(axis=1)
            ]
            raise ValueError(
                f"No exact closed-candle quote exists for {self.symbol}/{self.quote_timeframe} "
                f"at {missing_times[0].isoformat()}"
            )

        quotes: list[MarketQuote] = []
        for timestamp, row in selected.iterrows():
            bid = float(row["close"])
            spread_points = float(row["spread"])
            if not np.isfinite(bid) or bid <= 0.0:
                raise ValueError(f"invalid bid at {timestamp}")
            if not np.isfinite(spread_points) or spread_points < 0.0:
                raise ValueError(f"invalid spread at {timestamp}")
            ask = bid + spread_points * self.point
            quotes.append(
                MarketQuote(
                    symbol=self.symbol,
                    bid=bid,
                    ask=ask,
                    timestamp=timestamp.to_pydatetime(),
                    source=CostSource.SIMULATION,
                )
            )
        return HistoricalQuoteBundle(tuple(quotes), self.instrument)
