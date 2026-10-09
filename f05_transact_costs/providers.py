from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from f03_data.instrument_specs import InstrumentSpec
from .contracts import MarketQuote


class HistoricalQuoteProvider(Protocol):
    def quote_at(self, *, symbol: str, timestamp: datetime) -> MarketQuote: ...


class BrokerQuoteProvider(Protocol):
    def quote(self, *, symbol: str) -> MarketQuote: ...


class InstrumentResolver(Protocol):
    def resolve(self, *, symbol: str) -> InstrumentSpec: ...


class CurrencyConversionProvider(Protocol):
    def rate_to_account(self, *, from_currency: str, account_currency: str) -> float: ...


class MappingInstrumentResolver:
    """
    Resolve an already-normalized canonical instrument catalog.

    This resolver does not:
        - derive pip_size or tick_size;
        - infer currencies;
        - read configuration or snapshot files;
        - perform MT5 or network I/O.

    All instrument normalization belongs to f03_data.
    """

    def __init__(
        self,
        instruments: Mapping[str, InstrumentSpec],
    ) -> None:
        if not isinstance(instruments, Mapping):
            raise TypeError("instruments must be a mapping")

        resolved: dict[str, InstrumentSpec] = {}

        for raw_symbol, instrument in instruments.items():
            symbol = str(raw_symbol).strip().upper()

            if not symbol:
                raise ValueError("instrument catalog contains an empty symbol")

            if symbol in resolved:
                raise ValueError(
                    f"duplicate symbol after normalization: {symbol}"
                )

            if not isinstance(instrument, InstrumentSpec):
                raise TypeError(
                    f"instruments[{symbol}] must be canonical InstrumentSpec"
                )

            if instrument.symbol != symbol:
                raise ValueError(
                    f"instrument symbol {instrument.symbol!r} "
                    f"does not match catalog key {symbol!r}"
                )

            resolved[symbol] = instrument

        self._instruments = resolved

    def resolve(self, *, symbol: str) -> InstrumentSpec:
        key = str(symbol).strip().upper()

        if not key:
            raise ValueError("symbol is required")

        try:
            return self._instruments[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown canonical instrument: {key}"
            ) from exc
