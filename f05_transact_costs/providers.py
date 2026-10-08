from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from .contracts import InstrumentSpec, MarketQuote


class HistoricalQuoteProvider(Protocol):
    def quote_at(self, *, symbol: str, timestamp: datetime) -> MarketQuote: ...


class BrokerQuoteProvider(Protocol):
    def quote(self, *, symbol: str) -> MarketQuote: ...


class InstrumentResolver(Protocol):
    def resolve(self, *, symbol: str) -> InstrumentSpec: ...


class CurrencyConversionProvider(Protocol):
    def rate_to_account(self, *, from_currency: str, account_currency: str) -> float: ...


class MappingInstrumentResolver:
    """Resolver over an already-loaded symbol-spec mapping; it never reads files."""

    def __init__(self, symbol_specs: Mapping[str, Mapping[str, object]], *, default_tick_value_currency: str | None = None) -> None:
        self._specs = {str(k).strip().upper(): dict(v) for k, v in symbol_specs.items()}
        self._default_currency = None if default_tick_value_currency is None else str(default_tick_value_currency).strip().upper()

    def resolve(self, *, symbol: str) -> InstrumentSpec:
        key = str(symbol).strip().upper()
        if key not in self._specs:
            raise KeyError(f"unknown symbol specification: {key}")
        raw = self._specs[key]
        digits = int(raw.get("digits", 0))
        tick_size = float(raw["trade_tick_size"])
        pip_size = float(raw.get("pip_size", tick_size * (10.0 if digits in (3, 5) else 1.0)))
        currency = raw.get("tick_value_currency", self._default_currency)
        if currency is None:
            raise KeyError(f"missing tick_value_currency for {key}")
        return InstrumentSpec(
            symbol=key,
            pip_size=pip_size,
            tick_size=tick_size,
            tick_value=float(raw["trade_tick_value"]),
            tick_value_currency=str(currency),
            contract_size=(None if raw.get("contract_size") is None else float(raw["contract_size"])),
            volume_min=(None if raw.get("volume_min") is None else float(raw["volume_min"])),
            volume_step=(None if raw.get("volume_step") is None else float(raw["volume_step"])),
            volume_max=(None if raw.get("volume_max") is None else float(raw["volume_max"])),
        )
