# f03_data/instrument_specs.py
from __future__ import annotations

"""
Canonical Instrument Contract for Bot-RL-3.

Ownership:
    f03_data owns instrument discovery and normalization.

Purpose:
    Convert the already-discovered MT5 symbol specification snapshot into
    one immutable, provider-neutral InstrumentSpec.

Important boundary rules:
    - This module performs NO MT5 I/O.
    - This module performs NO YAML/file I/O.
    - This module does NOT read project config files.
    - This module does NOT depend on f05_transact_costs or f08_risk.
    - The caller supplies an already-resolved f03 snapshot.
    - f05/f08 must consume this contract rather than re-derive instrument data.

Source authority:
    MT5Connector.get_symbol_specs()
        -> f03_data snapshot
        -> this canonical normalization
        -> downstream consumers
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping


# ============================================================================
# Constants
# ============================================================================

# MQL5 ENUM_SYMBOL_CALC_MODE
SYMBOL_CALC_MODE_FOREX = 0
SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE = 5

_FOREX_CALC_MODES = frozenset(
    {
        SYMBOL_CALC_MODE_FOREX,
        SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE,
    }
)


# ============================================================================
# Validation helpers
# ============================================================================

def _symbol(value: Any) -> str:
    result = str(value).strip().upper()
    if not result:
        raise ValueError("symbol is required")
    return result


def _text(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _currency(value: Any, name: str) -> str:
    result = _text(value, name).upper()
    if len(result) != 3 or not result.isalpha():
        raise ValueError(f"{name} must be a 3-letter currency code")
    return result


def _finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    if minimum is not None:
        if strict and result <= minimum:
            raise ValueError(f"{name} must be > {minimum}")
        if not strict and result < minimum:
            raise ValueError(f"{name} must be >= {minimum}")

    return result


def _optional_finite(
    value: Any,
    name: str,
    *,
    minimum: float = 0.0,
    strict: bool = False,
) -> float | None:
    if value is None:
        return None

    return _finite(
        value,
        name,
        minimum=minimum,
        strict=strict,
    )


def _non_negative_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc

    if result < 0:
        raise ValueError(f"{name} must be >= 0")

    return result


def _positive_int(value: Any, name: str) -> int:
    result = _non_negative_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be > 0")
    return result


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name)


def _timestamp(value: Any, name: str) -> datetime:
    if isinstance(value, str):
        raw = value.strip()

        if not raw:
            raise ValueError(f"{name} must not be empty")

        if raw.endswith("Z"):
            candidate = raw[:-1]

            # Handle both normal UTC-Z timestamps and the existing
            # legacy representation: 2025-12-12T16:25:07+00:00Z
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                parsed = None

            if (
                parsed is not None
                and parsed.tzinfo is not None
                and parsed.utcoffset() is not None
            ):
                value = parsed
            else:
                try:
                    value = datetime.fromisoformat(
                        candidate + "+00:00"
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{name} must be a valid ISO datetime"
                    ) from exc
        else:
            try:
                value = datetime.fromisoformat(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be a valid ISO datetime"
                ) from exc

    if not isinstance(value, datetime):
        raise TypeError(
            f"{name} must be datetime or ISO datetime string"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


# ============================================================================
# Canonical Instrument
# ============================================================================

@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """
    Canonical immutable instrument specification.

    The first five positional fields intentionally remain compatible with the
    current f05 InstrumentSpec constructor:

        InstrumentSpec(
            symbol,
            pip_size,
            tick_size,
            tick_value,
            tick_value_currency,
            ...
        )

    This allows f05 migration to occur without forcing unrelated changes in
    downstream callers.

    Semantics:
        symbol:
            Broker/platform symbol identifier.

        pip_size:
            Project-normalized execution/risk unit.
            It is NOT always a native broker "pip".

        tick_size:
            Minimal price change from MT5.

        tick_value:
            MT5 trade_tick_value.
            Per MQL5, this is equivalent to trade_tick_value_profit.

        tick_value_profit / tick_value_loss:
            Directional MT5 tick values when available.

        tick_value_currency:
            Currency in which tick_value is expressed.
            For the MT5 source, this is the account/deposit currency.

        source:
            Provenance identifier, normally "mt5".

        as_of:
            UTC timestamp of the source snapshot.

        account_currency:
            Account/deposit currency associated with the snapshot.
    """

    # ------------------------------------------------------------------
    # Compatibility core (kept first)
    # ------------------------------------------------------------------

    symbol: str
    pip_size: float
    tick_size: float
    tick_value: float
    tick_value_currency: str

    contract_size: float | None = None
    volume_min: float | None = None
    volume_step: float | None = None
    volume_max: float | None = None

    # ------------------------------------------------------------------
    # Extended canonical fields
    # ------------------------------------------------------------------

    digits: int | None = None
    point: float | None = None

    tick_value_profit: float | None = None
    tick_value_loss: float | None = None

    volume_limit: float | None = None

    currency_base: str | None = None
    currency_profit: str | None = None
    currency_margin: str | None = None

    trade_calc_mode: int | None = None
    trade_mode: int | None = None
    trade_execution_mode: int | None = None

    trade_stops_level: int | None = None
    trade_freeze_level: int | None = None

    margin_initial: float | None = None
    margin_maintenance: float | None = None
    margin_hedged: float | None = None

    account_currency: str | None = None
    source: str = "mt5"
    as_of: datetime | None = None

    def __post_init__(self) -> None:
        # --------------------------------------------------------------
        # Identity
        # --------------------------------------------------------------
        symbol = _symbol(self.symbol)

        # --------------------------------------------------------------
        # Primary price/valuation fields
        # --------------------------------------------------------------
        pip_size = _finite(
            self.pip_size,
            "pip_size",
            minimum=0.0,
            strict=True,
        )

        tick_size = _finite(
            self.tick_size,
            "tick_size",
            minimum=0.0,
            strict=True,
        )

        tick_value = _finite(
            self.tick_value,
            "tick_value",
            minimum=0.0,
            strict=True,
        )

        tick_value_currency = _currency(
            self.tick_value_currency,
            "tick_value_currency",
        )

        # --------------------------------------------------------------
        # Optional numeric fields
        # --------------------------------------------------------------
        normalized: dict[str, Any] = {}

        for name, value in (
            ("contract_size", self.contract_size),
            ("volume_min", self.volume_min),
            ("volume_step", self.volume_step),
            ("volume_max", self.volume_max),
            ("volume_limit", self.volume_limit),
            ("point", self.point),
            ("tick_value_profit", self.tick_value_profit),
            ("tick_value_loss", self.tick_value_loss),
            ("margin_initial", self.margin_initial),
            ("margin_maintenance", self.margin_maintenance),
            ("margin_hedged", self.margin_hedged),
        ):
            if value is None:
                normalized[name] = None
                continue

            strict = name in {
                "contract_size",
                "volume_min",
                "volume_step",
                "volume_max",
                "point",
                "tick_value_profit",
                "tick_value_loss",
            }

            normalized[name] = _finite(
                value,
                name,
                minimum=0.0,
                strict=strict,
            )

        # --------------------------------------------------------------
        # Optional integer fields
        # --------------------------------------------------------------
        digits = (
            None
            if self.digits is None
            else _non_negative_int(self.digits, "digits")
        )

        trade_calc_mode = (
            None
            if self.trade_calc_mode is None
            else _non_negative_int(
                self.trade_calc_mode,
                "trade_calc_mode",
            )
        )

        trade_mode = (
            None
            if self.trade_mode is None
            else _non_negative_int(
                self.trade_mode,
                "trade_mode",
            )
        )

        trade_execution_mode = (
            None
            if self.trade_execution_mode is None
            else _non_negative_int(
                self.trade_execution_mode,
                "trade_execution_mode",
            )
        )

        trade_stops_level = (
            None
            if self.trade_stops_level is None
            else _non_negative_int(
                self.trade_stops_level,
                "trade_stops_level",
            )
        )

        trade_freeze_level = (
            None
            if self.trade_freeze_level is None
            else _non_negative_int(
                self.trade_freeze_level,
                "trade_freeze_level",
            )
        )

        # --------------------------------------------------------------
        # Volume constraints
        # --------------------------------------------------------------
        volume_min = normalized["volume_min"]
        volume_step = normalized["volume_step"]
        volume_max = normalized["volume_max"]
        volume_limit = normalized["volume_limit"]

        if volume_min is not None and volume_max is not None:
            if volume_min > volume_max:
                raise ValueError(
                    "volume_min must be <= volume_max"
                )

        if volume_step is not None and volume_max is not None:
            if volume_step > volume_max:
                raise ValueError(
                    "volume_step must be <= volume_max"
                )

        if volume_limit is not None and volume_max is not None:
            # volume_limit is an aggregate-side constraint in MT5.
            # It is not necessarily a single-deal limit.
            if volume_limit < 0.0:
                raise ValueError("volume_limit must be >= 0")

        # --------------------------------------------------------------
        # Currencies
        # --------------------------------------------------------------
        currency_base = (
            None
            if self.currency_base is None
            else _currency(self.currency_base, "currency_base")
        )

        currency_profit = (
            None
            if self.currency_profit is None
            else _currency(self.currency_profit, "currency_profit")
        )

        currency_margin = (
            None
            if self.currency_margin is None
            else _currency(self.currency_margin, "currency_margin")
        )

        account_currency = (
            None
            if self.account_currency is None
            else _currency(self.account_currency, "account_currency")
        )

        # --------------------------------------------------------------
        # Provenance
        # --------------------------------------------------------------
        source = _text(self.source, "source")

        as_of = (
            None
            if self.as_of is None
            else _timestamp(self.as_of, "as_of")
        )

        # --------------------------------------------------------------
        # MT5 valuation invariants
        # --------------------------------------------------------------
        tick_value_profit = normalized["tick_value_profit"]
        tick_value_loss = normalized["tick_value_loss"]

        if tick_value_profit is not None and tick_value_loss is not None:
            if tick_value_profit <= 0.0 or tick_value_loss <= 0.0:
                raise ValueError(
                    "tick_value_profit and tick_value_loss must be > 0"
                )

        # --------------------------------------------------------------
        # Normalize / freeze
        # --------------------------------------------------------------
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "pip_size", pip_size)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "tick_value", tick_value)
        object.__setattr__(
            self,
            "tick_value_currency",
            tick_value_currency,
        )

        object.__setattr__(
            self,
            "contract_size",
            normalized["contract_size"],
        )
        object.__setattr__(
            self,
            "volume_min",
            volume_min,
        )
        object.__setattr__(
            self,
            "volume_step",
            volume_step,
        )
        object.__setattr__(
            self,
            "volume_max",
            volume_max,
        )
        object.__setattr__(
            self,
            "volume_limit",
            volume_limit,
        )

        object.__setattr__(self, "digits", digits)
        object.__setattr__(self, "point", normalized["point"])

        object.__setattr__(
            self,
            "tick_value_profit",
            tick_value_profit,
        )
        object.__setattr__(
            self,
            "tick_value_loss",
            tick_value_loss,
        )

        object.__setattr__(
            self,
            "currency_base",
            currency_base,
        )
        object.__setattr__(
            self,
            "currency_profit",
            currency_profit,
        )
        object.__setattr__(
            self,
            "currency_margin",
            currency_margin,
        )

        object.__setattr__(
            self,
            "trade_calc_mode",
            trade_calc_mode,
        )
        object.__setattr__(self, "trade_mode", trade_mode)
        object.__setattr__(
            self,
            "trade_execution_mode",
            trade_execution_mode,
        )

        object.__setattr__(
            self,
            "trade_stops_level",
            trade_stops_level,
        )
        object.__setattr__(
            self,
            "trade_freeze_level",
            trade_freeze_level,
        )

        object.__setattr__(
            self,
            "margin_initial",
            normalized["margin_initial"],
        )
        object.__setattr__(
            self,
            "margin_maintenance",
            normalized["margin_maintenance"],
        )
        object.__setattr__(
            self,
            "margin_hedged",
            normalized["margin_hedged"],
        )

        object.__setattr__(
            self,
            "account_currency",
            account_currency,
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "as_of", as_of)

    # ==================================================================
    # Derived valuation properties
    # ==================================================================

    @property
    def value_per_price_unit_per_lot(self) -> float:
        """
        Backward-compatible value used by the current f05 calculator.
        """
        return self.tick_value / self.tick_size

    @property
    def value_per_price_unit_per_lot_profit(self) -> float:
        tick_value = (
            self.tick_value_profit
            if self.tick_value_profit is not None
            else self.tick_value
        )
        return tick_value / self.tick_size

    @property
    def value_per_price_unit_per_lot_loss(self) -> float:
        tick_value = (
            self.tick_value_loss
            if self.tick_value_loss is not None
            else self.tick_value
        )
        return tick_value / self.tick_size

    @property
    def pip_value_per_lot(self) -> float:
        """
        Backward-compatible project-normalized pip value.

        This is a derived project unit, not a native MT5 field.
        """
        return self.value_per_price_unit_per_lot * self.pip_size

    @property
    def pip_value_per_lot_profit(self) -> float:
        return (
            self.value_per_price_unit_per_lot_profit
            * self.pip_size
        )

    @property
    def pip_value_per_lot_loss(self) -> float:
        return (
            self.value_per_price_unit_per_lot_loss
            * self.pip_size
        )

    @property
    def stop_distance_price(self) -> float:
        """
        Broker minimum stop distance expressed in price units.
        """
        if self.trade_stops_level is None:
            return 0.0

        if self.point is None:
            raise ValueError(
                "point is required to derive stop_distance_price"
            )

        return self.trade_stops_level * self.point

    @property
    def freeze_distance_price(self) -> float:
        """
        Broker freeze distance expressed in price units.
        """
        if self.trade_freeze_level is None:
            return 0.0

        if self.point is None:
            raise ValueError(
                "point is required to derive freeze_distance_price"
            )

        return self.trade_freeze_level * self.point

    @property
    def is_forex(self) -> bool:
        return self.trade_calc_mode in _FOREX_CALC_MODES

    # ==================================================================
    # MT5 normalization
    # ==================================================================

    @classmethod
    def from_mt5_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        symbol: str,
    ) -> "InstrumentSpec":
        """
        Resolve exactly one canonical InstrumentSpec from an already-loaded
        f03 MT5 snapshot.

        This method deliberately does not:
            - read YAML
            - call MT5
            - read project config
            - invent missing instrument values
        """

        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot must be a mapping")

        meta = snapshot.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("snapshot.meta is required")

        raw_specs = snapshot.get("symbol_specs")
        if not isinstance(raw_specs, Mapping):
            raise ValueError("snapshot.symbol_specs is required")

        key = _symbol(symbol)

        if key not in raw_specs:
            raise KeyError(
                f"instrument specification not found in f03 snapshot: {key}"
            )

        record = raw_specs[key]

        if not isinstance(record, Mapping):
            raise TypeError(
                f"snapshot.symbol_specs[{key}] must be a mapping"
            )

        account_currency = meta.get("account_currency")
        if account_currency is None:
            raise ValueError(
                "snapshot.meta.account_currency is required"
            )

        as_of = meta.get("as_of")
        if as_of is None:
            raise ValueError(
                "snapshot.meta.as_of is required"
            )

        # --------------------------------------------------------------
        # Required native MT5 fields
        # --------------------------------------------------------------
        required = (
            "digits",
            "point",
            "trade_tick_value",
            "trade_tick_size",
            "contract_size",
            "volume_min",
            "volume_step",
            "volume_max",
        )

        missing = [
            name
            for name in required
            if record.get(name) is None
        ]

        if missing:
            raise ValueError(
                f"{key}: missing required MT5 instrument fields: "
                + ", ".join(missing)
            )

        digits = _non_negative_int(record["digits"], f"{key}.digits")
        point = _finite(
            record["point"],
            f"{key}.point",
            minimum=0.0,
            strict=True,
        )

        tick_size = _finite(
            record["trade_tick_size"],
            f"{key}.trade_tick_size",
            minimum=0.0,
            strict=True,
        )

        tick_value = _finite(
            record["trade_tick_value"],
            f"{key}.trade_tick_value",
            minimum=0.0,
            strict=True,
        )

        # --------------------------------------------------------------
        # Raw provider record
        # --------------------------------------------------------------
        raw = record.get("raw")
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError(
                f"{key}.raw must be a mapping when present"
            )

        raw_map: Mapping[str, Any] = (
            raw if isinstance(raw, Mapping) else {}
        )

        # --------------------------------------------------------------
        # Pull native detailed fields from raw when available.
        #
        # No fallback is invented for directional tick values:
        # if the source does not expose one, it remains None.
        # --------------------------------------------------------------
        tick_value_profit = raw_map.get(
            "trade_tick_value_profit"
        )

        tick_value_loss = raw_map.get(
            "trade_tick_value_loss"
        )

        currency_base = raw_map.get("currency_base")
        currency_profit = raw_map.get("currency_profit")
        currency_margin = raw_map.get("currency_margin")

        trade_calc_mode = raw_map.get("trade_calc_mode")
        trade_mode = raw_map.get("trade_mode")
        trade_execution_mode = raw_map.get("trade_exemode")

        trade_stops_level = raw_map.get(
            "trade_stops_level",
            record.get("stops_level"),
        )

        trade_freeze_level = raw_map.get(
            "trade_freeze_level"
        )

        volume_limit = raw_map.get("volume_limit")

        margin_initial = raw_map.get("margin_initial")
        margin_maintenance = raw_map.get(
            "margin_maintenance"
        )
        margin_hedged = raw_map.get("margin_hedged")

        # --------------------------------------------------------------
        # Project-normalized pip unit.
        #
        # MT5 provides point/digits but not a universal "pip".
        # Forex gets conventional FX pip normalization.
        # Non-Forex uses one MT5 point as the project cost unit.
        # --------------------------------------------------------------
        if trade_calc_mode in _FOREX_CALC_MODES:
            pip_size = (
                10.0 * point
                if digits in (3, 5)
                else point
            )
        else:
            pip_size = point

        # --------------------------------------------------------------
        # MT5 trade_tick_value is expressed in account/deposit currency.
        #
        # Therefore tick_value_currency is deliberately NOT taken from
        # currency_profit. For example, USDJPY may have:
        #     currency_profit = JPY
        #     account_currency = USD
        #     tick_value_currency = USD
        # --------------------------------------------------------------
        tick_value_currency = _currency(
            account_currency,
            f"{key}.tick_value_currency",
        )

        return cls(
            symbol=key,
            pip_size=pip_size,
            tick_size=tick_size,
            tick_value=tick_value,
            tick_value_currency=tick_value_currency,

            contract_size=_optional_finite(
                record.get("contract_size"),
                f"{key}.contract_size",
                strict=True,
            ),
            volume_min=_optional_finite(
                record.get("volume_min"),
                f"{key}.volume_min",
                strict=True,
            ),
            volume_step=_optional_finite(
                record.get("volume_step"),
                f"{key}.volume_step",
                strict=True,
            ),
            volume_max=_optional_finite(
                record.get("volume_max"),
                f"{key}.volume_max",
                strict=True,
            ),

            digits=digits,
            point=point,

            tick_value_profit=(
                None
                if tick_value_profit is None
                else _finite(
                    tick_value_profit,
                    f"{key}.tick_value_profit",
                    minimum=0.0,
                    strict=True,
                )
            ),
            tick_value_loss=(
                None
                if tick_value_loss is None
                else _finite(
                    tick_value_loss,
                    f"{key}.tick_value_loss",
                    minimum=0.0,
                    strict=True,
                )
            ),

            volume_limit=_optional_finite(
                volume_limit,
                f"{key}.volume_limit",
            ),

            currency_base=(
                None
                if currency_base is None
                else _currency(
                    currency_base,
                    f"{key}.currency_base",
                )
            ),
            currency_profit=(
                None
                if currency_profit is None
                else _currency(
                    currency_profit,
                    f"{key}.currency_profit",
                )
            ),
            currency_margin=(
                None
                if currency_margin is None
                else _currency(
                    currency_margin,
                    f"{key}.currency_margin",
                )
            ),

            trade_calc_mode=(
                None
                if trade_calc_mode is None
                else _non_negative_int(
                    trade_calc_mode,
                    f"{key}.trade_calc_mode",
                )
            ),
            trade_mode=(
                None
                if trade_mode is None
                else _non_negative_int(
                    trade_mode,
                    f"{key}.trade_mode",
                )
            ),
            trade_execution_mode=(
                None
                if trade_execution_mode is None
                else _non_negative_int(
                    trade_execution_mode,
                    f"{key}.trade_execution_mode",
                )
            ),

            trade_stops_level=(
                None
                if trade_stops_level is None
                else _non_negative_int(
                    trade_stops_level,
                    f"{key}.trade_stops_level",
                )
            ),
            trade_freeze_level=(
                None
                if trade_freeze_level is None
                else _non_negative_int(
                    trade_freeze_level,
                    f"{key}.trade_freeze_level",
                )
            ),

            margin_initial=_optional_finite(
                margin_initial,
                f"{key}.margin_initial",
            ),
            margin_maintenance=_optional_finite(
                margin_maintenance,
                f"{key}.margin_maintenance",
            ),
            margin_hedged=_optional_finite(
                margin_hedged,
                f"{key}.margin_hedged",
            ),

            account_currency=_currency(
                account_currency,
                f"{key}.account_currency",
            ),
            source="mt5",
            as_of=as_of,
        )


# ============================================================================
# Catalog resolver
# ============================================================================

def resolve_instrument_specs(
    snapshot: Mapping[str, Any],
    symbols: list[str] | tuple[str, ...] | None = None,
) -> dict[str, InstrumentSpec]:
    """
    Resolve a canonical instrument catalog from an already-loaded f03 snapshot.

    Resolution policy:
        - When symbols is None, resolve every symbol present in the snapshot.
        - When symbols is supplied, every requested symbol must exist.
        - No symbol is silently skipped.
        - No missing field is silently defaulted.
    """

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    raw_specs = snapshot.get("symbol_specs")
    if not isinstance(raw_specs, Mapping):
        raise ValueError("snapshot.symbol_specs is required")

    available = {
        _symbol(name)
        for name in raw_specs.keys()
    }

    requested = (
        sorted(available)
        if symbols is None
        else [_symbol(symbol) for symbol in symbols]
    )

    duplicates = {
        symbol
        for symbol in requested
        if requested.count(symbol) > 1
    }

    if duplicates:
        raise ValueError(
            "duplicate symbols requested: "
            + ", ".join(sorted(duplicates))
        )

    missing = [
        symbol
        for symbol in requested
        if symbol not in available
    ]

    if missing:
        raise KeyError(
            "instrument specifications missing from f03 snapshot: "
            + ", ".join(sorted(missing))
        )

    result: dict[str, InstrumentSpec] = {}

    for symbol in requested:
        result[symbol] = InstrumentSpec.from_mt5_snapshot(
            snapshot,
            symbol=symbol,
        )

    return result


__all__ = [
    "InstrumentSpec",
    "resolve_instrument_specs",
    "SYMBOL_CALC_MODE_FOREX",
    "SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE",
]