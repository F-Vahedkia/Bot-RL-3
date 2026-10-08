# f08_risk/risk_context.py (7)
# =============================================================================
# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Purpose:
#   Immutable, execution-independent snapshot of account and symbol risk state.
#
# Design rule:
#   This module does NOT talk to MT5, Broker, Execution, or Portfolio accounting.
#   It only defines the normalized risk state consumed by the Risk Layer.
# =============================================================================
"""
نکات مهم این فایل که chatGPT نوشته است:
-----------------------------------------
    - AccountRiskSnapshot وضعیت نرمال‌شدهٔ حساب را نگه می‌دارد:
        balance، equity، margin، leverage، peak equity، day-start equity و تعداد کل پوزیشن‌های باز.
    - margin_utilization، drawdown و daily_drawdown به‌صورت property از همین snapshot محاسبه می‌شوند.
    - SymbolRiskSnapshot وضعیت فعلی هر نماد را نگه می‌دارد و اکنون position_count نیز دارد.
    - برای backward compatibility، وقتی position_count=None باشد،
        از روی current_lots مقدار 0 یا 1 ساخته می‌شود؛
        ولی اگر مقدار صریح داده شود باید integer نامنفی باشد.
    - RiskContext مرز بین state acquisition و RiskEngine است
        و عمداً مستقل از execution، broker و MT5 تعریف شده است.
    - نمادها normalize می‌شوند و mappingهای symbols و correlation به MappingProxyType تبدیل می‌شوند؛
        بنابراین snapshot عملاً immutable است.
    - correlation به‌صورت عددی در بازهٔ [-1, 1] اعتبارسنجی می‌شود.
    - total_current_exposure مجموع absolute exposureها و
        total_current_used_margin مجموع marginهای فعلی نمادها را برمی‌گرداند.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Optional


# =============================================================================
# Validation helpers
# =============================================================================

def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _require_non_negative(name: str, value: float) -> float:
    value = _require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return value


def _validate_mapping(
    name: str,
    values: Mapping[str, float],
) -> Mapping[str, float]:
    if values is None:
        raise ValueError(f"{name} is required.")

    normalized = {}
    for symbol, value in values.items():
        symbol = str(symbol).strip().upper()

        if not symbol:
            raise ValueError(f"{name} contains an empty symbol.")

        normalized[symbol] = _require_finite(
            f"{name}[{symbol}]",
            value,
        )

    return MappingProxyType(normalized)


# =============================================================================
# Account snapshot
# =============================================================================

@dataclass(frozen=True, slots=True)
class AccountRiskSnapshot:
    """
    Normalized account-level state used by the Risk Engine.
    All monetary values must already be expressed in the account's reporting currency.
    No broker-specific calculation is performed here.
    """
    balance: float
    equity: float
    used_margin: float
    free_margin: float
    margin_level: float
    leverage: float
    peak_equity: float
    day_start_equity: float
    open_position_count: int

    def __post_init__(self) -> None:
        balance = _require_non_negative("balance", self.balance)
        equity = _require_non_negative("equity", self.equity)
        used_margin = _require_non_negative("used_margin", self.used_margin)
        free_margin = _require_non_negative("free_margin", self.free_margin)
        margin_level = _require_finite("margin_level", self.margin_level)
        leverage = _require_finite("leverage", self.leverage)
        peak_equity = _require_non_negative("peak_equity", self.peak_equity)
        day_start_equity = _require_non_negative("day_start_equity", self.day_start_equity)

        if leverage <= 0.0:
            raise ValueError("leverage must be > 0.")

        if margin_level < 0.0:
            raise ValueError("margin_level must be >= 0.")

        if int(self.open_position_count) != self.open_position_count:
            raise ValueError("open_position_count must be an integer.")

        if self.open_position_count < 0:
            raise ValueError("open_position_count must be >= 0.")

        if free_margin > equity + 1e-12:
            raise ValueError("free_margin cannot exceed equity.")

        if used_margin > equity + 1e-12:
            raise ValueError("used_margin cannot exceed equity.")

        object.__setattr__(self, "balance", balance)
        object.__setattr__(self, "equity", equity)
        object.__setattr__(self, "used_margin", used_margin)
        object.__setattr__(self, "free_margin", free_margin)
        object.__setattr__(self, "margin_level", margin_level)
        object.__setattr__(self, "leverage", leverage)
        object.__setattr__(self, "peak_equity", peak_equity)
        object.__setattr__(self, "day_start_equity", day_start_equity)
        object.__setattr__(self, "open_position_count", int(self.open_position_count))

    @property
    def margin_utilization(self) -> float:
        """
        Used-margin / equity.
        Returns 0 when equity is zero.
        """
        if self.equity <= 0.0:
            return 0.0

        return self.used_margin / self.equity

    @property
    def drawdown(self) -> float:
        """
        Current equity drawdown from peak equity.
        """
        if self.peak_equity <= 0.0:
            return 0.0

        return max(
            0.0,
            1.0 - (self.equity / self.peak_equity),
        )

    @property
    def daily_drawdown(self) -> float:
        """
        Current equity loss relative to day-start equity.
        """
        if self.day_start_equity <= 0.0:
            return 0.0

        return max(
            0.0,
            1.0 - (self.equity / self.day_start_equity),
        )


# =============================================================================
# Symbol snapshot
# =============================================================================

@dataclass(frozen=True, slots=True)
class SymbolRiskSnapshot:
    """
    Normalized current risk state of one symbol.

    `exposure` is a normalized signed/unsigned value defined by the
    portfolio-risk convention. The Risk Engine does not infer its meaning.
    """
    symbol: str
    exposure: float
    notional: float
    used_margin: float
    current_lots: float
    current_side: int
    position_count: int | None = None

    def __post_init__(self) -> None:

        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol is required.")

        exposure = _require_finite("exposure", self.exposure)
        notional = _require_non_negative("notional", self.notional)
        used_margin = _require_non_negative("used_margin", self.used_margin)
        current_lots = _require_non_negative("current_lots", self.current_lots)

        if int(self.current_side) != self.current_side:
            raise ValueError("current_side must be an integer.")

        current_side = int(self.current_side)

        if current_side not in (-1, 0, 1):
            raise ValueError("current_side must be one of -1, 0, 1.")

        if current_lots == 0.0 and current_side != 0:
            raise ValueError("current_side must be 0 when current_lots is zero.")

        if current_lots > 0.0 and current_side == 0:
            raise ValueError("current_side cannot be 0 when current_lots > 0.")

        if self.position_count is None:
            position_count = 0 if current_lots == 0.0 else 1
        else:
            if (
                isinstance(self.position_count, bool)
                or not isinstance(self.position_count, int)
            ):
                raise TypeError("position_count must be a non-negative integer.")

            if self.position_count < 0:
                raise ValueError("position_count must be >= 0.")

            position_count = self.position_count

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "notional", notional)
        object.__setattr__(self, "used_margin", used_margin)
        object.__setattr__(self, "current_lots", current_lots)
        object.__setattr__(self, "current_side", current_side)
        object.__setattr__(self, "position_count", position_count)

# =============================================================================
# Portfolio risk context
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskContext:
    """
    Complete execution-independent risk snapshot.

    This is the boundary object between account/position state acquisition
    and the deterministic Risk Engine.
    """

    timestamp: object
    account: AccountRiskSnapshot
    symbols: Mapping[str, SymbolRiskSnapshot]
    correlation: Mapping[str, Mapping[str, float]]
    risk_blocked: bool = False

    def __post_init__(self) -> None:
        if self.timestamp is None:
            raise ValueError("timestamp is required.")

        if self.account is None:
            raise ValueError("account is required.")

        if not isinstance(self.account, AccountRiskSnapshot):
            raise TypeError("account must be AccountRiskSnapshot.")

        if self.symbols is None:
            raise ValueError("symbols is required.")

        normalized_symbols = {}
        for symbol, snapshot in self.symbols.items():
            symbol = str(symbol).strip().upper()

            if not symbol:
                raise ValueError("symbols contains an empty symbol.")

            if not isinstance(snapshot, SymbolRiskSnapshot):
                raise TypeError(f"symbols[{symbol}] must be SymbolRiskSnapshot.")

            if snapshot.symbol != symbol:
                raise ValueError(
                    f"symbols[{symbol}] does not match "
                    f"snapshot.symbol={snapshot.symbol}."
                )
            normalized_symbols[symbol] = snapshot


        normalized_correlation = {}
        if self.correlation is None:
            raise ValueError("correlation is required.")

        for symbol_a, row in self.correlation.items():
            symbol_a = str(symbol_a).strip().upper()
            if not symbol_a:
                raise ValueError("correlation contains an empty symbol.")

            if not isinstance(row, Mapping):
                raise TypeError(f"correlation[{symbol_a}] must be a mapping.")

            normalized_row = {}

            for symbol_b, value in row.items():
                symbol_b = str(symbol_b).strip().upper()
                if not symbol_b:
                    raise ValueError("correlation contains an empty symbol.")

                corr = _require_finite(
                    f"correlation[{symbol_a}][{symbol_b}]",
                    value,
                )

                if corr < -1.0 or corr > 1.0:
                    raise ValueError("Correlation values must be in [-1, 1].")

                normalized_row[symbol_b] = corr

            normalized_correlation[symbol_a] = MappingProxyType(normalized_row)


        object.__setattr__(self, "symbols", MappingProxyType(normalized_symbols))
        object.__setattr__(self, "correlation", MappingProxyType(normalized_correlation))
        object.__setattr__(self, "risk_blocked", bool(self.risk_blocked))

    @property
    def total_current_exposure(self) -> float:
        """
        Sum of absolute current symbol exposures.
        """
        return sum(
            abs(snapshot.exposure)
            for snapshot in self.symbols.values()
        )

    @property
    def total_current_used_margin(self) -> float:
        """
        Sum of currently used symbol margins.
        """
        return sum(
            snapshot.used_margin
            for snapshot in self.symbols.values()
        )

    def symbol_snapshot(
        self,
        symbol: str,
    ) -> Optional[SymbolRiskSnapshot]:
        
        return self.symbols.get(
            str(symbol).strip().upper()
        )

# ============================================================================= END