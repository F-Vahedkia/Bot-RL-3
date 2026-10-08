# f08_risk/margin_calculator.py (12)
"""
نکات اصلی این فایل:
--------------------
    - InstrumentMarginSpec مشخصات ثابت لازم برای محاسبهٔ Margin را تعریف می‌کند:
        contract_size، leverage و در صورت وجود margin_rate.
    - margin_rate صریح، در صورت ارائه، بر 1 / leverage اولویت دارد.
    - MarginRequest ورودی یک محاسبهٔ deterministic برای یک symbol است.
    - currency_conversion_rate تبدیل Notional به currency حساب را پوشش می‌دهد.
    - مدل فعلی فقط forex_leverage را پشتیبانی می‌کند.
    - MarginResult خروجی immutable محاسبه است.
    - فرمول فعلی در MarginCalculator این است:
        notional           = volume_lots × contract_size × price
        converted_notional = notional × currency_conversion_rate
        required_margin    = converted_notional × margin_rate
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

# =============================================================================
# Margin Mode
# =============================================================================

MARGIN_MODE_FOREX_LEVERAGE = "forex_leverage"

# =============================================================================
# Instrument Specification
# =============================================================================

@dataclass(frozen=True, slots=True)
class InstrumentMarginSpec:
    """
    Static instrument parameters required for margin calculation.

    All monetary quantities are expressed in the same account-currency
    convention after applying currency_conversion_rate where required.
    """
    symbol: str
    contract_size: float
    leverage: float

    # Optional explicit margin rate.
    # When supplied, it takes precedence over 1 / leverage.
    margin_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        if not isfinite(self.contract_size) or self.contract_size <= 0.0:
            raise ValueError("contract_size must be finite and > 0")

        if not isfinite(self.leverage) or self.leverage <= 0.0:
            raise ValueError("leverage must be finite and > 0")

        if self.margin_rate is not None:
            if not isfinite(self.margin_rate) or self.margin_rate <= 0.0:
                raise ValueError("margin_rate must be finite and > 0")

            if self.margin_rate > 1.0:
                raise ValueError("margin_rate must be <= 1")


# =============================================================================
# Margin Request
# =============================================================================

@dataclass(frozen=True, slots=True)
class MarginRequest:
    """
    One deterministic margin calculation request.
    """
    symbol: str
    volume_lots: float
    price: float
    spec: InstrumentMarginSpec

    # Conversion from the instrument-notional currency into account currency.
    # 1.0 means no conversion is required.
    currency_conversion_rate: float = 1.0

    margin_mode: str = MARGIN_MODE_FOREX_LEVERAGE

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        if self.symbol != self.spec.symbol:
            raise ValueError(
                f"symbol mismatch: request={self.symbol!r}, "
                f"spec={self.spec.symbol!r}"
            )

        if not isfinite(self.volume_lots) or self.volume_lots < 0.0:
            raise ValueError("volume_lots must be finite and >= 0")

        if not isfinite(self.price) or self.price <= 0.0:
            raise ValueError("price must be finite and > 0")

        if (
            not isfinite(self.currency_conversion_rate)
            or self.currency_conversion_rate <= 0.0
        ):
            raise ValueError("currency_conversion_rate must be finite and > 0")

        if self.margin_mode != MARGIN_MODE_FOREX_LEVERAGE:
            raise ValueError(f"Unsupported margin_mode: {self.margin_mode!r}")


# =============================================================================
# Margin Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class MarginResult:
    """
    Deterministic result of one margin calculation.
    """
    symbol: str
    volume_lots: float
    price: float
    notional_value: float
    margin_rate: float
    required_margin: float
    calculation_method: str

    def __post_init__(self) -> None:
        values = (
            self.volume_lots,
            self.price,
            self.notional_value,
            self.margin_rate,
            self.required_margin,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("MarginResult contains non-finite values")

        if self.volume_lots < 0.0:
            raise ValueError("volume_lots must be >= 0")

        if self.price <= 0.0:
            raise ValueError("price must be > 0")

        if self.notional_value < 0.0:
            raise ValueError("notional_value must be >= 0")

        if self.margin_rate <= 0.0:
            raise ValueError("margin_rate must be > 0")

        if self.required_margin < 0.0:
            raise ValueError("required_margin must be >= 0")


# =============================================================================
# Calculator
# =============================================================================

class MarginCalculator:
    """
    Pure, deterministic margin calculator.
    This class performs no broker calls and does not mutate account state.
    Forex leverage model:
        notional =
            volume_lots * contract_size * price
        converted_notional =
            notional * currency_conversion_rate
        margin =
            converted_notional * margin_rate

    When an explicit margin_rate exists in InstrumentMarginSpec, it is used.
    Otherwise:
        margin_rate = 1 / leverage
    """

    def calculate(self, request: MarginRequest) -> MarginResult:
        if not isinstance(request, MarginRequest):
            raise TypeError(
                f"Expected MarginRequest, got {type(request).__name__}"
            )

        margin_rate = self._resolve_margin_rate(request.spec)
        notional_value = (
            request.volume_lots
            * request.spec.contract_size
            * request.price
        )
        converted_notional = (
            notional_value
            * request.currency_conversion_rate
        )
        required_margin = converted_notional * margin_rate

        return MarginResult(
            symbol=request.symbol,
            volume_lots=request.volume_lots,
            price=request.price,
            notional_value=converted_notional,
            margin_rate=margin_rate,
            required_margin=required_margin,
            calculation_method=request.margin_mode,
        )

    @staticmethod
    def _resolve_margin_rate(spec: InstrumentMarginSpec) -> float:
        if spec.margin_rate is not None:
            return spec.margin_rate

        return 1.0 / spec.leverage

# ============================================================================= END
