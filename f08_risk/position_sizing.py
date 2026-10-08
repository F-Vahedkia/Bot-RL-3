# f08_risk/position_sizing.py (14)
"""
این فایل نقش مشخصی در Risk Layer دارد:

    PositionSizingRequest
            ↓
    PositionSizingCalculator
            ↓
    PositionSizingResult

    قرارداد فعلی آن این است که target_exposure به‌صورت کسری از equity در نظر گرفته می‌شود و
    سپس به notional و در نهایت lots تبدیل می‌شود:

target_notional = equity × target_exposure
volume_lots = target_notional / (price × contract_size × currency_conversion_rate)

ویژگی‌های مهم:
--------------
    - کاملاً deterministic است.
    - broker و execution را نمی‌شناسد.
    - state حساب را تغییر نمی‌دهد.
    - target_exposure در این calculator غیرمنفی است؛ sign جهت معامله در لایهٔ دیگری قرار دارد.
    - PositionSizingResult تمام مقادیر ورودی و خروجی مهم را نگه می‌دارد و immutable است.
    - تستی که قبلاً برای PositionSizingCalculator داشتیم 12/12 سبز شده بود؛ بنابراین این بخش را فعلاً جزو بخش‌های تثبیت‌شدهٔ Chapter 4 در نظر می‌گیرم.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite


# =============================================================================
# Position Sizing Request
# =============================================================================

@dataclass(frozen=True, slots=True)
class PositionSizingRequest:
    """
    Convert normalized target exposure into notional value and lots.
    target_exposure is expressed as a fraction of account equity.
        target_notional = equity * target_exposure
        lots =
            target_notional
            / (price * contract_size * currency_conversion_rate)
    """
    symbol: str
    target_exposure: float
    equity: float
    price: float
    contract_size: float
    currency_conversion_rate: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        if not isfinite(self.target_exposure):
            raise ValueError("target_exposure must be finite")

        if self.target_exposure < 0.0:
            raise ValueError("target_exposure must be >= 0")

        if not isfinite(self.equity) or self.equity < 0.0:
            raise ValueError("equity must be finite and >= 0")

        if not isfinite(self.price) or self.price <= 0.0:
            raise ValueError("price must be finite and > 0")

        if (
            not isfinite(self.contract_size)
            or self.contract_size <= 0.0
        ):
            raise ValueError("contract_size must be finite and > 0")

        if (
            not isfinite(self.currency_conversion_rate)
            or self.currency_conversion_rate <= 0.0
        ):
            raise ValueError("currency_conversion_rate must be finite and > 0")


# =============================================================================
# Position Sizing Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class PositionSizingResult:
    symbol: str
    target_exposure: float
    equity: float
    target_notional: float
    price: float
    contract_size: float
    currency_conversion_rate: float
    volume_lots: float

    def __post_init__(self) -> None:
        values = (
            self.target_exposure,
            self.equity,
            self.target_notional,
            self.price,
            self.contract_size,
            self.currency_conversion_rate,
            self.volume_lots,
        )

        if not all(isfinite(value) for value in values):
            raise ValueError("PositionSizingResult contains non-finite values")

        if self.target_exposure < 0.0:
            raise ValueError("target_exposure must be >= 0")

        if self.equity < 0.0:
            raise ValueError("equity must be >= 0")

        if self.target_notional < 0.0:
            raise ValueError("target_notional must be >= 0")

        if self.price <= 0.0:
            raise ValueError("price must be > 0")

        if self.contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")

        if self.currency_conversion_rate <= 0.0:
            raise ValueError("currency_conversion_rate must be > 0")

        if self.volume_lots < 0.0:
            raise ValueError("volume_lots must be >= 0")


# =============================================================================
# Calculator
# =============================================================================

class PositionSizingCalculator:
    """
    Pure and deterministic exposure-to-volume calculator.
    No broker calls.
    No account-state mutation.
    """

    def calculate(
        self,
        request: PositionSizingRequest,
    ) -> PositionSizingResult:
        if not isinstance(request, PositionSizingRequest):
            raise TypeError(
                "Expected PositionSizingRequest, "
                f"got {type(request).__name__}"
            )

        target_notional = (
            request.equity
            * request.target_exposure
        )

        denominator = (
            request.price
            * request.contract_size
            * request.currency_conversion_rate
        )

        volume_lots = target_notional / denominator

        return PositionSizingResult(
            symbol=request.symbol,
            target_exposure=request.target_exposure,
            equity=request.equity,
            target_notional=target_notional,
            price=request.price,
            contract_size=request.contract_size,
            currency_conversion_rate=request.currency_conversion_rate,
            volume_lots=volume_lots,
        )

# ============================================================================= END
