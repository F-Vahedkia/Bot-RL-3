# f08_risk/projection_margin.py (16)
"""
این فایل در معماری فعلی نقش پل بین Position Sizing و Margin Calculation برای Projection را دارد:

    ProjectionMarginRequest
            ↓
    ProjectionMarginCalculator
            ├── PositionSizingCalculator
            └── MarginCalculator
            ↓
    ProjectionMarginResult

نکات مهم:
---------
    - target_exposure به‌صورت fraction از equity دریافت می‌شود.
    - برای محاسبهٔ lots، از PositionSizingCalculator استفاده می‌شود.
    - سپس همان lots به MarginCalculator داده می‌شود تا required margin محاسبه شود.
    - جهت exposure در این مسیر برای sizing با abs(target_exposure) حذف می‌شود؛
        بنابراین calculation مقدار ریسک/اندازه را می‌سنجد، نه direction را.
    - خروجی همزمان target_notional، target_lots، required_margin، margin_rate و calculation_method را ارائه می‌کند.
    - calculate_many() همین محاسبات را برای چند symbol انجام می‌دهد.
    - کلاس stateless و broker-independent است.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

from f08_risk.margin_calculator import (
    InstrumentMarginSpec,
    MarginCalculator,
    MarginRequest,
)
from f08_risk.position_sizing import (
    PositionSizingCalculator,
    PositionSizingRequest,
)


# =============================================================================
# Request
# =============================================================================

@dataclass(frozen=True, slots=True)
class ProjectionMarginRequest:
    symbol: str

    # Exposure is expressed as a fraction of account equity.
    target_exposure: float

    equity: float
    price: float
    instrument: InstrumentMarginSpec
    currency_conversion_rate: float = 1.0


# =============================================================================
# Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class ProjectionMarginResult:
    symbol: str
    target_exposure: float

    target_notional: float
    target_lots: float

    required_margin: float
    margin_rate: float

    calculation_method: str


# =============================================================================
# Calculator
# =============================================================================

class ProjectionMarginCalculator:
    """
    Combines PositionSizingCalculator and MarginCalculator.
    This is intentionally stateless and broker-independent.
    """

    def __init__(
        self,
        *,
        position_sizer: PositionSizingCalculator | None = None,
        margin_calculator: MarginCalculator | None = None,
    ) -> None:
        self._position_sizer = (
            position_sizer
            if position_sizer is not None
            else PositionSizingCalculator()
        )

        self._margin_calculator = (
            margin_calculator
            if margin_calculator is not None
            else MarginCalculator()
        )


    def calculate(
        self,
        request: ProjectionMarginRequest,
    ) -> ProjectionMarginResult:
        if not isinstance(request, ProjectionMarginRequest):
            raise TypeError(
                "Expected ProjectionMarginRequest, "
                f"got {type(request).__name__}"
            )

        sizing = self._position_sizer.calculate(
            PositionSizingRequest(
                symbol=request.symbol,
                target_exposure=abs(request.target_exposure),
                equity=request.equity,
                price=request.price,
                contract_size=(request.instrument.contract_size),
                currency_conversion_rate=(request.currency_conversion_rate),
            )
        )
        
        margin = self._margin_calculator.calculate(
            MarginRequest(
                symbol=request.symbol,
                volume_lots=sizing.volume_lots,
                price=request.price,
                spec=request.instrument,
                currency_conversion_rate=(request.currency_conversion_rate),
            )
        )

        return ProjectionMarginResult(
            symbol=request.symbol,
            target_exposure=request.target_exposure,
            target_notional=sizing.target_notional,
            target_lots=sizing.volume_lots,
            required_margin=margin.required_margin,
            margin_rate=margin.margin_rate,
            calculation_method=margin.calculation_method,
        )


    def calculate_many(
        self,
        requests: Mapping[str, ProjectionMarginRequest],
    ) -> dict[str, ProjectionMarginResult]:
        if requests is None:
            raise ValueError("requests are required")

        results: dict[str, ProjectionMarginResult] = {}

        for symbol, request in requests.items():
            if symbol != request.symbol:
                raise ValueError(
                    f"request symbol mismatch: "
                    f"mapping_key={symbol!r}, "
                    f"request_symbol={request.symbol!r}"
                )

            results[symbol] = self.calculate(request)

        return results

# ============================================================================= END
