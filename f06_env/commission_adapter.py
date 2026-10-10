

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging

from f05_transact_costs.commission_profiles import (
    CommissionProfileCatalog,
)
from .execution_simulator import ExecutionCost, RateSource


logger = logging.getLogger(__name__)


def _normalize_sources(
    values: Mapping[str, RateSource] | None,
    *,
    name: str,
) -> dict[str, RateSource]:
    result: dict[str, RateSource] = {}

    for raw_symbol, source in (values or {}).items():
        symbol = str(raw_symbol).strip().upper()

        if not symbol:
            raise ValueError(f"{name} contains an empty symbol")

        if symbol in result:
            raise ValueError(
                f"duplicate symbol in {name}: {symbol}"
            )

        if callable(source):
            result[symbol] = source
        else:
            if isinstance(source, bool):
                raise TypeError(
                    f"{name}[{symbol}] must not be bool"
                )

            try:
                result[symbol] = float(source)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(
                    f"{name}[{symbol}] must be a rate "
                    "or timestamp resolver"
                ) from exc

    return result


def build_execution_costs(
    *,
    profiles: CommissionProfileCatalog,
    symbols: Sequence[str],
    account_currency: str,
    broker: str | None = None,
    account_type: str | None = None,
    slippage_price_by_symbol: Mapping[str, float] | None = None,
    commission_to_account_rate_by_symbol: (
        Mapping[str, RateSource] | None
    ) = None,
    notional_to_commission_rate_by_symbol: (
        Mapping[str, RateSource] | None
    ) = None,
) -> dict[str, ExecutionCost]:
    """تبدیل پروفایل‌های f05 به هزینه‌های اجرایی per-fill در f06.

    برای تبدیل ارزهای متفاوت، نرخ ثابت یا تابع زمان‌محور بده.
    تابع زمان‌محور باید نرخ متناظر با زمان fill را برگرداند.
    این آداپتور نرخ تبدیل را حدس نمی‌زند.
    """
    if not isinstance(profiles, CommissionProfileCatalog):
        raise TypeError(
            "profiles must be CommissionProfileCatalog"
        )

    currency = str(account_currency).strip().upper()
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError(
            "account_currency must be a 3-letter currency code"
        )

    slippage: dict[str, float] = {}

    for raw_symbol, raw_value in (
        slippage_price_by_symbol or {}
    ).items():
        symbol = str(raw_symbol).strip().upper()

        if not symbol:
            raise ValueError(
                "slippage_price_by_symbol contains an empty symbol"
            )

        if symbol in slippage:
            raise ValueError(
                "duplicate symbol in slippage_price_by_symbol: "
                f"{symbol}"
            )

        if isinstance(raw_value, bool):
            raise TypeError(
                f"slippage_price_by_symbol[{symbol}] must not be bool"
            )

        try:
            slippage[symbol] = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"slippage_price_by_symbol[{symbol}] must be numeric"
            ) from exc

    commission_rates = _normalize_sources(
        commission_to_account_rate_by_symbol,
        name="commission_to_account_rate_by_symbol",
    )

    notional_rates = _normalize_sources(
        notional_to_commission_rate_by_symbol,
        name="notional_to_commission_rate_by_symbol",
    )

    result: dict[str, ExecutionCost] = {}

    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()

        if not symbol:
            raise ValueError(
                "symbols must not contain an empty symbol"
            )

        if symbol in result:
            raise ValueError(
                f"duplicate symbol after normalization: {symbol}"
            )

        profile = profiles.resolve(
            symbol=symbol,
            broker=broker,
            account_type=account_type,
        )

        if not profile.verified:
            logger.warning(
                "Commission profile %s selected for %s is unverified; "
                "simulation results are estimates, not validated "
                "broker costs",
                profile.key,
                symbol,
            )

        result[symbol] = ExecutionCost(
            slippage_price=slippage.get(symbol, 0.0),
            commission_model=profile.per_side_model,
            commission_billing_basis="per_side",
            commission_notional_basis=profile.notional_basis,
            account_currency=currency,
            commission_to_account_rate=commission_rates.get(symbol),
            notional_to_commission_rate=notional_rates.get(symbol),
        )
        """
        نکته دربارهٔ round_turn:
        آداپتور از profile.per_side_model استفاده می‌کند.

        اگر تعرفهٔ کارگزار برای کل رفت‌وبرگشت (round_turn) تعریف شده باشد،
        کلاس پروفایل آن را به سهم هر سمت تبدیل می‌کند.

        در نتیجه، آداپتور آن را دوباره نصف نمی‌کند؛
        این کار از نصف‌شدن دوبارهٔ کمیسیون جلوگیری می‌کند.

        این پیاده‌سازی هنوز حداقل کمیسیون تجمعی یک معاملهٔ کامل را برای چند fill جداگانه مدیریت نمی‌کند.
        
        به همین علت، پروفایل round_turn با minimum غیرصفر عمداً رد می‌شود.
        """
    if not result:
        raise ValueError("symbols must not be empty")

    return result

