
"""
این کلاس سه مسئولیت دارد:
    - اعتبارسنجی ساختار پروفایل‌ها،
    - انتخاب تعرفه بر اساس کارگزار/نوع حساب/نماد،
    - و تبدیل تعرفهٔ round_turn به نرخ قابل استفاده برای محاسبهٔ هر سمت معامله.

برای پروفایل‌های دارای strict_symbols: true،
نبودن نماد در تنظیمات باعث خطا می‌شود؛ سیستم بی‌صدا به نرخ عمومی برنمی‌گردد.
"""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from .contracts import CommissionModel



def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _key(value: Any, path: str) -> str:
    result = str(value).strip().lower()
    if not result:
        raise ValueError(f"{path} is required")
    return result


def _symbol(value: Any, path: str) -> str:
    result = str(value).strip().upper()
    if not result:
        raise ValueError(f"{path} is required")
    return result


@dataclass(frozen=True, slots=True)
class CommissionProfile:
    """One explicit commission schedule for an account or symbol."""

    key: str
    model: CommissionModel
    billing_basis: str = "per_side"
    notional_basis: str = "quote"
    verified: bool = False

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("commission profile key is required")
        if not isinstance(self.model, CommissionModel):
            raise TypeError("commission profile model must be CommissionModel")

        billing_basis = str(self.billing_basis).strip().lower()
        if billing_basis not in {"per_side", "round_turn"}:
            raise ValueError("billing_basis must be per_side or round_turn")

        notional_basis = str(self.notional_basis).strip().lower()
        if notional_basis not in {"base", "quote"}:
            raise ValueError("notional_basis must be base or quote")

        if billing_basis == "round_turn" and self.model.minimum != 0.0:
            raise ValueError(
                "round_turn profiles require minimum=0 until round-turn "
                "minimum accounting across partial fills is implemented"
            )

        if not isinstance(self.verified, bool):
            raise TypeError("verified must be bool")

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "billing_basis", billing_basis)
        object.__setattr__(self, "notional_basis", notional_basis)


    @property
    def amount_multiplier(self) -> float:
        return 0.5 if self.billing_basis == "round_turn" else 1.0


    @property
    def per_side_model(self) -> CommissionModel:
        """Normalize a schedule for engines that charge each fill/side."""
        if self.billing_basis == "per_side":
            return self.model

        return CommissionModel(
            basis=self.model.basis,
            rate=self.model.rate * 0.5,
            currency=self.model.currency,
            minimum=0.0,
        )



@dataclass(frozen=True, slots=True)
class _AccountProfiles:
    default: CommissionProfile | None
    symbols: Mapping[str, CommissionProfile]
    strict_symbols: bool



@dataclass(frozen=True, slots=True)
class CommissionProfileCatalog:
    """Validated broker/account/symbol commission schedules."""

    active_broker: str
    active_account_type: str
    default: CommissionProfile
    brokers: Mapping[str, Mapping[str, _AccountProfiles]]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        fallback_model: CommissionModel,
    ) -> "CommissionProfileCatalog":
        raw = _mapping(raw, "commission_profiles")
        if not isinstance(fallback_model, CommissionModel):
            raise TypeError("fallback_model must be CommissionModel")

        active_broker = _key(
            raw.get("active_broker", "default"), "active_broker"
        )
        active_account = _key(
            raw.get("active_account_type", "default"),
            "active_account_type",
        )

        default_raw = raw.get("default")
        if default_raw is not None:
            default = cls._parse_profile(default_raw, "default")
        else:
            default = CommissionProfile(
                key="legacy_default",
                model=fallback_model,
                billing_basis="per_side",
                notional_basis="quote",
                verified=False,
            )

        brokers_raw = _mapping(
            raw.get("brokers", {}),
            "commission_profiles.brokers",
        )
        brokers: dict[str, dict[str, _AccountProfiles]] = {}

        for raw_broker, raw_broker_config in brokers_raw.items():
            broker_key = _key(raw_broker, "broker name")
            if broker_key in brokers:
                raise ValueError(
                    f"duplicate broker profile after normalization: {broker_key}"
                )

            broker_config = _mapping(
                raw_broker_config,
                f"commission_profiles.brokers.{broker_key}",
            )
            accounts_raw = _mapping(
                broker_config.get("accounts", {}),
                f"commission_profiles.brokers.{broker_key}.accounts",
            )
            accounts: dict[str, _AccountProfiles] = {}

            for raw_account, raw_account_config in accounts_raw.items():
                account_key = _key(raw_account, "account type")
                if account_key in accounts:
                    raise ValueError(
                        f"duplicate account profile after normalization: "
                        f"{broker_key}/{account_key}"
                    )

                account_config = _mapping(
                    raw_account_config,
                    f"commission_profiles.brokers.{broker_key}."
                    f"accounts.{account_key}",
                )
                strict_symbols = account_config.get("strict_symbols", False)
                if not isinstance(strict_symbols, bool):
                    raise TypeError(
                        f"strict_symbols must be bool for "
                        f"{broker_key}/{account_key}"
                    )

                account_default_raw = account_config.get("default")
                account_default = (
                    cls._parse_profile(
                        account_default_raw,
                        f"{broker_key}/{account_key}/default",
                    )
                    if account_default_raw is not None
                    else None
                )

                symbols_raw = _mapping(
                    account_config.get("symbols", {}),
                    f"commission_profiles.brokers.{broker_key}."
                    f"accounts.{account_key}.symbols",
                )
                symbols: dict[str, CommissionProfile] = {}

                for raw_symbol, raw_profile in symbols_raw.items():
                    symbol = _symbol(raw_symbol, "symbol profile key")
                    if symbol in symbols:
                        raise ValueError(
                            f"duplicate symbol profile after normalization: "
                            f"{broker_key}/{account_key}/{symbol}"
                        )

                    symbols[symbol] = cls._parse_profile(
                        raw_profile,
                        f"{broker_key}/{account_key}/{symbol}",
                    )

                accounts[account_key] = _AccountProfiles(
                    default=account_default,
                    symbols=symbols,
                    strict_symbols=strict_symbols,
                )

            brokers[broker_key] = accounts

        if active_broker != "default" and active_broker not in brokers:
            raise KeyError(
                f"active commission broker profile not found: {active_broker}"
            )

        if active_broker != "default":
            if active_account not in brokers[active_broker]:
                raise KeyError(
                    f"active commission account profile not found: "
                    f"{active_broker}/{active_account}"
                )
        elif active_account != "default":
            raise ValueError(
                "active_account_type must be default when active_broker is default"
            )

        return cls(
            active_broker=active_broker,
            active_account_type=active_account,
            default=default,
            brokers=brokers,
        )


    @staticmethod
    def _parse_profile(raw: Any, key: str) -> CommissionProfile:
        item = _mapping(raw, f"commission profile {key}")
        for required in ("basis", "rate", "currency"):
            if required not in item:
                raise KeyError(
                    f"commission profile {key} missing {required}"
                )

        model = CommissionModel(
            basis=item["basis"],
            rate=item["rate"],
            currency=item["currency"],
            minimum=item.get("minimum", 0.0),
        )

        return CommissionProfile(
            key=key,
            model=model,
            billing_basis=item.get("billing_basis", "per_side"),
            notional_basis=item.get("notional_basis", "quote"),
            verified=item.get("verified", False),
        )


    def resolve(
        self,
        *,
        symbol: str,
        broker: str | None = None,
        account_type: str | None = None,
    ) -> CommissionProfile:
        symbol_key = _symbol(symbol, "symbol")
        broker_key = _key(
            self.active_broker if broker is None else broker,
            "broker",
        )
        account_key = _key(
            self.active_account_type
            if account_type is None
            else account_type,
            "account_type",
        )

        if broker_key == "default":
            if account_key != "default":
                raise ValueError(
                    "account_type cannot be selected without a non-default broker"
                )
            return self.default

        try:
            account = self.brokers[broker_key][account_key]
        except KeyError as exc:
            raise KeyError(
                f"commission profile not found for broker/account: "
                f"{broker_key}/{account_key}"
            ) from exc

        exact = account.symbols.get(symbol_key)
        if exact is not None:
            return exact

        if account.strict_symbols:
            raise KeyError(
                f"no commission profile for "
                f"{broker_key}/{account_key}/{symbol_key}; "
                "strict_symbols=true prevents fallback"
            )

        if account.default is not None:
            return account.default

        return self.default

