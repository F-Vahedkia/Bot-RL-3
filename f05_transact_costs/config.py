
# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
from typing import Any

# =============================================================================
# Functions
# =============================================================================

def _required(m: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in m:
        raise KeyError(f"missing required config key: {path}.{key}")
    return m[key]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _text(value: Any, path: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{path} is required")
    return result


def _currency(value: Any, path: str) -> str:
    result = _text(value, path).upper()
    if len(result) != 3 or not result.isalpha():
        raise ValueError(f"{path} must be a 3-letter currency code")
    return result


def _float(value: Any, path: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{path} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if strict and result <= minimum:
        raise ValueError(f"{path} must be > {minimum}")
    if not strict and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    return result


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be bool")
    return value


# =============================================================================
# Classes
# =============================================================================

@dataclass(frozen=True, slots=True)
class SimulationConfig:
    spread_source: str
    require_quote: bool
    slippage_model: str
    slippage_mean_pips: float
    slippage_std_pips: float
    slippage_cap_pips: float
    commission_basis: str
    commission_rate: float
    commission_currency: str
    commission_minimum: float
    additional_fee: float
    additional_fee_currency: str
    charge_on_entry: bool
    charge_on_exit: bool


@dataclass(frozen=True, slots=True)
class LiveConfig:
    spread_source: str
    slippage_source: str
    slippage_cap_pips: float
    slippage_cap_policy: str
    slippage_on_exceed: str
    commission_source: str


@dataclass(frozen=True, slots=True)
class AccountingConfig:
    price_costs_embedded_in_fill: bool

    def __post_init__(self) -> None:
        if not isinstance(self.price_costs_embedded_in_fill, bool):
            raise TypeError(
                "accounting.price_costs_embedded_in_fill must be bool"
            )

        if not self.price_costs_embedded_in_fill:
            raise ValueError(
                "accounting.price_costs_embedded_in_fill must be true "
                "for the current executable-fill architecture"
            )


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    version: int
    account_currency_source: str
    simulation: SimulationConfig
    live: LiveConfig
    accounting: AccountingConfig
    deterministic_seed_source: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TransactionCostConfig":
        root = _mapping(raw, "root")
        if "transaction_costs" in root:
            root = _mapping(root["transaction_costs"], "root.transaction_costs")
        else:
            root = _mapping(root, "transaction_costs")

        version = int(_required(root, "version", "transaction_costs"))
        if version != 1:
            raise ValueError("transaction_costs.version must be 1")
        currency_source = _text(_required(root, "account_currency_source", "transaction_costs"), "transaction_costs.account_currency_source")
        if currency_source != "project.base_currency":
            raise ValueError("transaction_costs.account_currency_source must be project.base_currency")

        sim = _mapping(_required(root, "simulation", "transaction_costs"), "transaction_costs.simulation")
        spread = _mapping(_required(sim, "spread", "transaction_costs.simulation"), "transaction_costs.simulation.spread")
        slippage = _mapping(_required(sim, "slippage", "transaction_costs.simulation"), "transaction_costs.simulation.slippage")
        commission = _mapping(_required(sim, "commission", "transaction_costs.simulation"), "transaction_costs.simulation.commission")
        fee = _mapping(_required(sim, "additional_fee", "transaction_costs.simulation"), "transaction_costs.simulation.additional_fee")

        spread_source = _text(_required(spread, "source", "transaction_costs.simulation.spread"), "transaction_costs.simulation.spread.source")
        if spread_source != "historical_bid_ask":
            raise ValueError("simulation.spread.source must be historical_bid_ask")
        require_quote = _bool(_required(spread, "require_quote", "transaction_costs.simulation.spread"), "transaction_costs.simulation.spread.require_quote")

        slippage_model = _text(_required(slippage, "model", "transaction_costs.simulation.slippage"), "transaction_costs.simulation.slippage.model")
        if slippage_model != "normal_capped":
            raise ValueError("simulation.slippage.model must be normal_capped")
        mean = _float(_required(slippage, "mean_pips", "transaction_costs.simulation.slippage"), "simulation.slippage.mean_pips")
        std = _float(_required(slippage, "std_pips", "transaction_costs.simulation.slippage"), "simulation.slippage.std_pips")
        cap = _float(_required(slippage, "cap_pips", "transaction_costs.simulation.slippage"), "simulation.slippage.cap_pips", strict=True)
        if mean > cap:
            raise ValueError("simulation.slippage.mean_pips must be <= cap_pips")

        basis = _text(_required(commission, "basis", "transaction_costs.simulation.commission"), "transaction_costs.simulation.commission.basis")
        if basis not in {"fixed", "per_lot", "percent_notional"}:
            raise ValueError("unsupported simulation commission basis")
        rate = _float(_required(commission, "rate", "transaction_costs.simulation.commission"), "simulation.commission.rate")
        minimum = _float(_required(commission, "minimum", "transaction_costs.simulation.commission"), "simulation.commission.minimum")
        commission_currency = _currency(_required(commission, "currency", "transaction_costs.simulation.commission"), "simulation.commission.currency")
        if basis == "percent_notional" and rate > 1.0:
            raise ValueError("simulation percent_notional commission rate must be <= 1")

        additional_fee = _float(_required(fee, "value", "transaction_costs.simulation.additional_fee"), "simulation.additional_fee.value")
        fee_currency = _currency(_required(fee, "currency", "transaction_costs.simulation.additional_fee"), "simulation.additional_fee.currency")

        live = _mapping(_required(root, "live", "transaction_costs"), "transaction_costs.live")
        live_spread = _mapping(_required(live, "spread", "transaction_costs.live"), "transaction_costs.live.spread")
        live_slippage = _mapping(_required(live, "slippage", "transaction_costs.live"), "transaction_costs.live.slippage")
        live_commission = _mapping(_required(live, "commission", "transaction_costs.live"), "transaction_costs.live.commission")
        if _text(_required(live_spread, "source", "transaction_costs.live.spread"), "transaction_costs.live.spread.source") != "broker":
            raise ValueError("live.spread.source must be broker")
        if _text(_required(live_slippage, "source", "transaction_costs.live.slippage"), "transaction_costs.live.slippage.source") != "broker":
            raise ValueError("live.slippage.source must be broker")
        live_cap = _float(_required(live_slippage, "cap_pips", "transaction_costs.live.slippage"), "transaction_costs.live.slippage.cap_pips", strict=True)
        cap_policy = _text(_required(live_slippage, "cap_policy", "transaction_costs.live.slippage"), "transaction_costs.live.slippage.cap_policy").lower()
        on_exceed = _text(_required(live_slippage, "on_exceed", "transaction_costs.live.slippage"), "transaction_costs.live.slippage.on_exceed").lower()
        if cap_policy != "tolerance":
            raise ValueError("live.slippage.cap_policy must be tolerance")
        if on_exceed not in {"flag_and_reconcile", "reject"}:
            raise ValueError("live.slippage.on_exceed must be flag_and_reconcile or reject")
        if _text(_required(live_commission, "source", "transaction_costs.live.commission"), "transaction_costs.live.commission.source") != "broker":
            raise ValueError("live.commission.source must be broker")

        # ----- accounting & accounting_cfg --------------------------------------------------------------------------- start
        accounting = _mapping(
            _required(root, "accounting", "transaction_costs"),
            "transaction_costs.accounting",
        )

        price_embedded = _bool(
            _required(
                accounting,
                "price_costs_embedded_in_fill",
                "transaction_costs.accounting",
            ),
            "transaction_costs.accounting.price_costs_embedded_in_fill",
        )

        accounting_cfg = AccountingConfig(
            price_costs_embedded_in_fill=price_embedded,
        )

        # New canonical location:
        # transaction_costs.simulation.cost_application
        policy_raw = sim.get("cost_application")

        if policy_raw is None:
            # Backward-compatible parsing of the previous config layout.
            # These legacy values now govern simulated/expected costs only.
            charge_on_entry = _bool(
                _required(
                    accounting,
                    "charge_on_entry",
                    "transaction_costs.accounting",
                ),
                "transaction_costs.accounting.charge_on_entry",
            )
            charge_on_exit = _bool(
                _required(
                    accounting,
                    "charge_on_exit",
                    "transaction_costs.accounting",
                ),
                "transaction_costs.accounting.charge_on_exit",
            )
        else:
            policy = _mapping(
                policy_raw,
                "transaction_costs.simulation.cost_application",
            )

            charge_on_entry = _bool(
                _required(
                    policy,
                    "charge_on_entry",
                    "transaction_costs.simulation.cost_application",
                ),
                "transaction_costs.simulation.cost_application.charge_on_entry",
            )
            charge_on_exit = _bool(
                _required(
                    policy,
                    "charge_on_exit",
                    "transaction_costs.simulation.cost_application",
                ),
                "transaction_costs.simulation.cost_application.charge_on_exit",
            )

            # Do not silently accept contradictory duplicate settings.
            for name, new_value in (
                ("charge_on_entry", charge_on_entry),
                ("charge_on_exit", charge_on_exit),
            ):
                if name in accounting:
                    legacy_value = _bool(
                        accounting[name],
                        f"transaction_costs.accounting.{name}",
                    )
                    if legacy_value != new_value:
                        raise ValueError(
                            f"conflicting settings for {name}: "
                            "simulation.cost_application and legacy accounting"
                        )
        # ----- accounting & accounting_cfg --------------------------------------------------------------------------- end

        deterministic = _mapping(_required(root, "deterministic", "transaction_costs"), "transaction_costs.deterministic")
        seed_source = _text(_required(deterministic, "seed_source", "transaction_costs.deterministic"), "transaction_costs.deterministic.seed_source")

        return cls(
            version=version,
            account_currency_source=currency_source,
            simulation=SimulationConfig(
                spread_source=spread_source,
                require_quote=require_quote,
                slippage_model=slippage_model,
                slippage_mean_pips=mean,
                slippage_std_pips=std,
                slippage_cap_pips=cap,
                commission_basis=basis,
                commission_rate=rate,
                commission_currency=commission_currency,
                commission_minimum=minimum,
                additional_fee=additional_fee,
                additional_fee_currency=fee_currency,
                charge_on_entry=charge_on_entry,
                charge_on_exit=charge_on_exit,
            ),
            live=LiveConfig(
                spread_source="broker",
                slippage_source="broker",
                slippage_cap_pips=live_cap,
                slippage_cap_policy=cap_policy,
                slippage_on_exceed=on_exceed,
                commission_source="broker",
            ),
            accounting=accounting_cfg,
            deterministic_seed_source=seed_source,
        )

    @staticmethod
    def resolve_account_currency(project_config: Mapping[str, Any]) -> str:
        project = _mapping(_required(project_config, "project", "root"), "root.project")
        return _currency(_required(project, "base_currency", "root.project"), "root.project.base_currency")

# ============================================================================= END