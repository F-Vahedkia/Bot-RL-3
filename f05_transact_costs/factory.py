

from __future__ import annotations

from dataclasses import dataclass

from .calculator import TransactionCostCalculator
from .config import TransactionCostConfig
from .contracts import CommissionModel
from .engine import LiveObservedExecutionCostEngine, SimulationExecutionCostEngine
from .models import NormalCappedSlippageModel
from .round_trip import RoundTripCostEstimator
from .commission_profiles import CommissionProfile, CommissionProfileCatalog


@dataclass(frozen=True, slots=True)
class CostComponents:
    calculator: TransactionCostCalculator
    round_trip_estimator: RoundTripCostEstimator
    simulation_engine: SimulationExecutionCostEngine
    live_engine: LiveObservedExecutionCostEngine
    simulation_commission: CommissionModel
    commission_profiles: CommissionProfileCatalog | None = None

    def commission_profile_for(
        self,
        *,
        symbol: str,
        broker: str | None = None,
        account_type: str | None = None,
    ) -> CommissionProfile:
        if self.commission_profiles is None:
            raise RuntimeError("commission profile catalog is not configured")
        return self.commission_profiles.resolve(
            symbol=symbol,
            broker=broker,
            account_type=account_type,
        )

    def simulation_commission_for(
        self,
        *,
        symbol: str,
        broker: str | None = None,
        account_type: str | None = None,
    ) -> CommissionModel:
        """Return the selected commission model normalized to one side/fill."""
        return self.commission_profile_for(
            symbol=symbol,
            broker=broker,
            account_type=account_type,
        ).per_side_model


def build_components(config: TransactionCostConfig, *, project_random_seed: int) -> CostComponents:
    if not isinstance(config, TransactionCostConfig):
        raise TypeError("config must be TransactionCostConfig")
    sim = config.simulation
    commission = CommissionModel(
        basis=sim.commission_basis,
        rate=sim.commission_rate,
        currency=sim.commission_currency,
        minimum=sim.commission_minimum,
    )
    slippage = NormalCappedSlippageModel(
        mean_pips=sim.slippage_mean_pips,
        std_pips=sim.slippage_std_pips,
        cap_pips=sim.slippage_cap_pips,
        seed=int(project_random_seed),
    )
    calculator = TransactionCostCalculator()

    return CostComponents(
        calculator=calculator,
        round_trip_estimator=RoundTripCostEstimator(
            calculator,
            charge_on_entry=sim.charge_on_entry,
            charge_on_exit=sim.charge_on_exit,
        ),
        simulation_engine=SimulationExecutionCostEngine(
            calculator,
            slippage,
            charge_on_entry=sim.charge_on_entry,
            charge_on_exit=sim.charge_on_exit,
        ),
        live_engine=LiveObservedExecutionCostEngine(
            calculator,
            slippage_cap_pips=config.live.slippage_cap_pips,
            cap_policy=config.live.slippage_cap_policy,
            on_exceed=config.live.slippage_on_exceed,
        ),
        simulation_commission=commission,
        commission_profiles=sim.commission_profiles,
    )

