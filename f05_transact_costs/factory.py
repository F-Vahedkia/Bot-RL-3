from __future__ import annotations

from dataclasses import dataclass

from .calculator import TransactionCostCalculator
from .config import TransactionCostConfig
from .contracts import CommissionModel
from .engine import LiveObservedExecutionCostEngine, SimulationExecutionCostEngine
from .models import NormalCappedSlippageModel
from .round_trip import RoundTripCostEstimator


@dataclass(frozen=True, slots=True)
class CostComponents:
    calculator: TransactionCostCalculator
    round_trip_estimator: RoundTripCostEstimator
    simulation_engine: SimulationExecutionCostEngine
    live_engine: LiveObservedExecutionCostEngine
    simulation_commission: CommissionModel


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
        round_trip_estimator=RoundTripCostEstimator(calculator),
        simulation_engine=SimulationExecutionCostEngine(calculator, slippage),
        live_engine=LiveObservedExecutionCostEngine(
            calculator,
            slippage_cap_pips=config.live.slippage_cap_pips,
            cap_policy=config.live.slippage_cap_policy,
            on_exceed=config.live.slippage_on_exceed,
        ),
        simulation_commission=commission,
    )
