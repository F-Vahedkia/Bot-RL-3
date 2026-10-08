# f07_agents/decision_engine.py (5)
#
# Created: 1405/06/19

# Chapter 3 - Symbol-Agent + Meta-Agent
# Multi-Symbol Decision Engine
#
# مسئولیت:
#     - مدیریت مجموعه Symbol-Agent ها
#     - تضمین symbol isolation
#     - تضمین یکسان بودن timestamp چرخه تصمیم
#     - جمع‌آوری خروجی Agent ها
#     - انتقال خروجی ها به Meta-Agent
#     - نگهداری آخرین decision cycle برای audit / replay
#
# عدم مسئولیت:
#     - execution
#     - broker
#     - MT5
#     - PnL
#     - reward
#
# Architecture:
#
#   Symbol Observations
#          |
#          v
#   Symbol-Agent ها
#          |
#          v
#   SymbolAgentOutput
#          |
#          v
#      Meta-Agent
#          |
#          v
#   PortfolioDecision
#
# نکته:
#   این Engine هنوز به f06_env وابسته نیست.


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from f07_agents.contracts import (
    AgentObservation,
    PortfolioContext,
    PortfolioDecision,
    SymbolAgentOutput,
    SymbolContext,
)
from f07_agents.meta_agent import MetaAgent
from f07_agents.symbol_agent import SymbolAgent


# =============================================================================
# Multi-Symbol Decision Engine
# =============================================================================

@dataclass(slots=True)
class MultiSymbolDecisionEngine:
    """
    Orchestrator سطح Decision Layer.

    هر Symbol-Agent دقیقاً مسئول یک symbol است.

    تمام symbol ها در یک decision cycle مشترک
    به Meta-Agent تحویل داده می‌شوند.
    """

    symbol_agents: Mapping[str, SymbolAgent]
    meta_agent: MetaAgent

    # آخرین خروجی هر Symbol-Agent
    last_symbol_outputs: dict[str, SymbolAgentOutput] = field(
        default_factory=dict,
        init=False,
    )

    # آخرین PortfolioDecision
    last_decision: PortfolioDecision | None = field(
        default=None,
        init=False,
    )

    # تعداد چرخه‌های موفق تصمیم‌گیری
    decision_count: int = field(
        default=0,
        init=False,
    )

    def __post_init__(self) -> None:

        normalized: dict[str, SymbolAgent] = {}

        for symbol, agent in self.symbol_agents.items():

            key = str(symbol).replace(" ","")

            if not key:
                raise ValueError("symbol key must not be empty")

            if not isinstance(agent, SymbolAgent):
                raise TypeError(f"Invalid SymbolAgent for {key}")

            if agent.symbol != key:
                raise ValueError(
                    f"Agent symbol mismatch: "
                    f"{agent.symbol} != {key}"
                )

            if key in normalized:
                raise ValueError(f"Duplicate symbol: {key}")

            normalized[key] = agent

        if not normalized:
            raise ValueError("symbol_agents must not be empty")

        self.symbol_agents = dict(
            sorted(normalized.items())
        )


    # =====================================================
    # Decision Cycle
    # =====================================================

    def decide(
        self,
        *,
        observations: Mapping[str, AgentObservation],
        contexts: Mapping[str, SymbolContext],
        portfolio: PortfolioContext,
        decision_id: str,
    ) -> PortfolioDecision:
        """
        اجرای یک decision cycle کامل.
        تضمین‌های این متد:
            1. تمام symbol ها باید حاضر باشند.
            2. Observation هر symbol باید متعلق به همان Agent باشد.
            3. Context هر symbol باید متعلق به همان Agent باشد.
            4. timestamp تمام Observation ها باید یکسان باشد.
            5. timestamp آنها باید با PortfolioContext یکی باشد.
            6. mode چرخه باید یکسان باشد.
            7. تمام Symbol-Agent خروجی متعلق به همان decision_id باشند.
        """

        if not str(decision_id).strip():
            raise ValueError("decision_id is required")

        if portfolio.mode != self.meta_agent.config.mode:
            raise ValueError("Portfolio mode does not match Meta-Agent mode")

        expected_symbols = set(self.symbol_agents)

        normalized_observations = {
            str(symbol).replace(" ", ""): observation
            for symbol, observation in observations.items()
        }
        if len(normalized_observations) != len(observations):
            raise ValueError(
                "Observation symbols collide after removing spaces"
            )

        normalized_contexts = {
            str(symbol).replace(" ", ""): context
            for symbol, context in contexts.items()
        }
        if len(normalized_contexts) != len(contexts):
            raise ValueError(
                "Context symbols collide after removing spaces"
            )

        observation_symbols = set(normalized_observations)
        context_symbols = set(normalized_contexts)

        if observation_symbols != expected_symbols:
            raise ValueError("Observation symbols do not match registered Symbol-Agents")
        
        if context_symbols != expected_symbols:
            raise ValueError("Context symbols do not match registered Symbol-Agents")

        # -------------------------------------------------
        # همه Observation ها باید یک timestamp داشته باشند.
        # -------------------------------------------------

        observation_timestamps = {
            normalized_observations[symbol].timestamp
            for symbol in expected_symbols
        }

        if len(observation_timestamps) != 1:
            raise ValueError("All observations in one decision cycle must have the same timestamp")

        observation_timestamp = next(
            iter(observation_timestamps)
        )

        if observation_timestamp != portfolio.timestamp:
            raise ValueError("Observation timestamp does not match PortfolioContext timestamp")

        outputs: dict[str, SymbolAgentOutput] = {}

        # -------------------------------------------------
        # Symbol-Agent inference
        # -------------------------------------------------
        for symbol in sorted(expected_symbols):

            observation = normalized_observations[symbol]
            context = normalized_contexts[symbol]

            if observation.symbol != symbol:
                raise ValueError(
                    f"Observation symbol mismatch: "
                    f"{observation.symbol} != {symbol}"
                )

            if context.symbol != symbol:
                raise ValueError(
                    f"Context symbol mismatch: "
                    f"{context.symbol} != {symbol}"
                )

            output = self.symbol_agents[symbol].decide(
                observation=observation,
                context=context,
                decision_id=decision_id,
            )

            # ---------------------------------------------
            # خروجی Agent باید متعلق به همان چرخه باشد.
            # ---------------------------------------------
            if output.symbol != symbol:
                raise ValueError(
                    f"Agent output symbol mismatch: "
                    f"{output.symbol} != {symbol}"
                )

            if output.timestamp != portfolio.timestamp:
                raise ValueError(
                    f"Agent output timestamp mismatch "
                    f"for {symbol}"
                )

            if output.mode != portfolio.mode:
                raise ValueError(
                    f"Agent output mode mismatch "
                    f"for {symbol}"
                )

            if output.decision_id != decision_id:
                raise ValueError(
                    f"Agent output decision_id mismatch "
                    f"for {symbol}"
                )

            outputs[symbol] = output

        # -------------------------------------------------
        # Meta-Agent inference
        # -------------------------------------------------
        decision = self.meta_agent.decide(
            signals=outputs,
            portfolio=portfolio,
            decision_id=decision_id,
        )

        # -------------------------------------------------
        # Audit state
        # -------------------------------------------------
        self.last_symbol_outputs = dict(outputs)
        self.last_decision = decision
        self.decision_count += 1

        return decision

    # =====================================================
    # Agent Access
    # =====================================================

    def get_symbol_agent(
        self,
        symbol: str,
    ) -> SymbolAgent:

        key = str(symbol).replace(" ","")

        if key not in self.symbol_agents:
            raise KeyError(f"Unknown Symbol-Agent: {key}")

        return self.symbol_agents[key]

    # =====================================================
    # Audit Access
    # =====================================================

    def get_last_symbol_output(
        self,
        symbol: str,
    ) -> SymbolAgentOutput:

        key = str(symbol).replace(" ","")

        if key not in self.last_symbol_outputs:
            raise KeyError(f"No previous decision for symbol: {key}")

        return self.last_symbol_outputs[key]

    # =====================================================
    # Lifecycle
    # =====================================================

    def reset(self) -> None:
        """
        Reset runtime state تمام Agent ها و Engine.
        Configuration و Policy تغییر نمی‌کنند.
        """
        for agent in self.symbol_agents.values():
            agent.reset()

        self.meta_agent.reset()
        self.last_symbol_outputs.clear()
        self.last_decision = None
        self.decision_count = 0

# ============================================================================= END