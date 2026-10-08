# f07_agents/tests/agent_factory_tester_A.py (t8)
#
# Run: pytest -v -s f07_agents/tests/agent_factory_tester_A.py
#
# Purpose:
#     Contract validation for agent_factory.py
# =============================================================================

from __future__ import annotations

import pytest

from f07_agents.agent_factory import AgentFactory, AgentFactoryConfig
from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PolicyOutput,
    MetaPolicyOutput,
)
from f07_agents.meta_agent import MetaAgent
from f07_agents.symbol_agent import SymbolAgent


class FixedSymbolPolicy:
    def predict(self, *, observation, context):
        signal = 1 if observation.symbol == "XAUUSD" else -1
        return PolicyOutput(
            signal=signal,
            confidence=0.90,
            expected_return=0.02 * signal,
            risk_score=0.10,
            desired_exposure=0.20,
            stop_price=1900.0 if signal > 0 else 2000.0,
        )


class FixedMetaPolicy:
    def predict(self, *, signals, portfolio):
        return MetaPolicyOutput(
            capital_allocation={symbol: 0.20 for symbol in signals},
            target_signals={symbol: value.signal for symbol, value in signals.items()},
            target_exposure={symbol: 0.20 * value.signal for symbol, value in signals.items()},
            margin_allocation={symbol: 2_000.0 for symbol in signals},
            portfolio_risk=0.20,
            reason_codes=("factory-test",),
        )


def model(name: str) -> ModelIdentity:
    return ModelIdentity(
        model_name=name,
        model_version="1.0.0",
        policy_version="policy-1",
        config_version="cfg-1",
        experiment_id="factory-test",
    )


def test_config_normalizes_symbols_and_rejects_duplicates():
    config = AgentFactoryConfig(
        symbols=("XAU USD", "eurusd"),
        mode=DecisionMode.BACKTEST,
    )

    assert config.symbols == ("XAUUSD", "EURUSD")

    with pytest.raises(ValueError, match="unique"):
        AgentFactoryConfig(
            symbols=("XAUUSD", "XAU USD"),
            mode=DecisionMode.BACKTEST,
        )


def test_config_validates_limits_and_mode():
    with pytest.raises(TypeError, match="DecisionMode"):
        AgentFactoryConfig(symbols=("XAUUSD",), mode="backtest")

    with pytest.raises(ValueError, match="max_total_exposure"):
        AgentFactoryConfig(
            symbols=("XAUUSD",),
            mode=DecisionMode.BACKTEST,
            max_total_exposure=1.1,
        )


def test_create_symbol_agents_uses_real_symbol_agent_constructor():
    factory = AgentFactory(
        config=AgentFactoryConfig(
            symbols=("XAU USD", "EURUSD"),
            mode=DecisionMode.BACKTEST,
        )
    )

    agents = factory.create_symbol_agents(
        policies={
            "XAUUSD": FixedSymbolPolicy(),
            "EURUSD": FixedSymbolPolicy(),
        },
        models={
            "XAUUSD": model("symbol-xau"),
            "EURUSD": model("symbol-eur"),
        },
    )

    assert set(agents) == {"XAUUSD", "EURUSD"}
    assert all(isinstance(agent, SymbolAgent) for agent in agents.values())
    assert agents["XAUUSD"].config.symbol == "XAUUSD"
    assert agents["EURUSD"].config.symbol == "EURUSD"
    assert agents["XAUUSD"].config.mode == DecisionMode.BACKTEST
    assert agents["EURUSD"].config.model == model("symbol-eur")


def test_create_symbol_agents_requires_exact_policy_and_model_symbol_sets():
    factory = AgentFactory(
        config=AgentFactoryConfig(
            symbols=("XAUUSD", "EURUSD"),
            mode=DecisionMode.BACKTEST,
        )
    )

    with pytest.raises(ValueError, match="Missing SymbolPolicy"):
        factory.create_symbol_agents(
            policies={"XAUUSD": FixedSymbolPolicy()},
            models={"XAUUSD": model("xau"), "EURUSD": model("eur")},
        )

    with pytest.raises(ValueError, match="unknown symbols"):
        factory.create_symbol_agents(
            policies={
                "XAUUSD": FixedSymbolPolicy(),
                "EURUSD": FixedSymbolPolicy(),
                "GBPUSD": FixedSymbolPolicy(),
            },
            models={
                "XAUUSD": model("xau"),
                "EURUSD": model("eur"),
            },
        )

    with pytest.raises(ValueError, match="Missing ModelIdentity"):
        factory.create_symbol_agents(
            policies={"XAUUSD": FixedSymbolPolicy(), "EURUSD": FixedSymbolPolicy()},
            models={"XAUUSD": model("xau")},
        )


def test_create_meta_agent_uses_real_meta_agent_constructor_and_limits():
    factory = AgentFactory(
        config=AgentFactoryConfig(
            symbols=("XAUUSD", "EURUSD"),
            mode=DecisionMode.LIVE,
            max_total_allocation=0.90,
            max_symbol_allocation=0.40,
            max_total_exposure=0.80,
        )
    )

    agent = factory.create_meta_agent(
        policy=FixedMetaPolicy(),
        model=model("meta"),
    )

    assert isinstance(agent, MetaAgent)
    assert agent.config.mode == DecisionMode.LIVE
    assert agent.config.model == model("meta")
    assert agent.config.max_total_allocation == pytest.approx(0.90)
    assert agent.config.max_symbol_allocation == pytest.approx(0.40)
    assert agent.config.max_total_exposure == pytest.approx(0.80)


def test_create_builds_both_layers_consistently():
    factory = AgentFactory(
        config=AgentFactoryConfig(
            symbols=("XAUUSD", "EURUSD"),
            mode=DecisionMode.REPLAY,
        )
    )

    agents, meta = factory.create(
        symbol_policies={
            "XAUUSD": FixedSymbolPolicy(),
            "EURUSD": FixedSymbolPolicy(),
        },
        symbol_models={
            "XAUUSD": model("xau"),
            "EURUSD": model("eur"),
        },
        meta_policy=FixedMetaPolicy(),
        meta_model=model("meta"),
    )

    assert set(agents) == {"XAUUSD", "EURUSD"}
    assert all(agent.config.mode == DecisionMode.REPLAY for agent in agents.values())
    assert meta.config.mode == DecisionMode.REPLAY


def test_create_rejects_missing_policy_or_model_objects():
    factory = AgentFactory(
        config=AgentFactoryConfig(
            symbols=("XAUUSD",),
            mode=DecisionMode.BACKTEST,
        )
    )

    with pytest.raises(ValueError, match="policies is required"):
        factory.create_symbol_agents(policies=None, models={"XAUUSD": model("xau")})

    with pytest.raises(ValueError, match="models is required"):
        factory.create_symbol_agents(policies={"XAUUSD": FixedSymbolPolicy()}, models=None)

    with pytest.raises(ValueError, match="policy is required"):
        factory.create_meta_agent(policy=None, model=model("meta"))

    with pytest.raises(ValueError, match="model is required"):
        factory.create_meta_agent(policy=FixedMetaPolicy(), model=None)
