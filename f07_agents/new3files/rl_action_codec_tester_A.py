# f07_agents/rl_action_codec_tester_A.py
#
# Tests for backend-independent RL action encoding/decoding.

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from f07_agents.contracts import (
    AgentObservation,
    DecisionMode,
    ModelIdentity,
    PortfolioContext,
    PolicyOutput,
    SymbolAgentOutput,
)
from f07_agents.rl_action_codec import (
    DiscreteSymbolActionCodec,
    DiscreteSymbolActionCodecConfig,
    MappingMetaActionCodec,
)
from f07_agents.contracts import SymbolContext


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
MODEL = ModelIdentity("m", "1", "p1")


def observation():
    return AgentObservation(
        symbol="XAUUSD",
        base_tf="M1",
        timestamp=TIMESTAMP,
        values=(1.0, 2.0, 3.0),
        feature_names=("f1", "f2", "f3"),
    )


def symbol_context(stop=100.5):
    return SymbolContext(
        symbol="XAUUSD",
        current_side=0,
        current_lots=0.0,
        exposure=0.0,
        drawdown=0.0,
        volatility=0.1,
        local_risk_score=0.1,
        metadata={"proposed_stop_price": stop},
    )


def portfolio():
    return PortfolioContext(
        timestamp=TIMESTAMP,
        equity=10000.0,
        balance=10000.0,
        used_margin=1000.0,
        free_margin=9000.0,
        margin_level=10.0,
        drawdown=0.0,
        daily_drawdown=0.0,
        mode=DecisionMode.BACKTEST,
    )


def signal_output(symbol, signal):
    return SymbolAgentOutput(
        symbol=symbol,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        signal=signal,
        confidence=0.9,
        expected_return=0.02 * signal,
        risk_score=0.1,
        desired_exposure=0.2 if signal else 0.0,
        stop_price=101.0 if signal else None,
        model=MODEL,
        decision_id="d1",
    )


# -----------------------------------------------------------------------------
# Symbol codec config
# -----------------------------------------------------------------------------
def test_config_defaults_match_three_level_action_contract():
    cfg = DiscreteSymbolActionCodecConfig()
    assert cfg.actions == ("short", "hold", "long")
    assert cfg.levels == (-1, 0, 1)


def test_config_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        DiscreteSymbolActionCodecConfig(actions=("short", "long"))


def test_config_rejects_duplicate_levels():
    with pytest.raises(ValueError, match="levels must be unique"):
        DiscreteSymbolActionCodecConfig(levels=(-1, -1, 1))


def test_config_requires_minus_one_zero_one_levels():
    with pytest.raises(ValueError, match="contain -1, 0 and 1"):
        DiscreteSymbolActionCodecConfig(levels=(-2, 0, 2))


def test_config_rejects_unknown_stop_source():
    with pytest.raises(ValueError, match="stop_price_source"):
        DiscreteSymbolActionCodecConfig(stop_price_source="unknown")


def test_config_requires_stop_price_key():
    with pytest.raises(ValueError, match="stop_price_key"):
        DiscreteSymbolActionCodecConfig(stop_price_key=" ")


def test_config_requires_all_signal_exposures():
    with pytest.raises(ValueError, match="missing signal 1"):
        DiscreteSymbolActionCodecConfig(exposure_by_signal={-1: 0.2, 0: 0.0})


def test_config_rejects_invalid_exposure():
    with pytest.raises(ValueError, match=r"must be finite and in \[0, 1\]"):
        DiscreteSymbolActionCodecConfig(
            exposure_by_signal={-1: 0.2, 0: 0.0, 1: 1.2}
        )


# -----------------------------------------------------------------------------
# Discrete decoding
# -----------------------------------------------------------------------------
def test_short_index_decodes_to_short_policy_output():
    codec = DiscreteSymbolActionCodec()
    out = codec.decode(0, observation(), symbol_context())
    assert isinstance(out, PolicyOutput)
    assert out.signal == -1
    assert out.desired_exposure == pytest.approx(0.2)
    assert out.stop_price == pytest.approx(100.5)
    assert out.metadata["action_label"] == "short"


def test_hold_index_decodes_to_flat_policy_output():
    codec = DiscreteSymbolActionCodec()
    out = codec.decode(1, observation(), symbol_context())
    assert out.signal == 0
    assert out.desired_exposure == 0.0
    assert out.stop_price is None


def test_long_index_decodes_to_long_policy_output():
    codec = DiscreteSymbolActionCodec()
    out = codec.decode(np.array([2]), observation(), symbol_context())
    assert out.signal == 1
    assert out.desired_exposure == pytest.approx(0.2)
    assert out.stop_price == pytest.approx(100.5)


def test_codec_call_matches_decode():
    codec = DiscreteSymbolActionCodec()
    assert codec(1, observation(), symbol_context()) == codec.decode(
        1, observation(), symbol_context()
    )


def test_policy_output_is_passed_through_unchanged():
    codec = DiscreteSymbolActionCodec()
    policy = PolicyOutput(
        signal=1,
        confidence=0.7,
        expected_return=0.01,
        risk_score=0.2,
        desired_exposure=0.3,
        stop_price=123.4,
    )
    assert codec.decode(policy, observation(), symbol_context()) is policy


def test_non_numeric_action_is_rejected():
    with pytest.raises(ValueError, match="must be numeric|cannot convert"):
        DiscreteSymbolActionCodec().decode("long", observation(), symbol_context())


def test_multi_value_discrete_action_is_rejected():
    with pytest.raises(ValueError, match="exactly one value"):
        DiscreteSymbolActionCodec().decode([0, 1], observation(), symbol_context())


def test_non_integer_discrete_action_is_rejected():
    with pytest.raises(ValueError, match="must be an integer"):
        DiscreteSymbolActionCodec().decode(1.5, observation(), symbol_context())


def test_unknown_discrete_action_index_is_rejected():
    with pytest.raises(ValueError, match="Unknown discrete Symbol action index"):
        DiscreteSymbolActionCodec().decode(3, observation(), symbol_context())


def test_non_finite_discrete_action_is_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        DiscreteSymbolActionCodec().decode(np.nan, observation(), symbol_context())


def test_non_flat_action_requires_stop_price_metadata():
    ctx = symbol_context()
    ctx.metadata.pop("proposed_stop_price")
    with pytest.raises(ValueError, match="does not contain"):
        DiscreteSymbolActionCodec().decode(0, observation(), ctx)


def test_non_flat_action_rejects_invalid_stop_price():
    with pytest.raises(ValueError, match="finite and > 0"):
        DiscreteSymbolActionCodec().decode(
            2,
            observation(),
            symbol_context(stop=0.0),
        )


def test_scalar_non_flat_action_requires_context_metadata_source():
    codec = DiscreteSymbolActionCodec(
        DiscreteSymbolActionCodecConfig(stop_price_source="action_payload")
    )
    with pytest.raises(ValueError, match="context_metadata"):
        codec.decode(2, observation(), symbol_context())


def test_mapping_symbol_action_decodes_to_policy_output():
    payload = {
        "signal": 1,
        "confidence": 0.8,
        "expected_return": 0.03,
        "risk_score": 0.2,
        "desired_exposure": 0.4,
        "stop_price": 150.0,
        "metadata": {"source": "test"},
    }
    out = DiscreteSymbolActionCodec().decode(payload, observation(), symbol_context())
    assert out.signal == 1
    assert out.desired_exposure == pytest.approx(0.4)
    assert out.stop_price == pytest.approx(150.0)
    assert out.metadata == {"source": "test"}


def test_mapping_symbol_action_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing fields"):
        DiscreteSymbolActionCodec().decode(
            {"signal": 1}, observation(), symbol_context()
        )


# -----------------------------------------------------------------------------
# Meta mapping codec
# -----------------------------------------------------------------------------
def test_meta_mapping_codec_decodes_full_payload():
    signals = {"XAUUSD": signal_output("XAUUSD", 1)}
    payload = {
        "capital_allocation": {"XAUUSD": 0.2},
        "target_signals": {"XAUUSD": 1},
        "target_exposure": {"XAUUSD": 0.2},
        "margin_allocation": {"XAUUSD": 2000.0},
        "portfolio_risk": 0.3,
        "reason_codes": ["test"],
        "target_stop_price": {"XAUUSD": 100.0},
    }
    out = MappingMetaActionCodec().decode(payload, signals, portfolio())
    assert isinstance(out, type(__import__("f07_agents.contracts", fromlist=["MetaPolicyOutput"]).MetaPolicyOutput))
    assert out.target_signals == {"XAUUSD": 1}
    assert out.target_exposure["XAUUSD"] == pytest.approx(0.2)
    assert out.target_stop_price["XAUUSD"] == pytest.approx(100.0)
    assert out.reason_codes == ("test",)


def test_meta_mapping_codec_passes_existing_output_through():
    from f07_agents.contracts import MetaPolicyOutput

    obj = MetaPolicyOutput(
        capital_allocation={"XAUUSD": 0.2},
        target_signals={"XAUUSD": 1},
        target_exposure={"XAUUSD": 0.2},
        margin_allocation={"XAUUSD": 2000.0},
        portfolio_risk=0.3,
    )
    assert MappingMetaActionCodec().decode(obj, {"XAUUSD": signal_output("XAUUSD", 1)}, portfolio()) is obj


def test_meta_mapping_codec_rejects_non_mapping():
    with pytest.raises(TypeError, match="mapping"):
        MappingMetaActionCodec().decode([], {}, portfolio())


def test_meta_mapping_codec_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing fields"):
        MappingMetaActionCodec().decode(
            {"target_signals": {"XAUUSD": 1}},
            {"XAUUSD": signal_output("XAUUSD", 1)},
            portfolio(),
        )


def test_meta_mapping_codec_normalizes_types():
    payload = {
        "capital_allocation": {"XAUUSD": "0.2"},
        "target_signals": {"XAUUSD": "-1"},
        "target_exposure": {"XAUUSD": "-0.2"},
        "margin_allocation": {"XAUUSD": "2000"},
        "portfolio_risk": "0.4",
    }
    out = MappingMetaActionCodec().decode(payload, {"XAUUSD": signal_output("XAUUSD", -1)}, portfolio())
    assert out.target_signals["XAUUSD"] == -1
    assert out.target_exposure["XAUUSD"] == pytest.approx(-0.2)
    assert out.margin_allocation["XAUUSD"] == "2000"
    assert out.portfolio_risk == pytest.approx(0.4)
