# f08_risk/tests_ch4_i5/net_exposure_direction_contract_tester_A.py (30)
#
# Run: pytest -v -s f08_risk/tests_ch4_i5/net_exposure_direction_contract_tester_A.py

# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Purpose
# -------
# Contract tester for Net Exposure / Direction semantics.
#
# Current contract under test:
#
#     target_exposure > 0  -> Long
#     target_exposure < 0  -> Short
#     target_exposure == 0 -> Flat
#
# Direction is conceptually derived from signed net exposure.
#
# Important:
# ----------
# This tester intentionally does NOT modify RiskEngine or RiskProjection.
# It establishes the contract before production implementation.


from __future__ import annotations

from datetime import datetime, timezone

import pytest

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioDecision,
)

from f08_risk.risk_context import SymbolRiskSnapshot


TS = datetime(
    2026,
    1,
    5,
    12,
    0,
    tzinfo=timezone.utc,
)


MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-net-exposure-direction",
)


def make_decision(
    target_exposure: float,
) -> PortfolioDecision:
    return PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="net-exposure-direction",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": target_exposure,
        },
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )


def make_snapshot(
    exposure: float,
    current_side: int,
) -> SymbolRiskSnapshot:
    return SymbolRiskSnapshot(
        symbol="XAUUSD",
        exposure=exposure,
        notional=1_000.0 if exposure != 0.0 else 0.0,
        used_margin=250.0 if exposure != 0.0 else 0.0,
        current_lots=0.05 if exposure != 0.0 else 0.0,
        current_side=current_side,
        position_count=1 if exposure != 0.0 else 0,
    )


# =============================================================================
# Target Exposure Contract
# =============================================================================


def test_positive_target_exposure_is_preserved() -> None:
    decision = make_decision(0.25)

    assert decision.target_exposure["XAUUSD"] == pytest.approx(0.25)
    assert decision.target_exposure["XAUUSD"] > 0.0


def test_negative_target_exposure_is_preserved() -> None:
    decision = make_decision(-0.25)

    assert decision.target_exposure["XAUUSD"] == pytest.approx(-0.25)
    assert decision.target_exposure["XAUUSD"] < 0.0


def test_zero_target_exposure_is_preserved() -> None:
    decision = make_decision(0.0)

    assert decision.target_exposure["XAUUSD"] == pytest.approx(0.0)


# =============================================================================
# Direction Semantics
# =============================================================================


def test_positive_exposure_means_long_direction() -> None:
    exposure = 0.25

    assert (
        "long"
        if exposure > 0.0
        else "short"
        if exposure < 0.0
        else "flat"
    ) == "long"


def test_negative_exposure_means_short_direction() -> None:
    exposure = -0.25

    assert (
        "long"
        if exposure > 0.0
        else "short"
        if exposure < 0.0
        else "flat"
    ) == "short"


def test_zero_exposure_means_flat_direction() -> None:
    exposure = 0.0

    assert (
        "long"
        if exposure > 0.0
        else "short"
        if exposure < 0.0
        else "flat"
    ) == "flat"


# =============================================================================
# Current Side Representation
# =============================================================================


def test_positive_current_exposure_uses_positive_current_side() -> None:
    snapshot = make_snapshot(
        exposure=0.20,
        current_side=1,
    )

    assert snapshot.exposure > 0.0
    assert snapshot.current_side == 1


def test_negative_current_exposure_uses_negative_current_side() -> None:
    snapshot = make_snapshot(
        exposure=-0.20,
        current_side=-1,
    )

    assert snapshot.exposure < 0.0
    assert snapshot.current_side == -1


def test_zero_current_exposure_uses_flat_current_side() -> None:
    snapshot = make_snapshot(
        exposure=0.0,
        current_side=0,
    )

    assert snapshot.exposure == pytest.approx(0.0)
    assert snapshot.current_side == 0


# =============================================================================
# Transition Semantics
# =============================================================================


@pytest.mark.parametrize(
    "current,target,expected",
    [
        (0.00, 0.20, "OPEN_LONG"),
        (0.00, -0.20, "OPEN_SHORT"),
        (0.20, 0.00, "CLOSE_LONG"),
        (-0.20, 0.00, "CLOSE_SHORT"),
        (0.20, 0.35, "ADJUST_LONG"),
        (0.35, 0.20, "ADJUST_LONG"),
        (-0.20, -0.35, "ADJUST_SHORT"),
        (-0.35, -0.20, "ADJUST_SHORT"),
        (0.20, -0.20, "REVERSE"),
        (-0.20, 0.20, "REVERSE"),
        (0.20, 0.20, "MAINTAIN_LONG"),
        (-0.20, -0.20, "MAINTAIN_SHORT"),
        (0.00, 0.00, "FLAT"),
    ],
)
def test_net_exposure_transition_semantics(
    current: float,
    target: float,
    expected: str,
) -> None:
    if current == 0.0 and target > 0.0:
        actual = "OPEN_LONG"
    elif current == 0.0 and target < 0.0:
        actual = "OPEN_SHORT"
    elif current > 0.0 and target == 0.0:
        actual = "CLOSE_LONG"
    elif current < 0.0 and target == 0.0:
        actual = "CLOSE_SHORT"
    elif current > 0.0 and target > 0.0:
        actual = (
            "MAINTAIN_LONG"
            if current == target
            else "ADJUST_LONG"
        )
    elif current < 0.0 and target < 0.0:
        actual = (
            "MAINTAIN_SHORT"
            if current == target
            else "ADJUST_SHORT"
        )
    elif current > 0.0 and target < 0.0:
        actual = "REVERSE"
    elif current < 0.0 and target > 0.0:
        actual = "REVERSE"
    else:
        actual = "FLAT"

    assert actual == expected


# =============================================================================
# Current Contract Gap
# =============================================================================


def test_current_snapshot_can_represent_exposure_side_mismatch() -> None:
    """
    Discovery test only.

    The current SymbolRiskSnapshot contract exposes both:
        exposure
        current_side

    but the current contract does not yet establish that they must be
    mathematically consistent.

    This test documents that gap without changing production behavior.
    """

    snapshot = make_snapshot(
        exposure=0.20,
        current_side=-1,
    )

    assert snapshot.exposure > 0.0
    assert snapshot.current_side == -1

