# f08_risk/tests_ch4_i8/risk_mode_policy_contract_tester_A.py (49)
#
# Run: pytest -v -s f08_risk/tests_ch4_i8/risk_mode_policy_contract_tester_A.py

"""
Tester A - RiskModePolicy Contract
"""

from __future__ import annotations

import pytest

from f08_risk.risk_mode_policy import (
    DEFAULT_RISK_MODE_POLICIES,
    RISK_MODES,
    RiskModePolicy,
    get_risk_mode_policy,
)


EXPECTED_MODES = {
    "train",
    "optimize",
    "backtest",
    "replay",
    "eval",
    "shadow",
    "paper",
    "live",
}


def test_official_modes_are_complete() -> None:
    assert set(RISK_MODES) == EXPECTED_MODES


def test_every_official_mode_has_a_policy() -> None:
    assert set(DEFAULT_RISK_MODE_POLICIES) == EXPECTED_MODES

    for mode in EXPECTED_MODES:
        policy = DEFAULT_RISK_MODE_POLICIES[mode]

        assert isinstance(policy, RiskModePolicy)
        assert policy.mode == mode


@pytest.mark.parametrize(
    "mode",
    sorted(EXPECTED_MODES),
)
def test_every_mode_enforces_full_risk(mode: str) -> None:
    policy = get_risk_mode_policy(mode)

    assert policy.enforce_hard_constraints is True
    assert policy.enforce_soft_constraints is True
    assert policy.enforce_stop_loss is True
    assert policy.record_audit is True


def test_mode_is_normalized() -> None:
    policy = RiskModePolicy(
        mode="  LIVE  ",
        enforce_hard_constraints=True,
        enforce_soft_constraints=True,
        enforce_stop_loss=True,
        record_audit=True,
    )

    assert policy.mode == "live"


def test_unsupported_mode_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported risk mode",
    ):
        RiskModePolicy(
            mode="unknown",
            enforce_hard_constraints=True,
            enforce_soft_constraints=True,
            enforce_stop_loss=True,
            record_audit=True,
        )


def test_empty_mode_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="mode is required",
    ):
        RiskModePolicy(
            mode="",
            enforce_hard_constraints=True,
            enforce_soft_constraints=True,
            enforce_stop_loss=True,
            record_audit=True,
        )


def test_hard_constraints_cannot_be_disabled() -> None:
    with pytest.raises(
        ValueError,
        match="Hard constraints cannot be disabled",
    ):
        RiskModePolicy(
            mode="live",
            enforce_hard_constraints=False,
            enforce_soft_constraints=True,
            enforce_stop_loss=True,
            record_audit=True,
        )


def test_policy_is_immutable() -> None:
    policy = get_risk_mode_policy("live")

    with pytest.raises(AttributeError):
        policy.mode = "paper"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    (
        "enforce_hard_constraints",
        "enforce_soft_constraints",
        "enforce_stop_loss",
        "record_audit",
    ),
)
def test_policy_flags_must_be_bool(field_name: str) -> None:
    values = {
        "enforce_hard_constraints": True,
        "enforce_soft_constraints": True,
        "enforce_stop_loss": True,
        "record_audit": True,
    }

    values[field_name] = 1  # type: ignore[assignment]

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be bool",
    ):
        RiskModePolicy(
            mode="live",
            **values,
        )

