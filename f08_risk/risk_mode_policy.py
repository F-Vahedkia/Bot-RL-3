# f08_risk/risk_mode_policy.py (48)
#
# Created: 1405/06/22

"""
Bot-RL-3 v8
Chapter 4 - Item 8
Risk Mode Policy Contract
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# =============================================================================
# Official Risk Modes
# =============================================================================

RISK_MODES = frozenset(
    {
        "train",
        "optimize",
        "backtest",
        "replay",
        "eval",
        "shadow",
        "paper",
        "live",
    }
)

# =============================================================================
# Risk Mode Policy
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskModePolicy:
    """
    Immutable policy contract defining Risk behavior for one execution mode.

    Item 8 rule:
        No supported mode may bypass hard-risk enforcement.

    Mode-specific execution side effects remain outside f08_risk.
    """

    mode: str
    enforce_hard_constraints: bool
    enforce_soft_constraints: bool
    enforce_stop_loss: bool
    record_audit: bool

    def __post_init__(self) -> None:

        raw_mode = getattr(
            self.mode,
            "value",
            self.mode,
        )
        normalized_mode = str(
            raw_mode
        ).strip().lower()


        if not normalized_mode:
            raise ValueError("mode is required")

        if normalized_mode not in RISK_MODES:
            raise ValueError(
                f"Unsupported risk mode: {normalized_mode}"
            )

        object.__setattr__(
            self,
            "mode",
            normalized_mode,
        )

        for field_name in (
            "enforce_hard_constraints",
            "enforce_soft_constraints",
            "enforce_stop_loss",
            "record_audit",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be bool"
                )

        if not self.enforce_hard_constraints:
            raise ValueError(
                "Hard constraints cannot be disabled."
            )


# =============================================================================
# Official Default Policy Registry
# =============================================================================

DEFAULT_RISK_MODE_POLICIES: Mapping[str, RiskModePolicy] = {
    mode: RiskModePolicy(
        mode=mode,
        enforce_hard_constraints=True,
        enforce_soft_constraints=True,
        enforce_stop_loss=True,
        record_audit=True,
    )
    for mode in RISK_MODES
}


def get_risk_mode_policy(
    mode: str,
) -> RiskModePolicy:
    """
    Return the official immutable Risk policy for one mode.
    """
    raw_mode = getattr(
        mode,
        "value",
        mode,
    )
    normalized_mode = str(
        raw_mode
    ).strip().lower()

    if normalized_mode not in DEFAULT_RISK_MODE_POLICIES:
        raise ValueError(
            f"Unsupported risk mode: {normalized_mode}"
        )

    return DEFAULT_RISK_MODE_POLICIES[normalized_mode]


# ============================================================================= END