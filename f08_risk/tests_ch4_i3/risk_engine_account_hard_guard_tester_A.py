# f08_risk/risk_engine_account_hard_guard_tester_A.py (28)
#
# Run: python -m f08_risk.risk_engine_account_hard_guard_tester_A
#      pytest -v -s f08_risk/risk_engine_account_hard_guard_tester_A.py

# Chapter 4 - Risk Layer
#
# Account-level hard safety guards.
#
# Scope
# -----
# Validate the current hard-guard contract for:
#     - usable equity / capital
#     - current free margin validity
#     - current margin level validity
#     - account leverage validity
#     - maximum account leverage
#     - projected free margin
#
# Important
# ---------
# Current free_margin == 0 is NOT rejected by the account-state guard
# by itself. A risk-reducing decision must still be allowed to proceed.
#
# Projected free_margin <= 0 is handled separately by the final projected
# safety guard in RiskEngine.


from __future__ import annotations

from datetime import datetime, timezone
import pytest
from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioContext,
    PortfolioDecision,
)
from f08_risk.contracts import (
    RiskDecisionStatus,
    RiskRequest,
)
from f08_risk.limits import RiskLimits
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.risk_engine import RiskEngine


# =============================================================================
# Shared test data
# =============================================================================

TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-account-hard-guard",
)


# =============================================================================
# Helpers
# =============================================================================

def make_account(
    *,
    equity: float = 10_000.0,
    free_margin: float = 9_000.0,
    used_margin: float = 1_000.0,
    margin_level: float = 1_000.0,
    leverage: float = 100.0,
    open_position_count: int = 1,
) -> AccountRiskSnapshot:

    return AccountRiskSnapshot(
        balance=equity,
        equity=equity,
        used_margin=used_margin,
        free_margin=free_margin,
        margin_level=margin_level,
        leverage=leverage,
        peak_equity=equity,
        day_start_equity=equity,
        open_position_count=open_position_count,
    )


def make_symbol(
    *,
    symbol: str = "EURUSD",
    exposure: float = 0.10,
    notional: float = 1_000.0,
    used_margin: float = 250.0,
    current_lots: float = 0.05,
    current_side: int = 1,
    position_count: int = 1,
) -> SymbolRiskSnapshot:

    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=exposure,
        notional=notional,
        used_margin=used_margin,
        current_lots=current_lots,
        current_side=current_side,
        position_count=position_count,
    )


def make_context(
    *,
    account: AccountRiskSnapshot | None = None,
    symbols: dict[str, SymbolRiskSnapshot] | None = None,
) -> RiskContext:

    actual_account = (
        account
        if account is not None
        else make_account()
    )

    actual_symbols = (
        symbols
        if symbols is not None
        else {
            "EURUSD": make_symbol(),
        }
    )

    correlation = {
        symbol: {
            other: (
                1.0
                if symbol == other
                else 0.0
            )
            for other in actual_symbols
        }
        for symbol in actual_symbols
    }

    return RiskContext(
        timestamp=TS,
        account=actual_account,
        symbols=actual_symbols,
        correlation=correlation,
    )


def make_portfolio(
    *,
    account: AccountRiskSnapshot,
    context: RiskContext,
) -> PortfolioContext:

    return PortfolioContext(
        timestamp=TS,
        equity=account.equity,
        balance=account.balance,
        used_margin=account.used_margin,
        free_margin=account.free_margin,
        margin_level=account.margin_level,
        drawdown=0.0,
        daily_drawdown=0.0,
        exposure={
            symbol: snapshot.exposure
            for symbol, snapshot
            in context.symbols.items()
        },
        concentration={},
        correlation=context.correlation,
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )


def make_request(
    *,
    context: RiskContext,
    target_exposure: dict[str, float] | None = None,
) -> RiskRequest:

    account = context.account

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="account-hard-guard-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure=(
            target_exposure
            if target_exposure is not None
            else {"EURUSD": 0.10}
        ),
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    return RiskRequest(
        decision=decision,
        portfolio=make_portfolio(
            account=account,
            context=context,
        ),
        risk_context=context,
    )


# =============================================================================
# RiskLimits contract
# =============================================================================

def test_max_account_leverage_none_disables_guard() -> None:

    limits = RiskLimits(
        max_account_leverage=None,
    )
    assert limits.max_account_leverage is None


def test_max_account_leverage_accepts_positive_value() -> None:

    limits = RiskLimits(
        max_account_leverage=100.0,
    )
    assert limits.max_account_leverage == 100.0


def test_max_account_leverage_rejects_zero() -> None:

    with pytest.raises(ValueError, match="max_account_leverage"):
        RiskLimits(
            max_account_leverage=0.0,
        )


def test_max_account_leverage_rejects_negative() -> None:

    with pytest.raises(ValueError, match="max_account_leverage"):
        RiskLimits(
            max_account_leverage=-1.0,
        )


def test_max_account_leverage_rejects_nan() -> None:

    with pytest.raises(ValueError, match="max_account_leverage"):
        RiskLimits(
            max_account_leverage=float("nan"),
        )


def test_max_account_leverage_rejects_positive_infinity() -> None:

    with pytest.raises(ValueError, match="max_account_leverage"):
        RiskLimits(
            max_account_leverage=float("inf"),
        )


# =============================================================================
# Projected-path account hard guards
# =============================================================================

def test_positive_equity_is_allowed() -> None:

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=9_000.0,
            used_margin=1_000.0,
            margin_level=1_000.0,
            leverage=100.0,
        )
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(context=context)
    )

    assert result.status == RiskDecisionStatus.APPROVED

def test_zero_equity_is_rejected() -> None:

    account = make_account(
        equity=0.0,
        free_margin=0.0,
        used_margin=0.0,
        margin_level=0.0,
    )

    context = make_context(
        account=account,
        symbols={
            "EURUSD": make_symbol(
                exposure=0.0,
                notional=0.0,
                used_margin=0.0,
                current_lots=0.0,
                current_side=0,
                position_count=0,
            )
        },
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(
            context=context,
            target_exposure={"EURUSD": 0.0},
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED

    assert any(
        violation.code == "NO_USABLE_CAPITAL"
        for violation in result.violations
    )

# def test_negative_equity_is_rejected_by_account_contract() -> None:

#     with pytest.raises(ValueError, match="equity"):
#         make_account(
#             equity=-1.0,
#             free_margin=0.0,
#             used_margin=0.0,
#         )


def test_negative_current_free_margin_is_rejected() -> None:

    with pytest.raises(ValueError, match="free_margin"):
        make_account(
            equity=10_000.0,
            free_margin=-1.0,
            used_margin=10_001.0,
        )


def test_zero_current_free_margin_is_allowed_at_account_guard() -> None:

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=0.0,
            used_margin=10_000.0,
            margin_level=100.0,
            leverage=100.0,
        )
    )

    # Risk-reducing / flat decision.
    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(
            context=context,
            target_exposure={"EURUSD": 0.0},
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED


def test_negative_margin_level_is_rejected_by_context_contract() -> None:

    with pytest.raises(ValueError, match="margin_level"):
        make_account(
            equity=10_000.0,
            free_margin=9_000.0,
            used_margin=1_000.0,
            margin_level=-1.0,
        )


def test_zero_margin_level_with_used_margin_is_rejected() -> None:

    # Context itself accepts margin_level == 0.
    # RiskEngine must reject it because used margin is positive.

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=9_000.0,
            used_margin=1_000.0,
            margin_level=0.0,
            leverage=100.0,
        )
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(context=context)
    )

    assert result.status == RiskDecisionStatus.REJECTED

    assert any(
        violation.code == "INVALID_MARGIN_LEVEL"
        for violation in result.violations
    )


def test_zero_margin_level_without_used_margin_is_allowed() -> None:

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=10_000.0,
            used_margin=0.0,
            margin_level=0.0,
            leverage=100.0,
            open_position_count=0,
        ),
        symbols={
            "EURUSD": make_symbol(
                exposure=0.0,
                notional=0.0,
                used_margin=0.0,
                current_lots=0.0,
                current_side=0,
                position_count=0,
            )
        },
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(
            context=context,
            target_exposure={"EURUSD": 0.0},
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED


def test_zero_account_leverage_is_rejected_by_account_contract() -> None:

    with pytest.raises(ValueError, match="leverage"):
        make_account(
            leverage=0.0,
        )


def test_negative_account_leverage_is_rejected_by_account_contract() -> None:

    with pytest.raises(ValueError, match="leverage"):
        make_account(
            leverage=-1.0,
        )


def test_account_leverage_equal_to_limit_is_allowed() -> None:

    context = make_context(
        account=make_account(
            leverage=100.0,
        )
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(context=context)
    )

    assert result.status == RiskDecisionStatus.APPROVED


def test_account_leverage_above_limit_is_rejected() -> None:

    context = make_context(
        account=make_account(
            leverage=101.0,
        )
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(context=context)
    )

    assert result.status == RiskDecisionStatus.REJECTED

    assert any(
        violation.code == "MAX_ACCOUNT_LEVERAGE"
        for violation in result.violations
    )


def test_account_leverage_guard_can_be_disabled() -> None:

    context = make_context(
        account=make_account(
            leverage=500.0,
        )
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=None,
        )
    ).evaluate(
        make_request(context=context)
    )

    assert result.status == RiskDecisionStatus.APPROVED


# =============================================================================
# Projected free-margin guard
# =============================================================================

def test_projected_free_margin_positive_is_allowed() -> None:

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=9_000.0,
            used_margin=1_000.0,
        )
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(
            context=context,
            target_exposure={"EURUSD": 0.10},
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED


def test_projected_free_margin_zero_is_rejected() -> None:

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=9_000.0,
            used_margin=1_000.0,
        ),
        symbols={
            "EURUSD": make_symbol(
                exposure=0.10,
                notional=9_000.0,
                used_margin=1_000.0,
                current_lots=0.10,
                current_side=1,
                position_count=1,
            )
        },
    )

    # Make the target exposure large enough so that the projected
    # normalized margin consumes the complete equity.

    result = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_margin_utilization=1.0,
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(
            context=context,
            target_exposure={"EURUSD": 1.0},
        )
    )
    assert result.status == RiskDecisionStatus.REJECTED
    assert any(
        violation.code == "NO_FREE_MARGIN"
        for violation in result.violations
    )


def test_projected_free_margin_negative_is_rejected() -> None:

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=9_000.0,
            used_margin=1_000.0,
        ),
        symbols={
            "EURUSD": make_symbol(
                exposure=0.10,
                notional=9_000.0,
                used_margin=1_000.0,
                current_lots=0.10,
                current_side=1,
                position_count=1,
            )
        },
    )

    # The current projection implementation clamps projected_free_margin
    # to >= 0. Therefore this scenario still resolves to the zero boundary.

    result = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=1.0,
            max_margin_utilization=1.0,
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(
            context=context,
            target_exposure={"EURUSD": 2.0},
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED

    assert any(
        violation.code == "NO_FREE_MARGIN"
        for violation in result.violations
    )


# =============================================================================
# Projected risk-reduction scenario
# =============================================================================

def test_zero_current_free_margin_does_not_block_risk_reducing_close() -> None:

    context = make_context(
        account=make_account(
            equity=10_000.0,
            free_margin=0.0,
            used_margin=10_000.0,
            margin_level=100.0,
            leverage=100.0,
        ),
        symbols={
            "EURUSD": make_symbol(
                exposure=0.10,
                notional=10_000.0,
                used_margin=10_000.0,
                current_lots=1.0,
                current_side=1,
                position_count=1,
            )
        },
    )

    result = RiskEngine(
        limits=RiskLimits(
            max_account_leverage=100.0,
        )
    ).evaluate(
        make_request(
            context=context,
            target_exposure={"EURUSD": 0.0},
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.target_exposure["EURUSD"] == pytest.approx(0.0)


# =============================================================================
# Projected hard guard must remain hard
# =============================================================================

# def test_no_free_margin_is_critical_rejection() -> None:

#     context = make_context(
#         account=make_account(
#             equity=10_000.0,
#             free_margin=9_000.0,
#             used_margin=1_000.0,
#             margin_level=1_000.0,
#             leverage=100.0,
#         )
#     )

#     result = RiskEngine(
#         limits=RiskLimits(
#             max_symbol_exposure=1.0,
#             max_margin_utilization=1.0,
#             max_account_leverage=100.0,
#         )
#     ).evaluate(
#         make_request(
#             context=context,
#             target_exposure={"EURUSD": 1.0},
#         )
#     )

#     assert result.status == RiskDecisionStatus.REJECTED

#     matching = [
#         violation
#         for violation in result.violations
#         if violation.code == "NO_FREE_MARGIN"
#     ]

#     assert matching
#     assert matching[-1].severity == "critical"


# =============================================================================
# Runner
# =============================================================================

def _run_all_tests() -> None:

    tests = [
        test_max_account_leverage_none_disables_guard,
        test_max_account_leverage_accepts_positive_value,
        test_max_account_leverage_rejects_zero,
        test_max_account_leverage_rejects_negative,
        test_max_account_leverage_rejects_nan,
        test_max_account_leverage_rejects_positive_infinity,
        test_positive_equity_is_allowed,
        test_zero_equity_is_rejected,
        # test_negative_equity_is_rejected_by_account_contract,
        test_negative_current_free_margin_is_rejected,
        test_zero_current_free_margin_is_allowed_at_account_guard,
        test_negative_margin_level_is_rejected_by_context_contract,
        test_zero_margin_level_with_used_margin_is_rejected,
        test_zero_margin_level_without_used_margin_is_allowed,
        test_zero_account_leverage_is_rejected_by_account_contract,
        test_negative_account_leverage_is_rejected_by_account_contract,
        test_account_leverage_equal_to_limit_is_allowed,
        test_account_leverage_above_limit_is_rejected,
        test_account_leverage_guard_can_be_disabled,
        test_projected_free_margin_positive_is_allowed,
        test_projected_free_margin_zero_is_rejected,
        test_projected_free_margin_negative_is_rejected,
        test_zero_current_free_margin_does_not_block_risk_reducing_close,
        # test_no_free_margin_is_critical_rejection,
    ]

    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"[PASS] {test.__name__}")

    print()
    print(f"{passed}/{len(tests)} tests green.")


if __name__ == "__main__":
    _run_all_tests()

# ============================================================================= END
