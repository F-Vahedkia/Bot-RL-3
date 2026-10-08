# f08_risk/tests_ch4_i1_i2/risk_projection_symbol_position_count_contract_tester_A.py (23)
#
# Run: pytest -v -s f08_risk/tests_ch4_i1_i2/risk_projection_symbol_position_count_contract_tester_A.py

"""
Purpose:
    Validate that RiskProjection preserves and projects the explicit
    per-symbol position-count contract from SymbolRiskSnapshot.

Contract:
    - ProjectedSymbolRisk.current_position_count
    - ProjectedSymbolRisk.projected_position_count

Projection semantics:
    1. Existing symbol + non-zero target:
       projected count remains the current symbol position count.

    2. Existing symbol + zero target:
       projected count becomes zero.

    3. Flat symbol + non-zero target:
       projected count becomes one.

    4. Portfolio projected_position_count:
       equals the sum of per-symbol projected position counts.

Important:
  target_exposure represents a target net state, not individual order/ticket
  decomposition. Therefore projection does not invent additional tickets when
  an existing position is resized.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioDecision,
)
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.risk_projection import (
    ProjectedSymbolRisk,
    RiskProjection,
)


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
    experiment_id="exp-projection-position-count",
)


def make_symbol_snapshot(
    symbol: str,
    *,
    exposure: float,
    position_count: int,
) -> SymbolRiskSnapshot:
    if exposure == 0.0:
        return SymbolRiskSnapshot(
            symbol=symbol,
            exposure=0.0,
            notional=0.0,
            used_margin=0.0,
            current_lots=0.0,
            current_side=0,
            position_count=position_count,
        )

    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=exposure,
        notional=1_000.0,
        used_margin=250.0,
        current_lots=0.20,
        current_side=1 if exposure > 0.0 else -1,
        position_count=position_count,
    )


def make_context(
    symbols: dict[str, SymbolRiskSnapshot],
) -> RiskContext:
    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
            leverage=100.0,
            peak_equity=10_000.0,
            day_start_equity=10_000.0,
            open_position_count=sum(
                snapshot.position_count
                for snapshot in symbols.values()
            ),
        ),
        symbols=symbols,
        correlation={
            symbol: {
                other: (
                    1.0
                    if symbol == other
                    else 0.0
                )
                for other in symbols
            }
            for symbol in symbols
        },
    )


def make_decision(
    target_exposure: dict[str, float],
) -> PortfolioDecision:
    return PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="projection-position-count-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )


def test_projected_symbol_risk_exposes_position_count_fields() -> None:
    field_names = {
        field.name
        for field in fields(ProjectedSymbolRisk)
    }

    assert "current_position_count" in field_names
    assert "projected_position_count" in field_names


def test_existing_multi_position_count_is_preserved() -> None:
    context = make_context(
        {
            "XAUUSD": make_symbol_snapshot(
                "XAUUSD",
                exposure=0.10,
                position_count=3,
            )
        }
    )

    projected = RiskProjection().project(
        context=context,
        decision=make_decision(
            {
                "XAUUSD": 0.15,
            }
        ),
    )

    snapshot = projected.symbols["XAUUSD"]

    assert snapshot.current_position_count == 3
    assert snapshot.projected_position_count == 3


def test_closing_symbol_projects_zero_position_count() -> None:
    context = make_context(
        {
            "XAUUSD": make_symbol_snapshot(
                "XAUUSD",
                exposure=0.10,
                position_count=3,
            )
        }
    )

    projected = RiskProjection().project(
        context=context,
        decision=make_decision(
            {
                "XAUUSD": 0.0,
            }
        ),
    )

    snapshot = projected.symbols["XAUUSD"]

    assert snapshot.current_position_count == 3
    assert snapshot.projected_position_count == 0


def test_flat_symbol_opening_projects_one_position() -> None:
    context = make_context(
        {
            "XAUUSD": make_symbol_snapshot(
                "XAUUSD",
                exposure=0.0,
                position_count=0,
            )
        }
    )

    projected = RiskProjection().project(
        context=context,
        decision=make_decision(
            {
                "XAUUSD": 0.10,
            }
        ),
    )

    snapshot = projected.symbols["XAUUSD"]

    assert snapshot.current_position_count == 0
    assert snapshot.projected_position_count == 1


def test_portfolio_projected_position_count_is_sum_of_symbol_counts() -> None:
    context = make_context(
        {
            "XAUUSD": make_symbol_snapshot(
                "XAUUSD",
                exposure=0.10,
                position_count=3,
            ),
            "EURUSD": make_symbol_snapshot(
                "EURUSD",
                exposure=0.20,
                position_count=2,
            ),
        }
    )

    projected = RiskProjection().project(
        context=context,
        decision=make_decision(
            {
                "XAUUSD": 0.10,
                "EURUSD": 0.20,
            }
        ),
    )

    assert projected.projected_position_count == 5

    assert (
        sum(
            snapshot.projected_position_count
            for snapshot in projected.symbols.values()
        )
        == 5
    )