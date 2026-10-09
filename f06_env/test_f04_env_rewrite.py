

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from f03_data.mtf_dataset import MTFDataset
from f03_data.instrument_specs import InstrumentSpec
from f06_env import (
    EnvironmentConfig,
    ExecutionCost,
    PortfolioAction,
    PositionIntent,
    TradingEnvironment,
    ExecutionSimulator,
)


def make_dataset(symbol: str = "EURUSD") -> MTFDataset:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(5)])
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            "high": [1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
            "low": [1.0990, 1.1000, 1.1010, 1.1020, 1.1030],
            "close": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            "volume": [10, 10, 10, 10, 10],
            "spread": [10, 10, 10, 10, 10],
        },
        index=idx,
    )
    ds = MTFDataset(symbol=symbol, base_tf="M1")
    ds.add("M1", frame)
    return ds


def instrument_spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="EURUSD",
        pip_size=0.0001,
        tick_size=0.00001,
        tick_value=1.0,
        tick_value_currency="USD",
        contract_size=100000.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=100.0,
        digits=5,
        point=0.00001,
        currency_base="EUR",
        currency_profit="USD",
        currency_margin="EUR",
        account_currency="USD",
        source="test",
    )


def make_env(
    *,
    execution: ExecutionSimulator | None = None,
) -> TradingEnvironment:
    ds = make_dataset()
    observations = pd.DataFrame(
        np.zeros((5, 2), dtype=np.float32),
        index=ds.get("M1").index,
        columns=["x", "y"],
    )
    return TradingEnvironment.from_mtf_datasets(
        observations={"EURUSD": observations},
        datasets={"EURUSD": ds},
        instruments={"EURUSD": instrument_spec()},
        config=EnvironmentConfig(leverage=100.0, window_size=1),
        execution=execution,
        execution_costs={"EURUSD": ExecutionCost(slippage_price=0.00002, commission=7.0)},
    )


def test_historical_quote_uses_bid_close_and_point_spread():
    env = make_env()
    quote = env.quotes["EURUSD"][0]
    assert quote.bid == pytest.approx(1.1000)
    assert quote.ask == pytest.approx(1.10010)


def test_open_uses_ask_not_mid_and_mark_long_on_bid():
    env = make_env()
    env.reset()
    result = env.step(PortfolioAction((PositionIntent("EURUSD", 1, 1.0),)))
    position = env.portfolio.get_position("EURUSD")
    assert position.entry_price == pytest.approx(1.10010 + 0.00002)
    assert result.info["used_margin"] > 0.0
    assert position.current_price == pytest.approx(1.1010)


def test_reduce_realizes_only_closed_volume():
    env = make_env()
    env.reset()
    env.step(PortfolioAction((PositionIntent("EURUSD", 1, 2.0),)))
    result = env.step(PortfolioAction((PositionIntent("EURUSD", 1, 1.0),)))
    assert result.info["execution"]["EURUSD"]["execution_type"] == "reduce"
    assert env.portfolio.get_position("EURUSD").lots == pytest.approx(1.0)


def test_reverse_has_two_fills_semantics():
    env = make_env()
    env.reset()
    env.step(PortfolioAction((PositionIntent("EURUSD", 1, 1.0),)))
    result = env.step(PortfolioAction((PositionIntent("EURUSD", -1, 1.0),)))
    info = result.info["execution"]["EURUSD"]
    assert info["execution_type"] == "reverse"
    assert info["commission"] == pytest.approx(14.0)
    assert env.portfolio.get_position("EURUSD").side == -1


def test_dataframe_observation_clock_must_match_quote_clock():
    ds = make_dataset()
    bad_index = ds.get("M1").index + pd.Timedelta(seconds=1)
    observations = pd.DataFrame(np.zeros((5, 2), dtype=np.float32), index=bad_index)
    with pytest.raises(ValueError, match="No exact closed-candle quote"):
        TradingEnvironment.from_mtf_datasets(
            observations={"EURUSD": observations},
            datasets={"EURUSD": ds},
            instruments={"EURUSD": instrument_spec()},
            config=EnvironmentConfig(window_size=1),
        )


def test_entry_commission_reaches_portfolio_accounting() -> None:
    env = make_env()
    env.reset()

    result = env.step(
        PortfolioAction(
            (PositionIntent("EURUSD", 1, 1.0),)
        )
    )

    execution_info = result.info["execution"]["EURUSD"]

    assert execution_info["commission"] == pytest.approx(7.0)
    assert execution_info["realized_pnl"] == pytest.approx(0.0)
    assert execution_info["accounting_realized_delta"] == pytest.approx(-7.0)

    assert env.portfolio.balance == pytest.approx(9993.0)
    assert env.portfolio.realized_pnl == pytest.approx(-7.0)


def test_disabled_entry_charge_does_not_disable_exit_policy() -> None:
    execution = ExecutionSimulator(
        charge_on_entry=False,
        charge_on_exit=True,
    )

    env = make_env(execution=execution)
    env.reset()

    result = env.step(
        PortfolioAction(
            (PositionIntent("EURUSD", 1, 1.0),)
        )
    )

    execution_info = result.info["execution"]["EURUSD"]

    assert execution_info["commission"] == pytest.approx(0.0)
    assert execution_info["accounting_realized_delta"] == pytest.approx(0.0)
    assert env.portfolio.balance == pytest.approx(10000.0)


