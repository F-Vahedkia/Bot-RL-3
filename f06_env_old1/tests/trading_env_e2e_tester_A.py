# Run:
# pytest -v -s f06_env/trading_env_e2e_tester_A.py

# ---------------------------------------------------------
import numpy as np
import pytest
from f06_env.contracts import (
    EnvironmentConfig,
    PortfolioAction,
    PositionIntent,
)
from f06_env.execution_simulator import ExecutionSimulator
from f06_env.trading_env import TradingEnvironment

# ---------------------------------------------------------
def make_e2e_env():
    observations = {
        "XAUUSD": np.zeros((6, 2), dtype=np.float32),
        "EURUSD": np.zeros((6, 2), dtype=np.float32),
    }
    prices = {
        "XAUUSD": np.array(
            [100, 102, 104, 103, 101, 100],
            dtype=np.float64,
        ),
        "EURUSD": np.array(
            [10, 11, 12, 11, 10, 9],
            dtype=np.float64,
        ),
    }
    cfg = EnvironmentConfig(
        base_tf="M1",
        window_size=1,
        initial_balance=10000.0,
        leverage=100.0,
        max_episode_steps=20,
        max_drawdown_pct=0.50,
    )
    return TradingEnvironment(
        observations=observations,
        prices=prices,
        config=cfg,
        execution=ExecutionSimulator(
            contract_size=1.0,
            point_value=1.0,
        ),
    )


# ---------------------------------------------------------
def test_end_to_end_multi_symbol_flow():
    env = make_e2e_env()
    obs, info = env.reset(seed=123)
    assert obs.shape == (4,)
    assert info["symbols"] == (
        "XAUUSD",
        "EURUSD",
    )
    result = env.step(
        PortfolioAction(
            (
                PositionIntent("XAUUSD", 1, 2.0),
                PositionIntent("EURUSD", -1, 5.0),
            )
        )
    )

    assert result.observation.shape == (4,)

    assert env.portfolio.get_position("XAUUSD").side == 1
    assert env.portfolio.get_position("EURUSD").side == -1

    assert result.info["used_margin"] > 0

    assert result.info["equity"] == pytest.approx(
        env.portfolio.equity
    )

    assert result.info["margin_level"] is not None


# ---------------------------------------------------------
def test_hold_action_preserves_positions():
    env = make_e2e_env()
    env.reset()
    env.step(
        PortfolioAction(
            (
                PositionIntent("XAUUSD", 1, 1.0),
            )
        )
    )

    result = env.step(PortfolioAction())

    pos = env.portfolio.get_position("XAUUSD")

    assert pos.side == 1
    assert pos.lots == 1.0

    assert result.terminated is False


# ---------------------------------------------------------
def test_close_all_positions():
    env = make_e2e_env()
    env.reset()
    env.step(
        PortfolioAction(
            (
                PositionIntent("XAUUSD", 1, 1.0),
            )
        )
    )

    env.step(
        PortfolioAction(
            (
                PositionIntent("XAUUSD", 0, 0.0),
            )
        )
    )

    pos = env.portfolio.get_position("XAUUSD")

    assert pos.side == 0
    assert pos.lots == 0.0
    assert pos.entry_price is None


# ---------------------------------------------------------
def test_deterministic_episode():

    def run_episode():
        env = make_e2e_env()
        env.reset(seed=999)
        actions = [
            PortfolioAction(
                (
                    PositionIntent("XAUUSD", 1, 1.0),
                )
            ),
            PortfolioAction(),
            PortfolioAction(
                (
                    PositionIntent("XAUUSD", 0, 0.0),
                )
            ),
        ]
        output = []

        for action in actions:
            result = env.step(action)
            output.append(
                (
                    result.reward,
                    result.observation.copy(),
                    result.terminated,
                    result.truncated,
                    result.info["equity"],
                )
            )
        return output

    a = run_episode()
    b = run_episode()

    for x, y in zip(a, b):
        assert x[0] == pytest.approx(y[0])
        np.testing.assert_array_equal(
            x[1],
            y[1],
        )
        assert x[2:] == y[2:]

# --------------------------------------------------------- END
