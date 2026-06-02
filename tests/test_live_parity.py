"""
Parity guarantee: ems_live.decider.replay() must reproduce
ems.engine.simulate() trade-for-trade on identical data.

If this test ever fails, the live bot has diverged from the validated backtest.

Two layers:
  - synthetic seeds -> logic equality (passes even when 0 trades)
  - real cached Binance slice -> non-vacuous equality on 100+ real trades
    (skipped automatically if the parquet cache is not present)
"""
import os

import numpy as np
import pandas as pd
import pytest

from ems.config import Config
from ems.engine import simulate
from ems.indicators import add_emas, add_h1_emas, mark_crossovers
from ems_live.decider import replay

DATA_M30 = "data/BTCUSDT_30m.parquet"
DATA_H1  = "data/BTCUSDT_1h.parquet"


def make_m30(n=300, base=30000.0, drift=80.0, seed=0):
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(drift + rng.normal(0, 40, n))
    opens = closes + rng.normal(0, 15, n)
    highs = np.maximum(opens, closes) + rng.uniform(10, 80, n)
    lows = np.minimum(opens, closes) - rng.uniform(10, 80, n)
    times = pd.date_range("2022-01-01", periods=n, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(n)},
        index=times,
    )
    df = add_emas(df, fast=20, slow=50)
    df = mark_crossovers(df)
    return df


def make_h1(n=300, base=30000.0, drift=200.0, seed=1):
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(drift + rng.normal(0, 60, n))
    opens = closes + rng.normal(0, 30, n)
    highs = np.maximum(opens, closes) + rng.uniform(20, 120, n)
    lows = np.minimum(opens, closes) - rng.uniform(20, 120, n)
    times = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(n)},
        index=times,
    )
    df = add_h1_emas(df, trend=50, exit_=100)
    return df


def cfg():
    return Config(min_risk_pct=0.1, warmup_bars=0, strategy_name="EMA-Cross")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 13, 42, 99])
def test_replay_matches_simulate(seed):
    # h1 fixed seed=1 (uptrend) pairs with m30 to actually produce entries,
    # mirroring the known-trading combo in test_engine.py
    m30 = make_m30(seed=seed)
    h1 = make_h1(seed=1)
    expected = simulate(m30, h1, cfg())
    actual = replay(m30, h1, cfg())

    assert len(actual) == len(expected), (
        f"trade count mismatch: replay={len(actual)} simulate={len(expected)}"
    )
    for a, e in zip(actual, expected):
        assert a == e, f"trade mismatch:\n  replay   = {a}\n  simulate = {e}"


@pytest.mark.skipif(
    not (os.path.exists(DATA_M30) and os.path.exists(DATA_H1)),
    reason="Binance parquet cache not present",
)
def test_replay_matches_simulate_real_data():
    """Non-vacuous parity: 100+ real trades on a cached Binance slice."""
    m30 = pd.read_parquet(DATA_M30).iloc[:30000]
    h1 = pd.read_parquet(DATA_H1).iloc[:15000]
    m30 = mark_crossovers(add_emas(m30, 20, 50))
    h1 = add_h1_emas(h1, 50, 100)
    c = Config(min_risk_pct=0.1, warmup_bars=500, strategy_name="EMA-Cross")

    expected = simulate(m30, h1, c)
    actual = replay(m30, h1, c)

    assert len(expected) > 100, "slice should produce 100+ trades"
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == e
