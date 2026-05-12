import numpy as np
import pandas as pd
import pytest

from ems.indicators import add_emas, add_h1_emas, mark_crossovers


def make_df(closes, freq="30min"):
    n   = len(closes)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    c   = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": c - 5, "high": c + 10, "low": c - 10, "close": c, "volume": np.ones(n)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# EMA formula
# ---------------------------------------------------------------------------

def test_ema_matches_pine_recursive():
    """
    Verify add_emas uses alpha = 2/(n+1), adjust=False (Pine's ta.ema formula).
    Manually compute and compare.
    """
    closes = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0]
    df     = make_df(closes)
    result = add_emas(df, fast=3, slow=5)

    alpha = 2 / (3 + 1)
    ema   = closes[0]
    expected = [ema]
    for c in closes[1:]:
        ema = alpha * c + (1 - alpha) * ema
        expected.append(ema)

    np.testing.assert_allclose(result["ema_fast"].values, expected, rtol=1e-9)


def test_ema_columns_present():
    df = make_df([100.0] * 60)
    result = add_emas(df, fast=20, slow=50)
    assert "ema_fast" in result.columns
    assert "ema_slow" in result.columns


def test_ema_fast_above_slow_in_uptrend():
    """After warmup, fast EMA should be above slow EMA in a rising series."""
    closes = list(range(1, 101))   # strictly increasing
    df     = make_df(closes)
    result = add_emas(df, fast=20, slow=50)
    # After 50+ bars the fast EMA should track closer to recent (higher) prices
    assert result["ema_fast"].iloc[-1] > result["ema_slow"].iloc[-1]


def test_ema_does_not_mutate_input():
    df     = make_df([100.0] * 30)
    before = df.copy()
    add_emas(df, fast=5, slow=10)
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# Crossover detection
# ---------------------------------------------------------------------------

def test_crossover_fires_on_correct_bar():
    """
    Step function: flat at 100, then jump to 200.
    Fast EMA (3) responds earlier than slow (10).
    cross_up should fire exactly once.
    """
    closes = [100.0] * 30 + [200.0] * 30
    df     = make_df(closes)
    df     = add_emas(df, fast=3, slow=10)
    df     = mark_crossovers(df)
    assert df["cross_up"].sum() == 1


def test_no_crossover_in_flat_series():
    closes = [100.0] * 100
    df     = make_df(closes)
    df     = add_emas(df, fast=5, slow=20)
    df     = mark_crossovers(df)
    assert not df["cross_up"].any()


def test_crossover_not_on_first_bar():
    """First bar can never be a crossover (no previous bar to compare)."""
    closes = list(range(1, 101))
    df     = make_df(closes)
    df     = add_emas(df, fast=5, slow=20)
    df     = mark_crossovers(df)
    assert not df["cross_up"].iloc[0]


# ---------------------------------------------------------------------------
# H1 EMAs
# ---------------------------------------------------------------------------

def test_h1_ema_columns():
    df     = make_df([30000.0] * 150, freq="1h")
    result = add_h1_emas(df, trend=50, exit_=100)
    assert "h1_ema_trend" in result.columns
    assert "h1_ema_exit"  in result.columns
