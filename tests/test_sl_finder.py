import numpy as np
import pytest

from ems.sl_finder import find_sl


def arrays(bars):
    """bars = list of (open, high, low, close)"""
    a = np.array(bars, dtype=float)
    opens  = a[:, 0]
    highs  = a[:, 1]
    lows   = a[:, 2]
    closes = a[:, 3]
    return opens, closes, highs, lows


# ---------------------------------------------------------------------------
# lb = 1: bearish immediately before crossover
# ---------------------------------------------------------------------------

def test_lb1_crossover_is_confirmer():
    """Bearish at idx=0, crossover at idx=1. Crossover bar has HH in wick+body."""
    bars = [
        # (open, high, low, close)
        (110, 115, 95, 100),   # idx 0: bearish (o=110, c=100), high=115, low=95
        (102, 120, 98, 118),   # idx 1: crossover -- high(120)>115 AND close(118)>open(110)
    ]
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=1, lookback=10)
    # SL = min(low[0], low[1]) = min(95, 98) = 95
    assert sl == 95.0


def test_lb1_no_wick_hh():
    """Crossover bar high does NOT exceed bearish high -> invalid."""
    bars = [
        (110, 115, 95, 100),   # idx 0: bearish, high=115
        (102, 114, 98, 118),   # idx 1: high=114 < 115 -> wick HH fails
    ]
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=1, lookback=10)
    assert sl is None


def test_lb1_no_body_hh():
    """Crossover bar high passes but close does NOT exceed bearish open -> invalid."""
    bars = [
        (110, 115, 95, 100),   # idx 0: bearish, open=110
        (102, 120, 98, 109),   # idx 1: high(120)>115 OK, close(109)<open(110) FAIL
    ]
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=1, lookback=10)
    assert sl is None


# ---------------------------------------------------------------------------
# Most-recent-valid logic
# ---------------------------------------------------------------------------

def test_skips_invalid_bearish_finds_older_valid():
    """
    idx=2: bearish, no confirmer after it -> invalid
    idx=0: bearish, idx=1 confirms -> valid (found as lb=2 fallback after lb=1 invalid)
    Wait -- actually lb=1 = idx 3 (crossover-1).  Let me re-design:

    crossover = idx 3
    lb=1 -> idx 2 bearish, only confirmer is idx3 (crossover) which doesn't confirm
    lb=2 -> idx 1 bearish, idx2+idx3 are subsequent; idx3 confirms -> valid
    """
    bars = [
        (105, 130, 90, 115),   # idx 0: bullish
        (116, 125, 102, 108),  # idx 1: bearish (o=116, c=108), high=125, low=102
        (110, 122, 104, 107),  # idx 2: bearish (o=110, c=107), high=122, low=104
        (108, 140, 106, 135),  # idx 3: crossover -- high(140)>125 AND close(135)>116 -> confirms idx1
                               #                  -- high(140)>122 AND close(135)>110 -> also confirms idx2
    ]
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=3, lookback=10)
    # lb=1: idx=2 bearish, confirmer=idx3: high(140)>122 AND close(135)>110 -> VALID
    # Most recent valid bearish = idx2
    # SL = min(low[2], low[3]) = min(104, 106) = 104
    assert sl == 104.0


def test_returns_most_recent_not_oldest():
    """
    Two valid bearish candles. Should return SL from the more recent one.
    crossover = idx 4
    lb=1 -> idx 3 bearish, confirmed by idx4 -> valid (use this one, stop here)
    lb=2 -> idx 2 bearish (would also be valid, but we never reach it)
    """
    bars = [
        (100, 120, 88, 115),   # idx 0: bullish
        (116, 130, 99, 128),   # idx 1: bullish
        (129, 132, 110, 112),  # idx 2: bearish (o=129,c=112), high=132, low=110
        (113, 128, 108, 111),  # idx 3: bearish (o=113,c=111), high=128, low=108
        (112, 145, 110, 140),  # idx 4: crossover -- confirms both above
    ]
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=4, lookback=10)
    # lb=1: idx3 bearish, confirmed by idx4: high(145)>128 AND close(140)>113 -> VALID immediately
    # SL = min(low[3], low[4]) = min(108, 110) = 108
    assert sl == 108.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_bullish_returns_none():
    """No bearish candles at all -> None."""
    bars = [(90, 110, 88, 108)] * 10  # close > open everywhere
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=9, lookback=10)
    assert sl is None


def test_bearish_exists_but_no_confirmer():
    """Bearish candles exist but no subsequent bar provides HH in both wick+body."""
    bars = [
        (105, 200, 90, 100),   # idx 0: bearish, high=200
        (101, 150, 99, 110),   # idx 1: high=150 < 200 -> no wick HH
        (111, 200, 95, 112),   # idx 2: bearish, high=200
        (113, 150, 100, 120),  # idx 3: high=150 < 200 -> no wick HH
        (122, 190, 108, 118),  # idx 4: crossover, high=190 < 200 -> no wick HH for any bearish
    ]
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=4, lookback=10)
    assert sl is None


def test_lookback_limit_respected():
    """Valid bearish at lb=60, but lookback=50 -> None."""
    n = 70
    opens  = np.full(n, 100.0)
    closes = np.full(n, 105.0)   # all bullish
    highs  = np.full(n, 110.0)
    lows   = np.full(n, 95.0)

    # Plant a bearish candle at idx=0 (lb=69 from crossover at idx=69)
    opens[0]  = 110.0
    closes[0] = 100.0   # bearish
    # Plant a confirmer at idx=1
    highs[1]  = 115.0
    closes[1] = 111.0   # close > bearish open (110)

    o, c, h, l = opens, closes, highs, lows
    sl = find_sl(o, c, h, l, crossover_idx=69, lookback=50)
    assert sl is None   # valid bearish is at lb=69, beyond lookback=50


def test_confirmer_not_at_crossover():
    """Confirmer is a middle bar, not the crossover bar itself."""
    bars = [
        (120, 125, 100, 110),  # idx 0: bearish (o=120, c=110), high=125, low=100
        (111, 135, 109, 130),  # idx 1: confirmer -- high(135)>125 AND close(130)>120
        (131, 138, 128, 136),  # idx 2: bullish
        (137, 142, 132, 140),  # idx 3: crossover (bullish)
    ]
    o, c, h, l = arrays(bars)
    sl = find_sl(o, c, h, l, crossover_idx=3, lookback=10)
    # lb=3: idx=0 bearish, idx=1 confirms (middle bar, not crossover) -> valid
    # SL = min(lows[0..3]) = min(100, 109, 128, 132) = 100
    assert sl == 100.0
