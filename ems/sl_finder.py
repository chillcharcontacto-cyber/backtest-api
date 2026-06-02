from typing import Optional, Tuple
import numpy as np


def find_sl_with_anchor(
    opens:         np.ndarray,
    closes:        np.ndarray,
    highs:         np.ndarray,
    lows:          np.ndarray,
    crossover_idx: int,
    lookback:      Optional[int] = None,
) -> Optional[Tuple[float, int]]:
    """
    Find structural stop loss for a long entry, returning BOTH the SL price and
    the anchor bar index.

    Called at crossover_idx (the crossover bar, i.e. i-1 from entry bar).

    Algorithm:
      Walk back from j = crossover_idx - 1 toward max(0, crossover_idx - max_lb).
      For each BEARISH candidate (close[j] < open[j]):
        Require at least one k in (j, crossover_idx] satisfying BOTH on the SAME k:
          highs[k] > highs[j]   (higher high in wick)
          closes[k] > opens[j]  (higher high in body / close above anchor open)
        CRITICAL: both conditions must be met by the SAME k (not different k's).
      First qualifying j is the anchor.
      SL = min(lows[j : crossover_idx + 1])

    Parameters:
      lookback: max bars to walk back. None = unlimited (walk to bar 0).

    Returns (sl_price, anchor_idx) or None if no qualifying anchor found.

    The anchor_idx is needed by the live bot to map the SL timestamp range
    (anchor_idx .. crossover_idx) onto Hyperliquid candles for venue-correct
    stop placement.
    """
    max_lb = crossover_idx if lookback is None else min(lookback, crossover_idx)

    for lb in range(1, max_lb + 1):
        b_idx = crossover_idx - lb
        if b_idx < 0:
            break

        b_open  = opens[b_idx]
        b_close = closes[b_idx]
        b_high  = highs[b_idx]

        if b_close >= b_open:
            continue  # not bearish (strict)

        # Both conditions must be met by the SAME k
        valid = False
        for k in range(b_idx + 1, crossover_idx + 1):
            if highs[k] > b_high and closes[k] > b_open:
                valid = True
                break

        if valid:
            sl = float(lows[b_idx : crossover_idx + 1].min())
            return sl, b_idx

    return None


def find_sl(
    opens:         np.ndarray,
    closes:        np.ndarray,
    highs:         np.ndarray,
    lows:          np.ndarray,
    crossover_idx: int,
    lookback:      Optional[int] = None,
) -> Optional[float]:
    """
    Structural stop loss price for a long entry. Thin wrapper over
    find_sl_with_anchor() that returns only the price (backward-compatible).
    """
    result = find_sl_with_anchor(
        opens=opens, closes=closes, highs=highs, lows=lows,
        crossover_idx=crossover_idx, lookback=lookback,
    )
    return None if result is None else result[0]
