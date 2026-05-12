from typing import Optional
import numpy as np


def find_sl(
    opens:         np.ndarray,
    closes:        np.ndarray,
    highs:         np.ndarray,
    lows:          np.ndarray,
    crossover_idx: int,
    lookback:      int = 50,
) -> Optional[float]:
    """
    Port of Pine Script findSL() — called at crossover_idx (the crossover bar).

    Algorithm (mirrors Pine bar-offset logic):
      For lb = 1 .. lookback:
        b_idx = crossover_idx - lb              (bearish candidate)
        if b_idx < 0: stop
        if close[b_idx] < open[b_idx]:          (bearish candle)
          check indices b_idx+1 .. crossover_idx (subsequent candles)
          valid if ANY has: high > bearish.high AND close > bearish.open
          if valid: SL = min(lows[b_idx : crossover_idx+1])
                    return SL immediately (most recent valid bearish wins)

    Returns None if no valid bearish candle found within lookback.
    """
    for lb in range(1, lookback + 1):
        b_idx = crossover_idx - lb
        if b_idx < 0:
            break

        b_open  = opens[b_idx]
        b_close = closes[b_idx]
        b_high  = highs[b_idx]
        b_low   = lows[b_idx]

        if b_close >= b_open:
            continue  # not bearish

        # Check every candle from b_idx+1 to crossover_idx inclusive
        valid = False
        for check_idx in range(b_idx + 1, crossover_idx + 1):
            if highs[check_idx] > b_high and closes[check_idx] > b_open:
                valid = True
                break

        if valid:
            sl = float(lows[b_idx : crossover_idx + 1].min())
            return sl

    return None
