"""
Stop-loss adaptation: Binance signal -> Hyperliquid stop price.

The EMS structural SL is identified on Binance candles (sl_finder picks an anchor
bar; SL = min low over anchor..crossover). But the bot trades on Hyperliquid,
whose candles differ from Binance by a small basis (observed ~15-45 USD on BTC).

Placing the stop at the Binance low would be the wrong price on Hyperliquid. So:
  1. Binance signal gives the anchor->crossover TIMESTAMP range.
  2. We pull Hyperliquid candles over that same range.
  3. HL stop price = min(low) of those Hyperliquid candles.

This yields a venue-correct structural stop at the same swing the signal found.
"""
import pandas as pd

from .feed import fetch_hl_candles, INTERVAL_MS


def adapt_sl_to_hl(
    anchor_time:    pd.Timestamp,
    crossover_time: pd.Timestamp,
    coin:           str,
    api_url:        str,
    interval:       str = "30m",
) -> float:
    """
    Hyperliquid structural stop price = lowest HL low across the M30 bars in
    [anchor_time, crossover_time] inclusive.

    Raises RuntimeError if Hyperliquid returns no candles for the range (caller
    should treat as "skip trade" — never place a stop at an unknown price).
    """
    start_ms = int(anchor_time.timestamp() * 1000)
    # include the crossover bar fully: pad end by one interval
    end_ms = int(crossover_time.timestamp() * 1000) + INTERVAL_MS[interval]

    hl = fetch_hl_candles(coin, interval, start_ms, end_ms, api_url)

    # keep only bars within the inclusive anchor..crossover window
    hl = hl[(hl.index >= anchor_time) & (hl.index <= crossover_time)]
    if hl.empty:
        raise RuntimeError(
            f"Hyperliquid returned no candles for SL range "
            f"{anchor_time} .. {crossover_time}"
        )
    return float(hl["low"].min())
