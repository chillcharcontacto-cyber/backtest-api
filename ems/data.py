import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_ohlcv(
    symbol:   str,
    interval: str,        # "30m" | "1h"
    start:    str,        # "2017-08-17"
    end:      str,        # "2026-05-12"
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Fetch Binance public klines with pagination (1000 bars / request).
    Caches result as parquet at data_dir/{symbol}_{interval}.parquet.
    Returns from cache if file already exists.

    DataFrame index: open_time (UTC-aware DatetimeIndex)
    Columns: open, high, low, close, volume (all float64)
    """
    cache_path = Path(data_dir) / f"{symbol}_{interval}.parquet"

    if cache_path.exists():
        print(f"  Cache hit: {cache_path}")
        return pd.read_parquet(cache_path)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Fetching {symbol} {interval} from Binance ({start} -> {end}) ...")

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)

    all_bars = []
    current_start = start_ms
    request_count = 0

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     1000,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        bars = resp.json()

        if not bars:
            break

        all_bars.extend(bars)
        request_count += 1

        last_open_ms = bars[-1][0]
        if last_open_ms <= current_start:
            break  # no progress, shouldn't happen

        current_start = last_open_ms + 1

        if request_count % 50 == 0:
            print(f"    ... {len(all_bars):,} bars fetched ({request_count} requests)")

        time.sleep(0.05)  # stay well within Binance rate limits (1200 req/min)

    print(f"  Total: {len(all_bars):,} bars in {request_count} requests")

    df = pd.DataFrame(all_bars, columns=KLINE_COLUMNS)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.set_index("open_time")
    df = df.sort_index()

    # Exclude the end date itself (match Pine's exclusive end)
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[df.index < end_ts]

    df.to_parquet(cache_path)
    print(f"  Cached -> {cache_path}")
    return df
