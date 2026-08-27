"""
Live candle feed.

Binance  = signal source  (crossover, H1 trend filter, H1 EMA100 exit)
Hyperliquid = SL price adaptation source (and the execution venue)

Both fetch the most recent CLOSED bars. The still-forming bar is dropped so the
bot only ever acts on finalized candles (no repaint).
"""
from typing import Optional

import pandas as pd
import requests

from .nethttp import retry, TRANSIENT_READ

# Binance public market-data mirror — NOT geo-blocked (api.binance.com returns
# HTTP 451 from US IPs like Render's). Same klines payload, no API key.
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

INTERVAL_MS = {"30m": 1_800_000, "1h": 3_600_000}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _now_ms() -> int:
    # time.time() avoided in workflow scripts; fine in normal runtime
    import time
    return int(time.time() * 1000)


def _drop_forming(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Drop the last bar if it has not closed yet (open_time + interval > now)."""
    if df.empty:
        return df
    step = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None \
        else pd.Timestamp.utcnow()
    last_open = df.index[-1]
    if last_open + step > now:
        return df.iloc[:-1]
    return df


def fetch_binance_recent(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    """
    Most recent `lookback` CLOSED Binance klines.
    Index: open_time (UTC). Columns: open, high, low, close, volume (float).
    """
    def _get():
        r = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": lookback + 1},
            timeout=30,
        )
        r.raise_for_status()
        return r
    resp = retry(_get, TRANSIENT_READ)
    bars = resp.json()
    df = pd.DataFrame(bars, columns=KLINE_COLUMNS)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time").sort_index()
    return _drop_forming(df, interval)


def fetch_hl_candles(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    api_url: str,
) -> pd.DataFrame:
    """
    Hyperliquid candles for [start_ms, end_ms].
    Index: open_time (UTC). Columns: open, high, low, close, volume (float).

    HL raw fields: t(open ms), T(close ms), o,c,h,l, v, n.
    """
    body = {"type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval,
                    "startTime": start_ms, "endTime": end_ms}}
    def _post():
        r = requests.post(f"{api_url}/info", json=body, timeout=30)
        r.raise_for_status()
        return r
    resp = retry(_post, TRANSIENT_READ)
    data = resp.json()
    if not data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time").sort_index()
    return _drop_forming(df, interval)


def fetch_hl_recent(coin: str, interval: str, lookback: int, api_url: str) -> pd.DataFrame:
    """Most recent `lookback` CLOSED Hyperliquid candles."""
    end = _now_ms()
    start = end - (lookback + 2) * INTERVAL_MS[interval]
    return fetch_hl_candles(coin, interval, start, end, api_url)
