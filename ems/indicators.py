import pandas as pd


def add_emas(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """
    Add ema_fast and ema_slow columns.
    Uses ewm(span=n, adjust=False) — matches Pine Script ta.ema() exactly.
    alpha = 2 / (n + 1),  EMA[i] = alpha * close[i] + (1 - alpha) * EMA[i-1]
    """
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def mark_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross_up (bool) column.
    True on bar where ema_fast crosses strictly above ema_slow:
      prev: fast <= slow,  current: fast > slow
    NaN on prev bars propagates as False (no false crossover on first bar).
    """
    df = df.copy()
    prev_fast = df["ema_fast"].shift(1)
    prev_slow = df["ema_slow"].shift(1)
    df["cross_up"] = (prev_fast <= prev_slow) & (df["ema_fast"] > df["ema_slow"])
    return df


def add_h1_emas(df: pd.DataFrame, trend: int, exit_: int) -> pd.DataFrame:
    """Add h1_ema_trend and h1_ema_exit columns to an H1 DataFrame."""
    df = df.copy()
    df["h1_ema_trend"] = df["close"].ewm(span=trend, adjust=False).mean()
    df["h1_ema_exit"]  = df["close"].ewm(span=exit_, adjust=False).mean()
    return df


def build_h4(h1: pd.DataFrame) -> pd.DataFrame:
    """
    Resample H1 OHLCV to H4 by taking the last H1 close in each 4-hour window.
    H4 bars align to UTC boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00.
    Matches Samuel's reference: h4_close = h1['close'].resample('4h').last()
    """
    h4 = pd.DataFrame({
        "open":   h1["open"].resample("4h", closed="left", label="left").first(),
        "high":   h1["high"].resample("4h", closed="left", label="left").max(),
        "low":    h1["low"].resample("4h", closed="left", label="left").min(),
        "close":  h1["close"].resample("4h", closed="left", label="left").last(),
        "volume": h1["volume"].resample("4h", closed="left", label="left").sum(),
    })
    return h4.dropna(subset=["close"])


def add_h4_emas(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Add h4_ema_fast and h4_ema_slow columns to an H4 DataFrame."""
    df = df.copy()
    df["h4_ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["h4_ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df
