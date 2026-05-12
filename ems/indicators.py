import pandas as pd


def add_emas(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """
    Add ema_fast and ema_slow columns.
    Uses ewm(span=n, adjust=False) which matches Pine Script's ta.ema() exactly:
      alpha = 2 / (n + 1),  EMA[i] = alpha * close[i] + (1 - alpha) * EMA[i-1]
    """
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def mark_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross_up (bool) column.
    True on the bar where ema_fast crosses above ema_slow:
      previous bar: fast <= slow
      current bar:  fast >  slow
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
