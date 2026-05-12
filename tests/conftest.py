import numpy as np
import pandas as pd


def make_ohlcv(
    n: int = 100,
    base: float = 30000.0,
    drift: float = 50.0,
    freq: str = "30min",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(drift + rng.normal(0, 30, n))
    opens  = closes + rng.normal(0, 15, n)
    highs  = np.maximum(opens, closes) + rng.uniform(5, 50, n)
    lows   = np.minimum(opens, closes) - rng.uniform(5, 50, n)
    vols   = rng.uniform(1, 100, n)
    idx    = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )
