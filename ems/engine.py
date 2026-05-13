from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .sl_finder import find_sl


@dataclass(frozen=True)
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    entry_price: float
    sl_price:    float
    exit_price:  float
    exit_reason: str    # "STRUCTURAL_SL" | "H1_EMA100"
    r_multiple:  float  # -1.00 for STRUCTURAL_SL; signed R for H1_EMA100 exits
    strategy:    str    # "EMA-Cross" | "EMA-Cross-H4F"


def simulate(
    m30: pd.DataFrame,
    h1:  pd.DataFrame,
    cfg: Config,
    h4:  Optional[pd.DataFrame] = None,
) -> List[Trade]:
    """
    Bar-by-bar EMS simulation aligned to Samuel's canonical rules.

    V2 changes vs V1:
      - Warmup: first cfg.warmup_bars M30 bars produce no entries (EMA-seed protection)
      - SL lookback: unlimited (find_sl called with lookback=None)
      - H1 exit timing: fires at :30 M30 bars (second M30 of H1 window)
        exit_price = H1 close of that bar; exit_time = h1_open + 1h
      - Gap-down SL: always fill at sl_price, r_multiple forced to -1.00
      - exit_reason labels: "STRUCTURAL_SL" | "H1_EMA100"
      - NaN guards on every EMA comparison

    V3 addition (h4 != None and cfg.h4_filter):
      - H4 EMA20 must be strictly above H4 EMA50 at most recent closed H4 bar
      - h4_lookup_ts = (entry_ts - 4h).floor('4h')
    """
    # --- numpy arrays ---
    m30_opens  = m30["open"].to_numpy(dtype=float)
    m30_highs  = m30["high"].to_numpy(dtype=float)
    m30_lows   = m30["low"].to_numpy(dtype=float)
    m30_closes = m30["close"].to_numpy(dtype=float)
    m30_cross  = m30["cross_up"].to_numpy(dtype=bool)
    m30_times  = m30.index

    h1_closes    = h1["close"].to_numpy(dtype=float)
    h1_ema_trend = h1["h1_ema_trend"].to_numpy(dtype=float)
    h1_ema_exit  = h1["h1_ema_exit"].to_numpy(dtype=float)
    h1_time_idx  = {t: i for i, t in enumerate(h1.index)}

    # H4 filter arrays (V3 only)
    use_h4 = (h4 is not None) and cfg.h4_filter
    h4_ema_fast_arr = None
    h4_ema_slow_arr = None
    h4_time_idx     = {}
    if use_h4:
        h4_ema_fast_arr = h4["h4_ema_fast"].to_numpy(dtype=float)
        h4_ema_slow_arr = h4["h4_ema_slow"].to_numpy(dtype=float)
        h4_time_idx     = {t: i for i, t in enumerate(h4.index)}

    # --- state ---
    trades:      List[Trade]           = []
    in_position: bool                  = False
    trade_sl:    float                 = 0.0
    entry_price: float                 = 0.0
    entry_time:  Optional[pd.Timestamp] = None

    min_risk_factor = cfg.min_risk_pct / 100.0
    one_hour    = pd.Timedelta(hours=1)
    four_hours  = pd.Timedelta(hours=4)
    thirty_min  = pd.Timedelta(minutes=30)

    for i in range(1, len(m30)):
        t        = m30_times[i]
        bar_open  = m30_opens[i]
        bar_low   = m30_lows[i]
        bar_close = m30_closes[i]

        # ------------------------------------------------------------------ #
        #  IN POSITION                                                         #
        # ------------------------------------------------------------------ #
        if in_position:

            # Rule 7: SL hit — intrabar, always first
            if bar_low <= trade_sl:
                # Always fill at sl_price (optimistic; gap-downs capped at -1R)
                trades.append(Trade(
                    entry_time  = entry_time,
                    exit_time   = t,
                    entry_price = entry_price,
                    sl_price    = trade_sl,
                    exit_price  = trade_sl,
                    exit_reason = "STRUCTURAL_SL",
                    r_multiple  = -1.00,
                    strategy    = cfg.strategy_name,
                ))
                in_position = False
                continue

            # Rule 8: H1 EMA100 exit — fires at :30 M30 bars only
            if t.minute == 30:
                h1_exit_open = t - thirty_min          # = t.floor('h')
                h1_exit_idx  = h1_time_idx.get(h1_exit_open)
                if h1_exit_idx is not None:
                    h1_c_exit = h1_closes[h1_exit_idx]
                    h1_e100   = h1_ema_exit[h1_exit_idx]
                    if (not np.isnan(h1_c_exit) and not np.isnan(h1_e100)
                            and h1_c_exit < h1_e100):
                        exit_price = h1_c_exit
                        exit_time  = h1_exit_open + one_hour  # H1 close timestamp
                        r = (exit_price - entry_price) / (entry_price - trade_sl)
                        trades.append(Trade(
                            entry_time  = entry_time,
                            exit_time   = exit_time,
                            entry_price = entry_price,
                            sl_price    = trade_sl,
                            exit_price  = round(exit_price, 8),
                            exit_reason = "H1_EMA100",
                            r_multiple  = round(r, 4),
                            strategy    = cfg.strategy_name,
                        ))
                        in_position = False
                        continue

        # ------------------------------------------------------------------ #
        #  FLAT — check entry                                                  #
        # ------------------------------------------------------------------ #
        else:
            # Warmup guard (Rule 1 / Section 4): no entries before bar 500
            # i is the entry bar; crossover must be at i-1 >= warmup_bars
            if i <= cfg.warmup_bars:
                continue

            if not m30_cross[i - 1]:
                continue

            # Rule 2: H1 trend filter — last CLOSED H1 bar before entry
            h1_lookup = t.floor("h") - one_hour
            h1_idx    = h1_time_idx.get(h1_lookup)
            if h1_idx is None:
                continue
            h1_c_entry = h1_closes[h1_idx]
            h1_trend   = h1_ema_trend[h1_idx]
            if np.isnan(h1_c_entry) or np.isnan(h1_trend):
                continue
            if h1_c_entry <= h1_trend:      # must be strictly above for longs
                continue

            # Rule H4: H4 confluence filter (V3 only)
            if use_h4:
                h4_lookup = (t - four_hours).floor("4h")
                h4_idx    = h4_time_idx.get(h4_lookup)
                if h4_idx is None:
                    continue
                h4_e20 = h4_ema_fast_arr[h4_idx]
                h4_e50 = h4_ema_slow_arr[h4_idx]
                if np.isnan(h4_e20) or np.isnan(h4_e50):
                    continue
                if h4_e20 <= h4_e50:        # H4 EMA20 must be strictly above H4 EMA50
                    continue

            # Rule 5 + 6: SL anchor — unlimited lookback
            sl = find_sl(
                opens         = m30_opens,
                closes        = m30_closes,
                highs         = m30_highs,
                lows          = m30_lows,
                crossover_idx = i - 1,
                lookback      = None,       # unlimited
            )
            if sl is None:
                continue

            ep = bar_open
            if ep <= sl:                    # defensive: degenerate SL
                continue

            # Rule 3: minimum risk filter
            if (ep - sl) / ep < min_risk_factor:
                continue

            in_position = True
            trade_sl    = sl
            entry_price = ep
            entry_time  = t

    return trades
