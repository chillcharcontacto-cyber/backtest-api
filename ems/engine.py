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
    exit_reason: str    # "SL" | "H1_EMA100"
    r_multiple:  float  # (exit_price - entry_price) / (entry_price - sl_price)


def simulate(
    m30: pd.DataFrame,
    h1:  pd.DataFrame,
    cfg: Config,
) -> List[Trade]:
    """
    Bar-by-bar simulation of the EMS System on M30 data.

    H1 alignment (mirrors Pine's request.security with close[1] + lookahead_on):
      For each M30 bar at time t:
        last_closed_h1_open = floor(t, 1h) - 1h
      This gives the open time of the H1 bar that just completed, regardless of
      whether t is on an hourly boundary or 30 min into the hour.

      h1_new_bar = True when t.minute == 0
        (first M30 bar of each new H1 period = when Pine's h1_t changes)

    Exit priority (mirrors Pine strategy.exit vs strategy.close):
      1. SL stop (intrabar): if low[i] <= trade_sl -> exit at trade_sl
         Gap-down handling: if open[i] < trade_sl -> exit at open[i]
      2. H1 EMA100 cross (bar close): if h1_new_bar and h1_c < h1_ema_exit
         -> exit at close[i]
    """
    # --- numpy arrays for speed ---
    m30_opens   = m30["open"].to_numpy(dtype=float)
    m30_highs   = m30["high"].to_numpy(dtype=float)
    m30_lows    = m30["low"].to_numpy(dtype=float)
    m30_closes  = m30["close"].to_numpy(dtype=float)
    m30_cross   = m30["cross_up"].to_numpy(dtype=bool)
    m30_times   = m30.index

    h1_closes    = h1["close"].to_numpy(dtype=float)
    h1_ema_trend = h1["h1_ema_trend"].to_numpy(dtype=float)
    h1_ema_exit  = h1["h1_ema_exit"].to_numpy(dtype=float)

    # O(1) H1 lookup by open_time
    h1_time_to_idx = {t: i for i, t in enumerate(h1.index)}

    # --- simulation state ---
    trades:       List[Trade]     = []
    in_position:  bool            = False
    trade_sl:     float           = 0.0
    entry_price:  float           = 0.0
    entry_time:   Optional[pd.Timestamp] = None

    min_risk_factor = cfg.min_risk_pct / 100.0
    one_hour = pd.Timedelta(hours=1)

    for i in range(1, len(m30)):
        t = m30_times[i]

        # --- resolve last-closed H1 bar ---
        last_h1_open = t.floor("h") - one_hour
        h1_idx = h1_time_to_idx.get(last_h1_open)
        if h1_idx is None:
            continue  # no H1 bar available yet (warmup period)

        h1_c          = h1_closes[h1_idx]
        h1_trend_val  = h1_ema_trend[h1_idx]
        h1_exit_val   = h1_ema_exit[h1_idx]
        h1_new_bar    = (t.minute == 0)

        h1_bull  = h1_c > h1_trend_val
        h1_exit  = h1_new_bar and (h1_c < h1_exit_val)

        bar_open  = m30_opens[i]
        bar_high  = m30_highs[i]
        bar_low   = m30_lows[i]
        bar_close = m30_closes[i]

        # ------------------------------------------------------------------ #
        #  IN POSITION                                                         #
        # ------------------------------------------------------------------ #
        if in_position:
            # 1. SL hit (intrabar stop) -- checked before H1 exit
            if bar_low <= trade_sl:
                # Gap-down: if we opened below stop, fill at open
                exit_price = trade_sl if bar_open >= trade_sl else bar_open
                r = (exit_price - entry_price) / (entry_price - trade_sl)
                trades.append(Trade(
                    entry_time  = entry_time,
                    exit_time   = t,
                    entry_price = entry_price,
                    sl_price    = trade_sl,
                    exit_price  = round(exit_price, 8),
                    exit_reason = "SL",
                    r_multiple  = round(r, 4),
                ))
                in_position = False
                continue

            # 2. H1 EMA100 exit (at bar close)
            if h1_exit:
                exit_price = bar_close
                r = (exit_price - entry_price) / (entry_price - trade_sl)
                trades.append(Trade(
                    entry_time  = entry_time,
                    exit_time   = t,
                    entry_price = entry_price,
                    sl_price    = trade_sl,
                    exit_price  = round(exit_price, 8),
                    exit_reason = "H1_EMA100",
                    r_multiple  = round(r, 4),
                ))
                in_position = False
                continue

        # ------------------------------------------------------------------ #
        #  FLAT -- check entry                                                 #
        # ------------------------------------------------------------------ #
        else:
            # crossover on PREVIOUS bar + H1 trend up + not in position
            if m30_cross[i - 1] and h1_bull:
                sl = find_sl(
                    opens         = m30_opens,
                    closes        = m30_closes,
                    highs         = m30_highs,
                    lows          = m30_lows,
                    crossover_idx = i - 1,
                    lookback      = cfg.sl_lookback,
                )
                if sl is not None:
                    ep   = bar_open
                    risk = ep - sl
                    if risk >= ep * min_risk_factor:
                        in_position = True
                        trade_sl    = sl
                        entry_price = ep
                        entry_time  = t

    return trades
