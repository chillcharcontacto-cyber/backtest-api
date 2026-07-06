"""
Single-bar decision predicates shared by the live runner AND the backtest.

This module is the ONE source of truth for EMS V2/V3 entry/exit rules. The live
runner (ems_live.runner) and the offline driver replay() below both compose the
exact same predicates, so live behavior cannot drift from the backtested engine.

replay() reproduces ems.engine.simulate() trade-for-trade on identical data,
including the V3 H4 confluence filter when an h4 frame is supplied.
tests/test_live_parity.py asserts that equality.

Long-only. V3 adds the H4 confluence gate: enter only when H4 EMA(fast) > H4
EMA(slow) at the last closed H4 bar (mirrors ems.engine.simulate H4 branch).
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ems.engine import Trade
from ems.sl_finder import find_sl_with_anchor


# --------------------------------------------------------------------------- #
#  Context: precomputed arrays for fast bar-by-bar evaluation                  #
# --------------------------------------------------------------------------- #

@dataclass
class Ctx:
    """Precomputed numpy views over the M30 + H1 indicator frames."""
    m30_opens:  np.ndarray
    m30_highs:  np.ndarray
    m30_lows:   np.ndarray
    m30_closes: np.ndarray
    m30_cross:  np.ndarray
    m30_times:  pd.DatetimeIndex

    h1_closes:     np.ndarray
    h1_ema_trend:  np.ndarray
    h1_ema_exit:   np.ndarray
    h1_time_idx:   dict       # Timestamp -> row index

    one_hour:   pd.Timedelta
    thirty_min: pd.Timedelta
    four_hours: pd.Timedelta

    # H4 confluence (V3) — None when no H4 frame supplied
    h4_ema_fast:  Optional[np.ndarray] = None
    h4_ema_slow:  Optional[np.ndarray] = None
    h4_time_idx:  Optional[dict]       = None


def build_ctx(m30: pd.DataFrame, h1: pd.DataFrame,
              h4: Optional[pd.DataFrame] = None) -> Ctx:
    """
    Build a Ctx from indicator-ready frames.

    m30 must have: open, high, low, close, ema_fast, ema_slow, cross_up
    h1  must have: close, h1_ema_trend, h1_ema_exit
    h4  (optional, V3): h4_ema_fast, h4_ema_slow
    """
    h4_fast = h4_slow = h4_idx = None
    if h4 is not None:
        h4_fast = h4["h4_ema_fast"].to_numpy(dtype=float)
        h4_slow = h4["h4_ema_slow"].to_numpy(dtype=float)
        h4_idx  = {t: i for i, t in enumerate(h4.index)}

    return Ctx(
        m30_opens  = m30["open"].to_numpy(dtype=float),
        m30_highs  = m30["high"].to_numpy(dtype=float),
        m30_lows   = m30["low"].to_numpy(dtype=float),
        m30_closes = m30["close"].to_numpy(dtype=float),
        m30_cross  = m30["cross_up"].to_numpy(dtype=bool),
        m30_times  = m30.index,
        h1_closes    = h1["close"].to_numpy(dtype=float),
        h1_ema_trend = h1["h1_ema_trend"].to_numpy(dtype=float),
        h1_ema_exit  = h1["h1_ema_exit"].to_numpy(dtype=float),
        h1_time_idx  = {t: i for i, t in enumerate(h1.index)},
        one_hour   = pd.Timedelta(hours=1),
        thirty_min = pd.Timedelta(minutes=30),
        four_hours = pd.Timedelta(hours=4),
        h4_ema_fast = h4_fast,
        h4_ema_slow = h4_slow,
        h4_time_idx = h4_idx,
    )


# --------------------------------------------------------------------------- #
#  Decisions                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EntrySignal:
    entry_price:   float    # M30 bar open (Binance) — live entry is a market order here
    sl_price:      float    # structural SL price on Binance (adapt to HL before placing)
    anchor_idx:    int      # M30 index of the SL anchor bar
    crossover_idx: int      # M30 index of the crossover bar (entry_bar - 1)
    anchor_time:   pd.Timestamp
    crossover_time: pd.Timestamp


@dataclass(frozen=True)
class ExitSignal:
    exit_price: float
    exit_time:  pd.Timestamp
    reason:     str         # "STRUCTURAL_SL" | "H1_EMA100"
    r_multiple: float


# --------------------------------------------------------------------------- #
#  Predicates (single bar i = the bar currently being evaluated)               #
# --------------------------------------------------------------------------- #

def check_sl_hit(ctx: Ctx, i: int, trade_sl: float) -> bool:
    """Rule 7: intrabar stop hit when the bar low pierces the stop."""
    return ctx.m30_lows[i] <= trade_sl


def check_h1_exit(ctx: Ctx, i: int, entry_price: float, trade_sl: float
                  ) -> Optional[ExitSignal]:
    """
    Rule 8: H1 EMA100 exit. Fires only at :30 M30 bars (concurrent H1 just closed).
    exit_price = that H1 close; exit_time = h1_open + 1h.
    """
    t = ctx.m30_times[i]
    if t.minute != 30:
        return None
    h1_exit_open = t - ctx.thirty_min           # = t.floor('h')
    idx = ctx.h1_time_idx.get(h1_exit_open)
    if idx is None:
        return None
    h1_c = ctx.h1_closes[idx]
    h1_e = ctx.h1_ema_exit[idx]
    if np.isnan(h1_c) or np.isnan(h1_e) or h1_c >= h1_e:
        return None
    exit_price = h1_c
    exit_time  = h1_exit_open + ctx.one_hour
    r = (exit_price - entry_price) / (entry_price - trade_sl)
    return ExitSignal(round(exit_price, 8), exit_time, "H1_EMA100", round(r, 4))


def check_h4_confluence(ctx: Ctx, i: int) -> bool:
    """
    V3 H4 gate (mirrors ems.engine.simulate H4 branch exactly):
    at the last closed H4 bar, require H4 EMA(fast) strictly above H4 EMA(slow).
    Returns True if the gate PASSES (or no H4 frame -> no gate).
    """
    if ctx.h4_time_idx is None:
        return True
    t = ctx.m30_times[i]
    h4_lookup = (t - ctx.four_hours).floor("4h")
    h4_idx = ctx.h4_time_idx.get(h4_lookup)
    if h4_idx is None:
        return False
    h4_fast = ctx.h4_ema_fast[h4_idx]
    h4_slow = ctx.h4_ema_slow[h4_idx]
    if np.isnan(h4_fast) or np.isnan(h4_slow):
        return False
    return h4_fast > h4_slow


def check_entry(ctx: Ctx, i: int, cfg) -> Optional[EntrySignal]:
    """
    Entry pipeline for bar i (crossover at i-1, entry at open[i]):
      Rule 1 : warmup gate (i > warmup_bars)
      Rule 2 : crossover confirmed on bar i-1
      Rule 3 : H1 trend filter (last closed H1 close > H1 EMA50)
      Rule H4: V3 confluence — H4 EMA(fast) > H4 EMA(slow) [only if h4 supplied
               AND cfg.h4_filter]
      Rule 5/6: structural SL via unlimited lookback
      Rule 4 : min risk distance
    Returns EntrySignal or None.
    """
    warmup = getattr(cfg, "warmup_bars", 0)
    if i <= warmup:
        return None

    if not ctx.m30_cross[i - 1]:
        return None

    t = ctx.m30_times[i]

    # H1 trend filter — last CLOSED H1 bar before entry
    h1_lookup = t.floor("h") - ctx.one_hour
    h1_idx = ctx.h1_time_idx.get(h1_lookup)
    if h1_idx is None:
        return None
    h1_c = ctx.h1_closes[h1_idx]
    h1_trend = ctx.h1_ema_trend[h1_idx]
    if np.isnan(h1_c) or np.isnan(h1_trend) or h1_c <= h1_trend:
        return None

    # H4 confluence gate (V3) — engine order: after H1 trend, before SL
    if getattr(cfg, "h4_filter", False) and ctx.h4_time_idx is not None:
        if not check_h4_confluence(ctx, i):
            return None

    # Structural SL (unlimited lookback) + anchor index for HL adaptation
    res = find_sl_with_anchor(
        opens=ctx.m30_opens, closes=ctx.m30_closes,
        highs=ctx.m30_highs, lows=ctx.m30_lows,
        crossover_idx=i - 1, lookback=None,
    )
    if res is None:
        return None
    sl, anchor_idx = res

    ep = ctx.m30_opens[i]
    if ep <= sl:
        return None

    min_risk_factor = cfg.min_risk_pct / 100.0
    if (ep - sl) / ep < min_risk_factor:
        return None

    return EntrySignal(
        entry_price    = ep,
        sl_price       = sl,
        anchor_idx     = anchor_idx,
        crossover_idx  = i - 1,
        anchor_time    = ctx.m30_times[anchor_idx],
        crossover_time = ctx.m30_times[i - 1],
    )


def check_entry_live(ctx: Ctx, i: int, cfg, interval) -> Optional[EntrySignal]:
    """
    LIVE entry timing. The backtest enters at open[i] when the crossover is on bar
    i-1. Live acts on CLOSED bars, so at the tick right after the CROSSOVER bar
    closes (that bar = the latest closed bar, index i), the bot must enter at market
    immediately — which is ~open[i+1] ~ close[i], the SAME price/instant the backtest
    uses. This is one M30 bar EARLIER than checking cross[i-1] would fire.

    So the crossover is on bar i (the just-closed bar); the entry instant is
    t_i + interval (= next bar's open time); H1/H4 filters and the SL anchor are
    evaluated as of that instant, exactly matching check_entry(ctx, i+1). entry_price
    is an ESTIMATE (close[i]) — the runner sizes off the live mid/fill.
    """
    warmup = getattr(cfg, "warmup_bars", 0)
    if (i + 1) <= warmup or i < 1:
        return None
    if not ctx.m30_cross[i]:                 # crossover on the just-closed bar
        return None

    entry_time = ctx.m30_times[i] + interval  # = next bar's open = backtest entry time

    # H1 trend filter — last closed H1 as of the entry instant
    h1_lookup = entry_time.floor("h") - ctx.one_hour
    h1_idx = ctx.h1_time_idx.get(h1_lookup)
    if h1_idx is None:
        return None
    h1_c = ctx.h1_closes[h1_idx]
    h1_trend = ctx.h1_ema_trend[h1_idx]
    if np.isnan(h1_c) or np.isnan(h1_trend) or h1_c <= h1_trend:
        return None

    # H4 confluence — as of the entry instant
    if getattr(cfg, "h4_filter", False) and ctx.h4_time_idx is not None:
        h4_lookup = (entry_time - ctx.four_hours).floor("4h")
        h4_idx = ctx.h4_time_idx.get(h4_lookup)
        if h4_idx is None:
            return None
        h4f = ctx.h4_ema_fast[h4_idx]
        h4s = ctx.h4_ema_slow[h4_idx]
        if np.isnan(h4f) or np.isnan(h4s) or h4f <= h4s:
            return None

    # Structural SL anchored to the crossover bar i
    res = find_sl_with_anchor(
        opens=ctx.m30_opens, closes=ctx.m30_closes,
        highs=ctx.m30_highs, lows=ctx.m30_lows,
        crossover_idx=i, lookback=None,
    )
    if res is None:
        return None
    sl, anchor_idx = res

    ep = ctx.m30_closes[i]                    # estimate (~ next open); runner uses live mid
    if ep <= sl:
        return None
    if (ep - sl) / ep < cfg.min_risk_pct / 100.0:
        return None

    return EntrySignal(
        entry_price    = ep,
        sl_price       = sl,
        anchor_idx     = anchor_idx,
        crossover_idx  = i,
        anchor_time    = ctx.m30_times[anchor_idx],
        crossover_time = ctx.m30_times[i],
    )


# --------------------------------------------------------------------------- #
#  replay() — offline driver, parity target vs ems.engine.simulate()          #
# --------------------------------------------------------------------------- #

def replay(m30: pd.DataFrame, h1: pd.DataFrame, cfg,
           h4: Optional[pd.DataFrame] = None) -> List[Trade]:
    """
    Drive the predicates bar-by-bar to produce a trade list.

    Must equal ems.engine.simulate(m30, h1, cfg, h4) exactly — for both V2 (h4
    None) and V3 (h4 supplied + cfg.h4_filter). Proves the live predicates encode
    the same rules as the validated backtest engine.
    """
    ctx = build_ctx(m30, h1, h4)

    trades: List[Trade] = []
    in_position = False
    trade_sl = 0.0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None

    for i in range(1, len(m30)):
        t = ctx.m30_times[i]

        if in_position:
            # SL first (intrabar)
            if check_sl_hit(ctx, i, trade_sl):
                trades.append(Trade(
                    entry_time=entry_time, exit_time=t,
                    entry_price=entry_price, sl_price=trade_sl,
                    exit_price=trade_sl, exit_reason="STRUCTURAL_SL",
                    r_multiple=-1.00, strategy=cfg.strategy_name,
                ))
                in_position = False
                continue

            ex = check_h1_exit(ctx, i, entry_price, trade_sl)
            if ex is not None:
                trades.append(Trade(
                    entry_time=entry_time, exit_time=ex.exit_time,
                    entry_price=entry_price, sl_price=trade_sl,
                    exit_price=ex.exit_price, exit_reason=ex.reason,
                    r_multiple=ex.r_multiple, strategy=cfg.strategy_name,
                ))
                in_position = False
                continue
        else:
            sig = check_entry(ctx, i, cfg)
            if sig is not None:
                in_position = True
                trade_sl = sig.sl_price
                entry_price = sig.entry_price
                entry_time = t

    return trades
