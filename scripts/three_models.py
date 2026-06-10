"""
Three EMS models per user spec (2026-06-05) — long-only BTC, Binance.

ALL models share:
  Trigger : M30 EMA20/50 bullish cross
  Entry   : next M30 bar open after the cross
  SL      : below last bearish candle before the cross (structural)
  Exit    : H1 close below H1 EMA50   <-- NOTE: EMA50, per spec (prior code used EMA100)
  Filter  : H1 EMA50 below price (last closed H1 close)  [H1 trend gate]

  Base       : the above, no H4 filter
  Refined 1  : + H4 EMA50  below price (price = last closed H1 close)
  Refined 2  : + H4 EMA100 below price (price = last closed H1 close)

Reuses the V2 entry/exit predicates from ems_live.decider (parity-guaranteed),
adds the H4 price gate here. H4 built from H1 (signal source = Binance).
"""
import csv
import os

import numpy as np
import pandas as pd

from ems.config import Config
from ems.engine import Trade
from ems.indicators import add_emas, add_h1_emas, mark_crossovers, build_h4
from ems_live.decider import build_ctx, check_entry, check_sl_hit, check_h1_exit

MADRID = "Europe/Madrid"
EXIT_EMA = 50          # H1 exit EMA per spec
TREND_EMA = 50         # H1 trend filter EMA
DATA_M30 = "data/BTCUSDT_30m.parquet"
DATA_H1  = "data/BTCUSDT_1h.parquet"


def load():
    m30 = mark_crossovers(add_emas(pd.read_parquet(DATA_M30), 20, 50))
    h1_raw = pd.read_parquet(DATA_H1)
    h1 = add_h1_emas(h1_raw, trend=TREND_EMA, exit_=EXIT_EMA)
    h4_raw = build_h4(h1_raw)
    return m30, h1, h4_raw


def h4_with_ema(h4_raw, period):
    h4 = h4_raw.copy()
    h4["h4ema"] = h4["close"].ewm(span=period, adjust=False).mean()
    return h4


def run(m30, h1, h4_raw, h4_period, strategy_name):
    """h4_period=None -> base (no H4 gate)."""
    ctx = build_ctx(m30, h1)
    one_hour = pd.Timedelta(hours=1)
    four_hours = pd.Timedelta(hours=4)

    use_h4 = h4_period is not None
    if use_h4:
        h4 = h4_with_ema(h4_raw, h4_period)
        h4_ema = h4["h4ema"].to_numpy(dtype=float)
        h4_idx = {t: i for i, t in enumerate(h4.index)}

    trades = []
    in_pos = False
    trade_sl = entry_price = 0.0
    entry_time = None

    for i in range(1, len(m30)):
        t = ctx.m30_times[i]
        if in_pos:
            if check_sl_hit(ctx, i, trade_sl):
                trades.append(Trade(entry_time, t, entry_price, trade_sl, trade_sl,
                                    "STRUCTURAL_SL", -1.00, strategy_name))
                in_pos = False
                continue
            ex = check_h1_exit(ctx, i, entry_price, trade_sl)
            if ex is not None:
                trades.append(Trade(entry_time, ex.exit_time, entry_price, trade_sl,
                                    ex.exit_price, ex.reason, ex.r_multiple, strategy_name))
                in_pos = False
                continue
        else:
            sig = check_entry(ctx, i, Config(min_risk_pct=0.1, warmup_bars=500))
            if sig is None:
                continue
            if use_h4:
                # price = last closed H1 close (same lookup as the H1 trend gate)
                h1_lookup = t.floor("h") - one_hour
                h1i = ctx.h1_time_idx.get(h1_lookup)
                if h1i is None:
                    continue
                price = ctx.h1_closes[h1i]
                # last closed H4 EMA
                h4l = (t - four_hours).floor("4h")
                hi = h4_idx.get(h4l)
                if hi is None:
                    continue
                if np.isnan(h4_ema[hi]) or price <= h4_ema[hi]:
                    continue
            in_pos = True
            trade_sl = sig.sl_price
            entry_price = sig.entry_price
            entry_time = t

    return trades


def write_csv(trades, path):
    cols = ["trade_id", "strategy", "open_madrid", "close_madrid", "duration_hours",
            "direction", "result", "rr", "entry_price", "sl_price", "exit_price",
            "sl_size", "sl_pct", "exit_reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for i, t in enumerate(trades, 1):
            o = t.entry_time.tz_convert(MADRID); c = t.exit_time.tz_convert(MADRID)
            dur = (t.exit_time - t.entry_time).total_seconds() / 3600
            slsz = abs(t.entry_price - t.sl_price); slpct = slsz / t.entry_price * 100
            res = "TP" if t.r_multiple > 0 else "SL"
            w.writerow([i, t.strategy, o.strftime("%Y-%m-%d %H:%M:%S%z"),
                        c.strftime("%Y-%m-%d %H:%M:%S%z"), round(dur, 2), "Long", res,
                        round(t.r_multiple, 2), round(t.entry_price, 2),
                        round(t.sl_price, 2), round(t.exit_price, 2),
                        round(slsz, 2), round(slpct, 4), t.exit_reason])


def summary(trades):
    rs = np.array([t.r_multiple for t in trades])
    wins = rs[rs > 0]; losses = rs[rs <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    durs = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
    return (f"n={len(trades)}  WR={len(wins)/len(rs)*100:.1f}%  totalR={rs.sum():.2f}  "
            f"PF={pf:.2f}  EV={rs.mean():.3f}  avg_hold={np.mean(durs):.1f}h  max_hold={max(durs):.0f}h")


if __name__ == "__main__":
    m30, h1, h4_raw = load()
    models = [
        (None, "EMS-base",          "trades_base_binance.csv"),
        (50,   "EMS-r1-h4ema50",    "trades_refined1_h4ema50_binance.csv"),
        (100,  "EMS-r2-h4ema100",   "trades_refined2_h4ema100_binance.csv"),
    ]
    desktop = r"C:\Users\chill\Desktop"
    for period, name, fn in models:
        tr = run(m30, h1, h4_raw, period, name)
        print(f"{name:<18} {summary(tr)}")
        for base in [os.getcwd(), desktop]:
            write_csv(tr, os.path.join(base, fn))
    print("\nCSVs written to repo root + Desktop.")
