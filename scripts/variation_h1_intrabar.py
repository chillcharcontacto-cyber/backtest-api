"""
V3 variation: H1 trend filter uses the CURRENT price (intrabar) instead of the last
CLOSED H1 close.

Locked model  : last CLOSED H1 close  > H1 EMA50   (must wait for the H1 to close)
Variation     : current price at the M30 cross > H1 EMA50 (take it intrabar)

The H1 EMA50 LEVEL is identical in both (EMAs only update on close) — only the price
compared against it changes. Everything else is the locked V3:
  trigger  M30 EMA20/50 bullish cross
  filter   H4 EMA20 > EMA100 (confluence)
  exit     H1 close < H1 EMA100   |   structural SL (last bearish candle pre-cross)
  long-only, min_risk 0.1%, warmup 500 bars

Outputs both the BASELINE (locked) and the VARIATION over the same fresh period so the
comparison is apples-to-apples.
"""
import csv
import os

import numpy as np
import pandas as pd

from ems.config import Config
from ems.data import fetch_ohlcv
from ems.engine import Trade
from ems.indicators import (add_emas, add_h1_emas, mark_crossovers,
                            build_h4, add_h4_emas)
from ems.sl_finder import find_sl_with_anchor
from ems_live.decider import build_ctx, check_sl_hit, check_h1_exit

MADRID = "Europe/Madrid"
SYMBOL = "BTCUSDT"
START = "2017-08-17"
END = "2026-07-22"          # through "now"
DATA_DIR = "data_now"        # separate cache so the old one isn't clobbered


def load():
    m30_raw = fetch_ohlcv(SYMBOL, "30m", START, END, DATA_DIR)
    h1_raw = fetch_ohlcv(SYMBOL, "1h", START, END, DATA_DIR)
    m30 = mark_crossovers(add_emas(m30_raw, 20, 50))
    h1 = add_h1_emas(h1_raw, 50, 100)
    h4 = add_h4_emas(build_h4(h1_raw), 20, 100)
    return m30, h1, h4


def run(m30, h1, h4, variation: bool, name: str):
    ctx = build_ctx(m30, h1, h4)
    cfg = Config(min_risk_pct=0.1, warmup_bars=500)
    one_h = pd.Timedelta(hours=1)
    four_h = pd.Timedelta(hours=4)

    trades = []
    in_pos = False
    sl = entry = 0.0
    entry_time = None

    for i in range(1, len(m30)):
        t = ctx.m30_times[i]

        if in_pos:
            if check_sl_hit(ctx, i, sl):
                trades.append(Trade(entry_time, t, entry, sl, sl,
                                    "STRUCTURAL_SL", -1.00, name))
                in_pos = False
                continue
            ex = check_h1_exit(ctx, i, entry, sl)
            if ex is not None:
                trades.append(Trade(entry_time, ex.exit_time, entry, sl,
                                    ex.exit_price, ex.reason, ex.r_multiple, name))
                in_pos = False
                continue
            continue

        # ---- entry ----
        if i <= cfg.warmup_bars or not ctx.m30_cross[i - 1]:
            continue

        h1_idx = ctx.h1_time_idx.get(t.floor("h") - one_h)   # last CLOSED H1
        if h1_idx is None:
            continue
        ema50 = ctx.h1_ema_trend[h1_idx]
        # THE ONLY DIFFERENCE:
        #   baseline  -> the last closed H1's CLOSE must be above EMA50
        #   variation -> the CURRENT price (entry px) must be above EMA50
        price_ref = ctx.m30_opens[i] if variation else ctx.h1_closes[h1_idx]
        if np.isnan(price_ref) or np.isnan(ema50) or price_ref <= ema50:
            continue

        # H4 confluence (unchanged)
        h4_idx = ctx.h4_time_idx.get((t - four_h).floor("4h"))
        if h4_idx is None:
            continue
        h4f, h4s = ctx.h4_ema_fast[h4_idx], ctx.h4_ema_slow[h4_idx]
        if np.isnan(h4f) or np.isnan(h4s) or h4f <= h4s:
            continue

        res = find_sl_with_anchor(opens=ctx.m30_opens, closes=ctx.m30_closes,
                                  highs=ctx.m30_highs, lows=ctx.m30_lows,
                                  crossover_idx=i - 1, lookback=None)
        if res is None:
            continue
        sl_px, _ = res
        ep = ctx.m30_opens[i]
        if ep <= sl_px or (ep - sl_px) / ep < cfg.min_risk_pct / 100.0:
            continue

        in_pos, sl, entry, entry_time = True, sl_px, ep, t

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
            w.writerow([i, t.strategy, o.strftime("%Y-%m-%d %H:%M:%S%z"),
                        c.strftime("%Y-%m-%d %H:%M:%S%z"), round(dur, 2), "Long",
                        "TP" if t.r_multiple > 0 else "SL", round(t.r_multiple, 2),
                        round(t.entry_price, 2), round(t.sl_price, 2),
                        round(t.exit_price, 2), round(slsz, 2), round(slpct, 4),
                        t.exit_reason])


def stats(trades):
    r = np.array([t.r_multiple for t in trades])
    w = r[r > 0]; l = r[r <= 0]
    pf = w.sum() / abs(l.sum()) if l.sum() else float("inf")
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min()
    durs = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
    return (f"n={len(r):4d}  WR={len(w)/len(r)*100:5.1f}%  EV={r.mean():+.3f}R  "
            f"PF={pf:.2f}  totalR={r.sum():+8.2f}  maxDD={dd:7.2f}R  "
            f"avg_hold={np.mean(durs):5.1f}h")


if __name__ == "__main__":
    m30, h1, h4 = load()
    print(f"\ndata: {m30.index[0]} .. {m30.index[-1]}  ({len(m30):,} M30 bars)\n")
    desktop = r"C:\Users\chill\Desktop"
    for variation, name, fn in [
        (False, "V3-locked",        "trades_v3_locked_to_now.csv"),
        (True,  "V3-H1intrabar",    "trades_v3_h1intrabar_to_now.csv"),
    ]:
        tr = run(m30, h1, h4, variation, name)
        print(f"{name:<16} {stats(tr)}")
        for base in (os.getcwd(), desktop):
            write_csv(tr, os.path.join(base, fn))
    print("\nCSVs written to repo root + Desktop.")
