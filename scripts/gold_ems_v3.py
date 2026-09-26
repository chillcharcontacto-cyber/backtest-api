"""
EMS V3 locked (winning model) on GOLD (XAU/USD), highest-quality Dukascopy-tick M5
(2016-2026, ~10y, already cached by the swing project) resampled to M30/H1.
Same model code as BTC: M30 EMA20/50 cross + H1 close>EMA50 + H4 EMA20>EMA100,
structural SL, H1 EMA100 exit, LONG-ONLY.
"""
import glob, csv, os
import numpy as np
import pandas as pd

from ems.indicators import add_emas, add_h1_emas, mark_crossovers, build_h4, add_h4_emas
from ems.sl_finder import find_sl_with_anchor
from ems_live.decider import build_ctx, check_sl_hit, check_h1_exit

SWING = r"C:/Users/chill/Desktop/MIS COSAS/TradingEdgeLabs/strategies/swing/data_cache"
DESK = r"C:/Users/chill/Desktop"
WARMUP, MINRISK = 500, 0.1

def resample(df, rule):
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()

m5 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(SWING + "/XAUUSD*m5.parquet"))]).sort_index()
m5 = m5[~m5.index.duplicated(keep="first")]
m5["volume"] = 0.0   # Dukascopy M5 is OHLC-only; volume unused for signals (EMA/cross)
h1_raw = resample(m5, "60min")
m30 = mark_crossovers(add_emas(resample(m5, "30min"), 20, 50))
h1 = add_h1_emas(h1_raw, 50, 100)
h4 = add_h4_emas(build_h4(h1_raw), 20, 100)
ctx = build_ctx(m30, h1, h4)
print(f"GOLD M5 {len(m5):,} -> M30 {len(m30):,} / H1 {len(h1):,}   {m30.index[0]} .. {m30.index[-1]}")

one_h, four_h = pd.Timedelta(hours=1), pd.Timedelta(hours=4)
trades = []
in_pos = False; sl = entry = 0.0; entry_time = None
for i in range(1, len(m30)):
    t = ctx.m30_times[i]
    if in_pos:
        if check_sl_hit(ctx, i, sl):
            trades.append((entry_time, t, entry, sl, sl, "STRUCTURAL_SL", -1.0)); in_pos = False; continue
        ex = check_h1_exit(ctx, i, entry, sl)
        if ex is not None:
            trades.append((entry_time, ex.exit_time, entry, sl, ex.exit_price, ex.reason, ex.r_multiple)); in_pos = False; continue
        continue
    if i <= WARMUP or not ctx.m30_cross[i - 1]:
        continue
    h1_idx = ctx.h1_time_idx.get(t.floor("h") - one_h)
    if h1_idx is None: continue
    ema50 = ctx.h1_ema_trend[h1_idx]; pref = ctx.h1_closes[h1_idx]
    if np.isnan(pref) or np.isnan(ema50) or pref <= ema50: continue
    h4_idx = ctx.h4_time_idx.get((t - four_h).floor("4h"))
    if h4_idx is None: continue
    h4f, h4s = ctx.h4_ema_fast[h4_idx], ctx.h4_ema_slow[h4_idx]
    if np.isnan(h4f) or np.isnan(h4s) or h4f <= h4s: continue
    res = find_sl_with_anchor(opens=ctx.m30_opens, closes=ctx.m30_closes,
                              highs=ctx.m30_highs, lows=ctx.m30_lows,
                              crossover_idx=i - 1, lookback=None)
    if res is None: continue
    sl_px, _ = res; ep = ctx.m30_opens[i]
    if ep <= sl_px or (ep - sl_px) / ep < MINRISK / 100.0: continue
    in_pos, sl, entry, entry_time = True, sl_px, ep, t

r = np.array([tr[6] for tr in trades])
w = r[r > 0]; l = r[r <= 0]
pf = w.sum() / abs(l.sum()) if l.sum() else float("inf")
eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min()
durs = [(tr[1] - tr[0]).total_seconds() / 3600 for tr in trades]
# net of HL-style fee for comparability (fee_R = 0.09/stop%); gold stop% from each trade
slpcts = np.array([abs(tr[2] - tr[3]) / tr[2] * 100 for tr in trades])
netR = r - 0.09 / slpcts
print(f"\nXAUUSD  V3-locked (LONG-ONLY)  2016-06 .. 2026-06  (~10y, Dukascopy tick M30)")
print(f"  n={len(r)}  WR={len(w)/len(r)*100:.1f}%  EV={r.mean():+.3f}R  netEV={netR.mean():+.3f}R  PF={pf:.2f}")
print(f"  totalR={r.sum():+.1f}  netTotalR={netR.sum():+.1f}  maxDD={dd:.1f}R  avg_hold={np.mean(durs):.1f}h")
print(f"  biggest winner +{r.max():.1f}R  biggest loser {r.min():.1f}R  median stop%={np.median(slpcts):.2f}")

MADRID = "Europe/Madrid"
out = os.path.join(DESK, "trades_ems_v3_XAUUSD_10y.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["trade_id", "open_madrid", "close_madrid", "duration_h", "result",
                 "rr", "entry", "sl", "exit", "sl_pct", "exit_reason"])
    for idx, tr in enumerate(trades, 1):
        et, xt, en, slp, ex, rs, rm = tr
        o = et.tz_convert(MADRID); c = xt.tz_convert(MADRID)
        wr.writerow([idx, o.strftime("%Y-%m-%d %H:%M%z"), c.strftime("%Y-%m-%d %H:%M%z"),
                     round((xt - et).total_seconds() / 3600, 1), "TP" if rm > 0 else "SL",
                     round(rm, 2), round(en, 2), round(slp, 2), round(ex, 2),
                     round(abs(en - slp) / en * 100, 3), rs])
print("CSV ->", out)
