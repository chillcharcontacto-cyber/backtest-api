"""
EMS V3 H4-EMA robustness check — 2-value, NOT a sweep.

Coded V3 filter = enter long only when H4 EMA(fast=20) > H4 EMA(slow).
The slow H4 EMA period was never tested: 50 or 100. Question: robust to that
choice, or knife-edge?

Run A: slow=50   Run B: slow=100   (fast=20 fixed, everything else locked at V3)
Reuses ems.engine.simulate directly.
"""
import numpy as np
import pandas as pd

from ems.config import Config
from ems.engine import simulate
from ems.indicators import (add_emas, add_h1_emas, mark_crossovers,
                            build_h4, add_h4_emas)

DATA_M30 = "data/BTCUSDT_30m.parquet"
DATA_H1  = "data/BTCUSDT_1h.parquet"
ROUND_TRIP_COST_PCT = 0.20   # %


def run(slow: int):
    m30 = mark_crossovers(add_emas(pd.read_parquet(DATA_M30), 20, 50))
    h1  = add_h1_emas(pd.read_parquet(DATA_H1), 50, 100)
    h4  = add_h4_emas(build_h4(pd.read_parquet(DATA_H1)), fast=20, slow=slow)
    cfg = Config(h4_filter=True, h4_ema_fast=20, h4_ema_slow=slow,
                 warmup_bars=500, strategy_name=f"V3-H4slow{slow}")
    return simulate(m30, h1, cfg, h4)


def metrics(trades):
    R = np.array([t.r_multiple for t in trades], dtype=float)
    n = len(R)
    wins = R[R > 0]; losses = R[R <= 0]
    ev = R.mean()
    wr = len(wins) / n
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
    total = R.sum()

    s = np.sort(R)[::-1]
    ev_m5  = s[5:].mean()  if n > 5  else float("nan")
    ev_m10 = s[10:].mean() if n > 10 else float("nan")

    eq = np.cumsum(R); peak = np.maximum.accumulate(eq); dd = eq - peak
    max_dd = dd.min()                     # most negative (R)
    longest_uw = cur = 0
    for x in dd:
        if x < -1e-9: cur += 1; longest_uw = max(longest_uw, cur)
        else: cur = 0
    lose_streak = curls = 0
    for r in R:
        if r <= 0: curls += 1; lose_streak = max(lose_streak, curls)
        else: curls = 0

    sd = R.std(ddof=1)
    sharpe = ev / sd if sd > 0 else float("nan")
    neg = np.minimum(R, 0.0); dside = np.sqrt((neg ** 2).mean())
    sortino = ev / dside if dside > 0 else float("nan")
    sp = R.std()
    skew = ((R - R.mean()) ** 3).mean() / sp ** 3 if sp > 0 else float("nan")

    # cost-adjusted EV
    costR = np.array([ROUND_TRIP_COST_PCT /
                      ((t.entry_price - t.sl_price) / t.entry_price * 100.0)
                      for t in trades])
    ev_net = (R - costR).mean()

    return dict(n=n, ev=ev, wr=wr, pf=pf, payoff=payoff, total=total,
                ev_m5=ev_m5, ev_m10=ev_m10, max_dd=max_dd,
                longest_uw=longest_uw, lose_streak=lose_streak,
                sharpe=sharpe, sortino=sortino, skew=skew,
                ev_net=ev_net, mean_cost=costR.mean())


def keyset(trades):
    return {t.entry_time.isoformat() for t in trades}


def rel_gap(a, b):
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom else 0.0


if __name__ == "__main__":
    tA = run(50)
    tB = run(100)
    mA = metrics(tA); mB = metrics(tB)

    rows = [
        ("n trades",        "{:.0f}",  "n"),
        ("EV / trade (R)",  "{:.4f}",  "ev"),
        ("win rate",        "{:.1%}",  "wr"),
        ("profit factor",   "{:.3f}",  "pf"),
        ("payoff",          "{:.3f}",  "payoff"),
        ("total R",         "{:.2f}",  "total"),
        ("EV-minus-top5",   "{:.4f}",  "ev_m5"),
        ("EV-minus-top10",  "{:.4f}",  "ev_m10"),
        ("max DD (R)",      "{:.2f}",  "max_dd"),
        ("longest DD dur",  "{:.0f}",  "longest_uw"),
        ("longest lose strk","{:.0f}", "lose_streak"),
        ("Sharpe /trade",   "{:.4f}",  "sharpe"),
        ("Sortino /trade",  "{:.4f}",  "sortino"),
        ("skew",            "{:.3f}",  "skew"),
        ("EV net 0.20%",    "{:.4f}",  "ev_net"),
        ("mean cost (R)",   "{:.4f}",  "mean_cost"),
    ]
    print(f"\n{'metric':<20}{'slow=50':>14}{'slow=100':>14}")
    print("-" * 48)
    for label, fmt, key in rows:
        print(f"{label:<20}{fmt.format(mA[key]):>14}{fmt.format(mB[key]):>14}")

    # overlap by entry date+time
    A, B = keyset(tA), keyset(tB)
    both = A & B; only50 = A - B; only100 = B - A; union = A | B
    pct = len(both) / len(union) * 100 if union else 0.0
    print("\n--- overlap (match by entry date+time) ---")
    print(f"  both:      {len(both)}")
    print(f"  50 only:   {len(only50)}")
    print(f"  100 only:  {len(only100)}")
    print(f"  union:     {len(union)}")
    print(f"  % overlap: {pct:.1f}%")

    # verdict
    ev_gap   = rel_gap(mA["ev"], mB["ev"])
    ev5_gap  = rel_gap(mA["ev_m5"], mB["ev_m5"])
    dd_gap   = rel_gap(mA["longest_uw"], mB["longest_uw"])
    print("\n--- deciding numbers ---")
    print(f"  EV gap:            {ev_gap*100:.1f}%")
    print(f"  EV-minus-top5 gap: {ev5_gap*100:.1f}%")
    print(f"  DD-duration gap:   {dd_gap*100:.1f}%")
    print(f"  overlap:           {pct:.1f}%")

    robust = (ev_gap <= 0.15 and ev5_gap <= 0.15 and dd_gap <= 0.15 and pct >= 70)
    knife  = (ev_gap > 0.25 or dd_gap > 0.25 or pct < 50)
    print("\n=== VERDICT ===")
    if robust:
        lock = 50 if mA["longest_uw"] <= mB["longest_uw"] else 100
        print(f"robust, lock period {lock}   (EV gap {ev_gap*100:.1f}%, overlap {pct:.1f}%)")
    elif knife:
        print(f"knife-edge, period-dependent   (EV gap {ev_gap*100:.1f}%, overlap {pct:.1f}%)")
    else:
        print(f"borderline — neither threshold met   (EV gap {ev_gap*100:.1f}%, overlap {pct:.1f}%)")
