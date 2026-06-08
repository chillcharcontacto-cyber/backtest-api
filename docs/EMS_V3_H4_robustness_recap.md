# EMS V3 — H4 EMA Robustness Check (recap)

**Date:** 2026-06-05
**Asset / strategy:** BTC, EMS V3 (EMA-Cross, long-only)
**Question:** Is the V3 H4-EMA filter robust to the EMA period choice, or knife-edge?
**Answer:** **ROBUST — lock period 100** (EV gap 4.2%, overlap 78.7%).

---

## Context

EMS is a long-only BTC trend system:
- **Entry:** M30 EMA20/50 bullish crossover, gated by an H1 trend filter (H1 close > H1 EMA50).
- **Exit:** structural stop loss, or H1 EMA100 exit.
- **V2** = the validated base engine.
- **V3** = V2 + an **H4 confluence filter**: only enter long when **H4 EMA(fast=20) > H4 EMA(slow)**.

The slow H4 EMA period was never tested — it's either **50** or **100**. This is a
**2-value robustness check, not a sweep.** Two sensible values agreeing is a genuine
robustness result, not multiple-testing.

## Method

- Reused the shipped `ems.engine.simulate` V3 path directly (no reimplementation).
- Data: cached Binance **BTCUSDT** M30 + H1, full history, warmup 500 bars.
- Everything locked at V3 except the slow H4 EMA period.
  - **Run A:** fast=20, **slow=50**
  - **Run B:** fast=20, **slow=100**
- Trade R = `r_multiple` (STRUCTURAL_SL always −1.00; H1_EMA100 exits signed R).
- Sanity: Run A reproduces the previously-committed V3 Binance run exactly
  (480 trades, PF 2.48, total R 459.69). Engine reuse confirmed.

## Results

| metric | slow=50 | slow=100 |
|---|---|---|
| n trades | 480 | 487 |
| EV / trade (R) | 0.958 | 1.000 |
| win rate | 22.1% | 23.2% |
| profit factor | 2.475 | 2.627 |
| payoff | 8.73 | 8.70 |
| total R | 459.7 | 487.1 |
| EV-minus-top5 | 0.388 | 0.439 |
| EV-minus-top10 | 0.056 | 0.092 |
| max DD (R) | −27.9 | −31.5 |
| longest DD duration (trades) | 70 | 62 |
| longest losing streak | 14 | 17 |
| Sharpe / trade | 0.135 | 0.139 |
| Sortino / trade | 1.235 | 1.334 |
| skew | 6.78 | 6.58 |
| **EV net of 0.20% round-trip** | **0.647** | **0.697** |
| mean cost (R) | 0.311 | 0.303 |

Cost model: `cost_in_R = 0.20 / (sl_size as % of entry)`, subtracted per trade.

## Overlap (matched by entry date+time)

- both: **426**
- 50-only: 54
- 100-only: 61
- union: 541
- **% overlap = 78.7%**

High overlap → the period is **cosmetic**, not structural. Both periods select
essentially the same trades.

## Deciding numbers vs thresholds

| | value | ROBUST threshold |
|---|---|---|
| EV gap | 4.2% | ≤ 15% ✅ |
| EV-minus-top5 gap | 11.6% | ≤ 15% ✅ |
| DD-duration gap | 11.4% | ≤ 15% ✅ |
| overlap | 78.7% | ≥ 70% ✅ |

(KNIFE-EDGE would be: EV or DD gap > 25%, or overlap < 50%. None hit.)

## VERDICT

> **robust, lock period 100 — EV gap 4.2%, overlap 78.7%**

The filter is real, not knife-edge. The two periods take the same ~79% of trades and
perform within noise; the slow-EMA period is cosmetic. Tiebreak to **100** on the
spec's rule (shorter DD duration: 62 vs 70); 100 also edges EV, PF, total R, Sortino.

**Nuances (both periods share these — not robustness fails):**
- 100 has marginally deeper max-DD-in-R (−31.5 vs −27.9) and a longer losing streak
  (17 vs 14), despite the shorter underwater duration.
- EV-minus-top10 collapses to ~0.06–0.09 R: a meaningful chunk of edge lives in the
  top ~10 winners (skew ~6.6–6.8, payoff ~8.7). The edge is **convexity / fat-tail
  driven** — real, but lumpy. Both periods agree on this shape.
- EV survives the 0.20% round-trip cost comfortably (net 0.65–0.70 R/trade).

## Reproduce

```
python -m scripts.robustness_h4_v3
```

(Harness: `scripts/robustness_h4_v3.py`. Requires cached
`data/BTCUSDT_30m.parquet` + `data/BTCUSDT_1h.parquet`.)
