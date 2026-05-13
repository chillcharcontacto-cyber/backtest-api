# Handoff

## Last Session Summary

**EMS V2 + V3 implemented, all 4 backtests run and committed**

V2 aligned engine to Samuel's canonical `RULES_EMS_AND_H4.md` — 6 rule fixes:
1. `warmup_bars=500` — no entries before bar 500 (EMA seeding protection)
2. SL lookback: unlimited (`find_sl` called with `lookback=None`)
3. H1 exit: fires only at `:30` M30 bars; exit_price = H1 close; exit_time = h1_open + 1h
4. Gap-down SL: always fill at `sl_price`, `r_multiple` forced to `-1.00`
5. exit_reason labels: `STRUCTURAL_SL` | `H1_EMA100` (was `SL` | `H1_EMA100`)
6. NaN guards on all EMA comparisons

V3 adds H4 EMA20/50 confluence filter via `--h4-filter` CLI flag.
- H4 built by resampling H1 → `resample('4h', closed='left', label='left')`
- Lookup: `(entry_ts - 4h).floor('4h')` = last fully closed H4 bar
- `strategy_name = "EMA-Cross-H4F"` when active

Output schema updated to Samuel's 11-column Quantprove format:
`trade_id, strategy, date, time, pair, direction, result, rr, duration, sl_size, exit_reason`

**Results (2017-08-17 → 2026-05-12):**

| Version | Exchange | Trades | WR | Avg R | Total R | PF |
|---|---|---|---|---|---|---|
| V2 | Binance | 976 | 27.2% | 0.584 | 569.53 | 2.20 |
| V2 | Bitstamp | 978 | 28.0% | 0.780 | 762.87 | 2.63 |
| V3 | Binance | 480 | 22.1% | 0.958 | 459.69 | 2.48 |
| V3 | Bitstamp | 475 | 23.2% | 1.254 | 595.45 | 2.94 |

V3 cuts trade count ~51% but PF and Avg R improve significantly on both exchanges.

**LB20 experiment (in-memory, no CSV):**
Tested SL lookback capped at 20 bars vs unlimited. Results identical — every qualifying crossover finds its structural SL anchor within 20 bars on this dataset.

Tests: 26/26 green. All committed and pushed (`1ad1881`).

---

## Currently Working On

Nothing — session complete.

---

## Parked / Unfinished

**EMS engine:**
- **Parity check**: compare Python V2 results against TradingView Pine strategy report (same date range)
- **`.gitignore` fix**: add `data/`, `__pycache__/`, `*.pyc`, `trades*.csv` — pycache and data dir getting committed
- **Short-side extension**: Samuel may have short rules — not yet discussed or implemented

**MCT engine (carried from previous sessions):**
- No-divergence test result still outstanding
- `max_sweep_age_bars` not implemented
- `div_found` counter cosmetic bug
- RSI divergence 4-model logic unaudited

---

## Next Steps

1. **EMS parity check** — open Pine strategy on TradingView with V2 settings, compare trade count + total R vs Python output
2. **Fix `.gitignore`** — data dir and pycache getting committed unnecessarily
3. **MCT no-divergence test** — get result, diagnose
4. **MCT `max_sweep_age_bars`** — implement + expose via `bos` indicator params
