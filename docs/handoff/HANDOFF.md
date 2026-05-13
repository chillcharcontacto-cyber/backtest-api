# Handoff

## Last Session Summary

**Bitstamp BTCUSD M30 backtest added and run**

- Added `fetch_ohlcv_bitstamp()` to `ems/data.py`
  - Bitstamp API quirk: ignores `start` param, returns 1000 bars ending at `end`
  - Fixed with backwards pagination: walk `end` backwards until `start_ts` covered
- Updated `ems/cli.py` with `--exchange binance|bitstamp` flag (auto-sets symbol)
- Output: `trades_bitstamp.csv`

**Results comparison -- same strategy, same date range (2017-08-17 -> 2026-05-12):**

| Metric | Binance BTCUSDT | Bitstamp BTCUSD |
|---|---|---|
| Total trades | 985 | 982 |
| Win rate | 27.4% | 28.2% |
| Avg R | 0.51 | 0.72 |
| Total R | 504.40 | 711.01 |
| Profit factor | 2.00 | 2.43 |
| Exit: SL | 387 | 368 |
| Exit: H1 EMA100 | 598 | 614 |

Trade counts are very close (~3 difference) -- expected given minor price differences between exchanges. R results diverge because BTCUSD and BTCUSDT prices differ at exact entry/exit bars.

All committed and pushed (`0ef1cf0`). 25 tests still green.

---

## Currently Working On

Nothing -- session complete.

---

## Parked / Unfinished

**EMS engine:**
- **Parity check**: compare Python results against TradingView Pine strategy report (same date range)
- **`.gitignore` fix**: add `data/`, `__pycache__/`, `*.pyc`, `trades*.csv`

**MCT engine (carried from previous sessions):**
- No-divergence test result still outstanding
- `max_sweep_age_bars` not implemented
- `div_found` counter cosmetic bug
- RSI divergence 4-model logic unaudited

---

## Next Steps

1. **EMS parity check** -- open Pine strategy on TradingView, same dates, compare trade count + total R
2. **Fix `.gitignore`** -- data dir and pycache getting committed
3. **MCT no-divergence test** -- get result, diagnose
4. **MCT `max_sweep_age_bars`** -- implement + expose via `bos` indicator params
