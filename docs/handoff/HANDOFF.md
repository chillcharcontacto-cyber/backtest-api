# Handoff

## Last Session Summary

**EMS Python backtest engine -- fully shipped**

All Phase 3 (implementation) and Phase 4 (full run) complete in one session.

Files created:
- `ems_system_m30.pine` -- Pine Script v5 strategy reference
- `ems/config.py` -- Config dataclass
- `ems/data.py` -- Binance klines fetch with pagination + parquet cache
- `ems/indicators.py` -- add_emas(), mark_crossovers(), add_h1_emas() (pure, df->df)
- `ems/sl_finder.py` -- find_sl() pure function, 1:1 port of Pine findSL()
- `ems/engine.py` -- Trade dataclass + simulate() pure function
- `ems/output.py` -- trades_to_csv()
- `ems/cli.py` -- main() CLI entry point
- `tests/` -- 25 pytest tests, all green

**Full backtest results -- BTCUSDT M30, 2017-08-17 -> 2026-05-12:**

| Metric | Value |
|---|---|
| Total trades | 985 |
| Win rate | 27.4% |
| Avg R | 0.51 |
| Total R | 504.40 |
| Profit factor | 2.00 |
| Exit: SL | 387 |
| Exit: H1 EMA100 | 598 |

Data cached at `data/BTCUSDT_30m.parquet` (152,837 bars) and `data/BTCUSDT_1h.parquet` (76,428 bars).
Results at `trades.csv` (985 rows).

---

## Currently Working On

Nothing -- session complete.

---

## Parked / Unfinished

**EMS engine -- next logical steps:**
- **Parity check**: load `trades.csv` into TradingView (or compare Pine strategy report) to verify trade count + R distribution matches Pine output
- **`data/` not gitignored**: parquet files committed to repo -- may want to add to `.gitignore` if data grows large
- **`__pycache__` committed**: should add to `.gitignore`

**MCT engine (carried from previous sessions):**
- No-divergence test result still outstanding -- user was about to run MCT without `rsi_divergence`
- `max_sweep_age_bars` not implemented
- `div_found` counter cosmetic bug
- RSI divergence 4-model logic unaudited

---

## Next Steps

1. **EMS parity check** -- run Pine Script strategy on TradingView for same date range, compare trade count + total R + WR against `trades.csv`
2. **Fix `.gitignore`** -- add `data/`, `__pycache__/`, `*.pyc`, `trades.csv`
3. **MCT no-divergence test** -- get result, diagnose
4. **MCT `max_sweep_age_bars`** -- implement + expose via `bos` indicator params
