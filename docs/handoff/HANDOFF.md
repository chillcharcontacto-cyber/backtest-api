# Handoff

## Last Session Summary

**EMS System Pine Script built (`ems_system_m30.pine`)**
Full Pine Script v5 strategy for BTCUSDT M30 trend-following system:
- EMA20/50 crossover on M30 as entry trigger
- H1 EMA50 trend filter (last confirmed H1 close)
- Structural SL: looks back from crossover candle for most recent valid bearish candle (valid = at least one subsequent candle has Higher High in both wick and body), SL placed at lowest low of that range
- Exit: first M30 bar after H1 close below H1 EMA100
- Filters: long only, one trade at a time, min 0.1% risk
- H1 new-bar detection via `ta.change(h1_t)` (timestamp, not price, to handle flat closes)
- `max_bars_back=500` required for dynamic loop indexing in `findSL()`

**EMS Python backtest engine -- Phase 1 recon complete**
Full state audit of the Pine file before any Python written:
- State inventory: 11 variables documented with types and Pine mechanisms
- State machine: FLAT/LONG with all transitions, actions, and snapshot moments
- `findSL()` algorithm documented step-by-step with correct bar-offset math
- 6 edge cases identified: EMA warmup period, H1 alignment/resampling, EMA formula (pandas `adjust=False`), `findSL` exhaustion, concurrent exit on entry bar, lb=1 boundary validity
- 5 open questions posed to user (see below) -- waiting for answers before Phase 2

---

## Currently Working On

**EMS Python engine -- Phase 2 blocked on user answers to:**
1. Data source preference (yfinance vs Binance API vs CCXT, or swappable DataLoader)
2. H1 construction: separate fetch vs resample M30 in-engine
3. CSV columns: R-multiple only or dollar P&L too
4. SL hit price: use stop price vs low of bar (gap handling)
5. Scope: how many years of BTCUSDT M30

---

## Parked / Unfinished

**MCT engine (carried from previous session):**
- No-divergence test result not received -- user was about to run MCT without `rsi_divergence` in entry_confirmations
- `max_sweep_age_bars` not implemented -- identified fix for stale sweeps where BOS fires days after SE
- `div_found` counter over-increments (cosmetic, does not affect logic)
- RSI divergence 4-model logic unaudited (`_div_in_sweep_context()`)

---

## Next Steps

1. **User answers Q1-Q5** above -> proceed to Phase 2 (EMS engine architecture)
2. **Phase 2**: propose file skeleton, function signatures, pure-function vs class split, unit test strategy per component -- pause for ack
3. **Phase 3**: implement file-by-file, single concern per commit, pytest green at every step
4. **Phase 4**: pull N years BTCUSDT M30, full run, per-trade CSV, aggregate stats
5. **MCT**: follow up on no-divergence test result
6. **MCT**: implement `max_sweep_age_bars` (suggested: 576 bars = 2 days on 5m)
