# Decisions Log

A running list of architectural and product decisions, newest first.

---

## 2026-05-13

**EMS engine: all pure functions, no mutable class state**
Config and Trade are dataclasses (Trade is frozen). Data fetch, indicators, SL finder, and simulate() are all pure functions. Given same inputs -> same outputs. No global state, no side effects outside output.py. Chosen for testability and reproducibility.

**EMS H1 alignment: `floor(t, 1h) - 1h` formula**
For any M30 bar at time t, last closed H1 bar open = floor(t, 1h) - 1h. Works for both :00 and :30 M30 bars (both see the same last-closed H1). h1_new_bar = (t.minute == 0). This mirrors Pine's `close[1]` + `lookahead_on` behavior without resampling.

**EMS SL gap-down handling: fill at open if open < stop**
If bar opens below the stop level (gap down), exit price = open[i], not trade_sl. Prevents unrealistic fills at prices that were never traded.

**EMS data: Binance public klines, parquet cache**
No auth required. Fetches 1000 bars/request with 50ms sleep between requests. First run: ~230 HTTP requests for M30+H1 2017-2026. Subsequent runs: instant from parquet cache. pyarrow added to requirements.txt.

---

## 2026-05-12

**EMS System built as Pine Script first, Python port second**
`ems_system_m30.pine` written and committed before any Python. Consistent with the pine-first methodology decided 2026-05-10. Python engine will be validated against TradingView backtest output for parity.

**EMS Python engine is a separate project from the MCT engine**
EMS is a standalone trend-following system (EMA crossover + H1 filter + structural SL). It shares no code or data structures with the MCT sequential protocol. Will live in its own files, not inside `engine.py`.

**EMS SL algorithm: "valid bearish" requires HH confirmation in both wick and body**
Structural SL looks back from crossover for a bearish candle where at least one subsequent candle (up to and including crossover bar) has `high > bearish.high` AND `close > bearish.open`. SL = lowest low from that candle to crossover inclusive. lb=1 case (bearish immediately before crossover) is valid if crossover bar itself satisfies the HH condition -- confirmed intentional per spec, needs user sign-off in Q6.

---

## 2026-05-10

**OL validity check — skip sweep if price already past the target**
Bear entry must be above `sw.ol_price`; bull entry must be below it. If BOS fires days after the sweep, price may have already moved through the OL, leaving no valid TP target. Filter added in `simulate()` before SL/RR checks. Adds `ol_expired` counter to debug funnel.

**RSI divergence is now optional — controlled by entry_confirmations**
`has_divergence` flag in `simulate()`. If `rsi_divergence` is not in `entry_confirmations`, the divergence step is skipped entirely. Protocol runs: Sweep → BOS → Session → SL → RR → Entry. Allows isolating divergence as a variable for testing and debugging. `div_bar` defaults to `sw.se_bar` when skipped so downstream context logging still works.

**Debug funnel must include sample values, not just counts**
Added `sl_pips_blocked_samples` and `rr_blocked_samples` (with full entry/sl/ol prices and computed ratios) to the debug output. Counts alone don't tell you *why* — sample values made every diagnosis instant. Pattern: always log sample data alongside blocked counts.

**`max_sweep_age_bars` identified as next required fix (not yet implemented)**
Stale sweeps (BOS 3 days after SE) produce near-zero RR because price is already near the OL. Fixing OL expiry helps, but doesn't address setups where price is close to OL but not past it. A bar-count limit per sweep (e.g. 576 bars = 2 days on 5m) is the cleaner solution. Deferred to next session.

**Pine-first workflow is better methodology for new systems**
Reviewed a friend's prompt that builds Pine Script first, ports to Python, then does a parity check. Agreed this is better than Python-first. Decision: apply this to any new strategy from scratch. Current MCT engine stays in Python — too deep to restart. Use Pine Script export as the parity check tool instead.

---

## 2026-05-09

**SL filter is forex-specific — only runs when `sl_filter` is explicitly in entry_confirmations**
The SL pip-range check in `simulate()` was running unconditionally for all markets. Decided to make it conditional: it only fires when the user explicitly adds `sl_filter` to their strategy config. This allows the MCT engine to work on stocks and crypto without a forex-specific filter blocking everything. The RR filter (`min_rr`) stays always-on because it's unit-agnostic and is a fundamental part of the MCT protocol.

**`pip_size` canonical default is `0.0001` (standard 4-decimal forex pip)**
Corrected from `0.00001` everywhere: `engine.py` defaults, `index.html` frontend defaults, and `DEFAULT_CATALOG` in the frontend. The `0.00001` value was a typo/error — it represents a pipette (1/10 of a pip), making the SL range 10x tighter than intended.

**Git workflow: all changes committed and pushed directly to `main`**
Claude Code edits files in the worktree, commits, and pushes to `origin/main`. Render auto-deploys the backend; Vercel auto-deploys the frontend. No manual file uploads needed going forward.
