# Handoff

## Last Session Summary

**Fixed Pydantic forward reference crash (500 on /openapi.json)**
`StrategyConfig` was used as a type annotation in `export_pinescript` (line 162) before being defined (line 312). Python 3.14 + Pydantic v2 strict evaluation caused PydanticUserError at schema generation. Fixed by rewriting `api.py` with all models defined before any endpoint. Deployed and confirmed working.

**Systematic debug funnel built into `engine.py::simulate()`**
Added `sl_pips_blocked_samples` (list of actual pip values blocked) and `rr_blocked_samples` (full entry/sl/ol/rr data per blocked setup). These made every subsequent diagnosis instant.

**Root causes found via funnel — chain of bugs in order:**

1. `sl_pip_blocked: 9` — all SLs were 47.5 pips, max was 35 → widened to diagnose
2. `rr_blocked: 9` — OL (TP) only 9.2 pips from entry, SL 41.7 pips → RR = 0.22
3. Root of bad RR: **stale sweeps** — BOS firing 3+ days after the sweep, price already near OL
4. **OL expiry check added** — if bear entry ≤ sweep OL (or bull entry ≥ sweep OL), skip. Filters truly expired sweeps. Adds `ol_expired` counter to debug.
5. **Divergence made optional** — if `rsi_divergence` not in `entry_confirmations`, skip div check entirely. Protocol becomes: Sweep → BOS → Session → SL → RR → Entry. `has_divergence` flag in `simulate()`.

**All fixes committed and deployed to Render.**

**Debug funnel now shows:** `sweeps_total, div_found, bos_hit, dir_mismatch, ol_expired, session_blocked, sl_nan, sl_pip_blocked, sl_pips_blocked_samples, ol_nan, rr_blocked, rr_blocked_samples, entries`

---

## Currently Working On

**No-divergence test in progress** — user is about to run JSON without `rsi_divergence` in entry_confirmations to isolate whether the rest of the pipeline produces valid trades. Result not yet received.

---

## Parked / Unfinished

- **`div_found` counter bug** — shows 279,467 from 145 sweeps (increments per-bar inside sweep loop, not per-divergence event). Cosmetic — does not affect entry logic. Low priority.
- **`max_sweep_age_bars` not implemented** — identified as the next fix: if BOS fires more than X bars after sweep SE, skip. On 5m: 1 day = 288 bars, 2 days = 576 bars. Would kill the stale-sweep RR problem at source.
- **Bear trade "tp" + "loss"** — TP was hit but 9.2 pip profit erased by 0.1% slippage round-trip. Not a bug per se, consequence of tiny RR.
- **RSI divergence logic unverified** — `_div_in_sweep_context()` not deeply audited. 4 models (M1 Regular, M2 Sweep, M3 Multiple, M4 Extended). Recommended: build Pine indicator for divergence in separate chat, verify visually on TradingView, then confirm Python logic matches.

---

## Next Steps

1. **Get no-divergence test result** — if entries > 0 with normal SL/RR filters, pipeline is sound and divergence is the isolated problem.
2. **Implement `max_sweep_age_bars`** — add param to `simulate()`, expose via `bos` indicator params in entry_confirmations. Start with 576 bars (2 days on 5m).
3. **If no-divergence works** → run Pine Script export (`/export/pinescript`) → load in TradingView → visual check.
4. **Audit RSI divergence** — either in separate Pine chart or by adding more debug to `_div_in_sweep_context()` showing which bars/models are firing and why.
5. **Once engine verified**: mass backtest across multiple pairs/date ranges.
