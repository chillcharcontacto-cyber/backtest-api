# Handoff

## Last Session Summary

**Autonomy hardening — V3 bot is now safe to run unattended. Render blueprint saved.**

All CODE for full-autonomous testnet operation is done. Remaining work is the Render
deploy (user actions — see Next Steps, START AT STEP 4). Bot stays `dry_run=True` by
default; arming is a single env-var flip on Render.

Built this session:
1. **Per-tick exchange reconcile** (`runner.tick`, live) — when IN_POSITION, check the
   exchange first; if the position is gone, the resting stop fired → record −1R, go
   flat. Closes a real gap: a stop fill was previously only detected on restart, so a
   running bot would wrongly believe it was still in a position. Dry path simulates the
   intrabar stop hit on the bar low.
2. **Max-daily-loss kill switch** (`ems_live/daylimit.py`) — `DayLedger` tracks realized
   R per UTC day, rolls at the date boundary; entries halt for the rest of the day once
   realized R ≤ −`max_daily_loss_r` (config default 3.0; 0 disables). R recorded on
   every close (SL fill or H1 exit).
3. **Render worker blueprint** (`render.yaml`) — `type: worker`,
   `startCommand: python -m ems_live.run`, persistent disk at `/data` (position +
   day-ledger survive deploys). Env preset: HL_TESTNET=true, EMS_DRY_RUN=true, risk 20,
   kill-switch 3R, EMS_STATE_PATH=/data/...; HL_MASTER_ADDRESS + HL_AGENT_KEY are
   `sync: false` (dashboard secrets, never committed).
4. **Entrypoint env** — `run.py` reads EMS_STATE_PATH + EMS_MAX_DAILY_LOSS_R.

Tests: **58 green** (+9 daylimit). Dry tick verified end-to-end. Commit `5a29d53`.

Prior in this thread: V3 locked as final model + bot converted to V3 + entrypoint
(`97a4954`, `55ce69c`, `4b28b70`).

---

## Previous Session Summary — V3 locked + bot converted to V3 + entrypoint

Final model LOCKED = **V3 confluence, H4 EMA20>EMA100, H1 EMA100 exit** (wins every
quality metric; R2's lower historical DD is an MC-p95 mirage). Decider gained
`check_h4_confluence` (mirrors `engine.simulate`); V3 parity test byte-identical;
config `h4_filter=True, slow=100, EMA-Cross-H4F`; entrypoint `ems_live/run.py`. Three
alt price-filter models built/compared + rejected (`scripts/three_models.py`).

---

## Older Session Summary — V3 trade CSVs with Madrid timestamps (swap-cost prep)

Generated V3 CSVs with exact Europe/Madrid open/close timestamps + `sl_pct`
(`cost_in_R = cost% / sl_pct`): `trades_ems_v3_binance_h4ema{50,100}_madrid.csv`
(`5179be5`). For modeling funding vs holding duration (avg ~33h, max 290–362h).

---

## Earlier Session Summary — V3 H4-EMA robustness (lock period 100)

**EMS V3 H4-EMA robustness check — VERDICT: robust, lock period 100**

Detour from the bot build (user request) to validate the V3 H4 filter before
relying on it. 2-value robustness check (NOT a sweep) on the coded V3 confluence
filter (H4 EMA fast=20 > slow), varying the slow period 50 vs 100. Reused
`ems.engine.simulate` directly on cached Binance BTCUSDT. Run A (slow=50) reproduced
the committed V3 Binance run exactly (480 trades, PF 2.48, total R 459.69).

Deciding numbers: EV gap **4.2%**, EV-minus-top5 gap 11.6%, DD-duration gap 11.4%,
**overlap 78.7%** — all inside ROBUST thresholds (≤15% gaps, ≥70% overlap). The
slow-EMA period is cosmetic, not structural; the filter is real. Locked **period 100**
(shorter DD duration 62 vs 70; also edges EV/PF/total R/Sortino). EV survives 0.20%
round-trip cost (net 0.65–0.70 R/trade). Caveat: EV-minus-top10 ≈ 0.06–0.09 R — edge
is fat-tail/convexity driven (skew ~6.7, payoff ~8.7), shared by both periods.

Artifacts: `scripts/robustness_h4_v3.py` (reproducible harness),
`docs/EMS_V3_H4_robustness_recap.md` (full recap for sharing). Recap also exported to
`C:\Users\chill\Desktop\EMS_V3_H4_robustness_recap.md` (Desktop root) for easy opening
outside the repo — the in-repo copy was buried in the worktree path and wouldn't open.

**ACTION FOR NEXT SESSION:** robustness done → resume the automated HL bot at
**Phase 4, step 1** (build `ems_live/run.py` entrypoint). See Next Steps below.

---

## Oldest Session Summary — live bot Phases 2+3 (testnet lifecycle)

**EMS live Hyperliquid bot: Phases 2 + 3 shipped — trades autonomously on TESTNET**

Built on top of the Phase 0+1 foundation. The bot can now run a full
entry→stop→exit lifecycle with real testnet orders. `dry_run` defaults True, so
autonomous trading is opt-in (flip to False to arm).

**Phase 2 — state machine + scheduler** (`position.py`, `runner.py`):
- `PositionState` + atomic JSON `PositionStore`
- `reconcile()` vs exchange truth (4-case matrix): flat/flat=OK_FLAT;
  inpos/inpos=OK_RESUME (keep local SL meta); inpos/exch-flat=CLOSED_WHILE_DOWN;
  flat/exch-pos=UNEXPECTED_POSITION
- `compute_size()` fixed-$ risk: `size = risk_usd / (entry - sl_hl)`
- `guard_order()` safety gate: sl<entry, risk band [min_risk_pct, max_risk_band_pct],
  absolute `max_notional` ceiling, positive size
- `tick()`, `boot_reconcile()`, `seconds_until_next_bar()`, `run_forever()` loop
- config gained: max_notional_usd, max_risk_band_pct, leverage, isolated_margin,
  dry_run, poll_buffer_sec

**Phase 3 — live order execution** (`broker.py`):
- `market_entry()` (market_open long, returns avg fill px), `place_stop()`
  (reduce-only stop-market trigger, returns resting oid), `market_close()`,
  `cancel_order()`, `set_margin_mode()` (leverage + isolated)
- `account_value()` sums perp + spot USDC (Unified Account: perp marginSummary
  reads 0 while collateral sits in spot)
- `round_px()` enforces HL perp rule: 5 significant figures AND ≤(6-szDecimals)
  decimals → BTC ~65k uses integer prices
- runner live path: capture actual avg_px; **stop-confirmed-or-flatten** (fill
  without working stop → immediate market_close); cancel resting stop on H1 exit

**Testnet verification (real fake-money orders):**
- direct broker lifecycle: entry filled, stop rested, close, flat ✅
- forced runner tick: SL adaptation + sizing + guards + persistence + stop_oid →
  position opened & protected → closed clean ✅
- caught a real bug live: 6-sig-fig stop px rejected ("Invalid TP/SL price");
  fixed `round_px`, dangling position closed safely, re-tested green
- proved agent safety model: agent CAN trade (update_leverage ok) but CANNOT move
  funds (usd_class_transfer rejected, keyed to agent) — exactly as designed

Tests: **48 green** (+13: reconcile matrix, store roundtrip, sizing, guards, scheduler).
Commits pushed: Phase 2 `7759439`, Phase 3 `8e82fb8`.

---

## Currently Working On

Nothing mid-flight. Bot is **V3, parity-locked, autonomy-hardened** (per-tick
reconcile + kill switch), Render blueprint saved, left in `dry_run=True` (safe).
**All code for unattended testnet operation is done — remaining is the Render deploy
(Next Steps, START AT STEP 4).** Account funded + configured:
- Testnet/master address: `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`
- Testnet equity: **~999 mock USDC** (Unified Account — usable for perps directly)
- Agent (API) wallet `ems-bot` authorized, BTC leverage 3x isolated
- Credentials in gitignored `.env` (HL_MASTER_ADDRESS, HL_AGENT_KEY, HL_TESTNET=true)
- Mainnet: 9.8 USDC in spot (the deposit that unlocked the faucet gate)
- Run locally: `python -m ems_live.run once` (single tick) / `python -m ems_live.run` (forever)

---

## Parked / Unfinished

**EMS live bot — code DONE (V3, autonomy-hardened); remaining is deploy + mainnet:**
- **Render deploy** — see Next Steps (the blueprint + worker are already in
  `render.yaml`; just connect on Render + set the two secrets).
- **Missed-bar catch-up on restart** — runner resumes at next bar; if down across a
  `:30` H1-exit bar, that exit is skipped. Still parked (low priority — stop rests on
  exchange so positions stay protected; only the computed H1 exit can be missed).
- **Phase 5: mainnet.** Flip HL_TESTNET=false, transfer real USDC spot→perp (master-
  signed, in UI — agent can't), tiny `risk_usd`, mainnet dry-run first, then arm.

DONE (no longer parked): max-daily-loss kill switch, per-tick stop-fill reconcile,
Render worker blueprint, entrypoint.

**EMS engine (pre-existing):**
- Parity check vs TradingView Pine strategy report
- Short-side extension — Samuel may have short rules, not discussed
- **Swap/funding cost analysis** — only a flat 0.20% round-trip fee has been tested;
  perp funding is unmodeled. Madrid-timestamped CSVs are ready
  (`trades_ems_v3_binance_h4ema{50,100}_madrid.csv`). To compute funding-in-R per
  trade, join a HL BTC funding-rate series to each [open, close] window
  (`funding_in_R ≈ Σ funding_rate / sl_pct`). Offered to build the filter script.

**MCT engine (carried):**
- No-divergence test result outstanding
- `max_sweep_age_bars` not implemented
- `div_found` counter cosmetic bug
- RSI divergence 4-model logic unaudited

---

## Next Steps — START AT STEP 4 (steps 1–3 are DONE: V3, parity, autonomy code)

**4. Deploy the Render worker (dry).** The blueprint is already in `render.yaml`.
   - Render dashboard → **Blueprints** (or New + → Blueprint) → connect this repo →
     it reads `render.yaml` and creates the **`ems-live-bot`** worker.
   - Paste the two secrets it prompts for (`sync: false`):
     **HL_MASTER_ADDRESS** = `0x18ce…6964`, **HL_AGENT_KEY** = the testnet agent key.
   - Presets already set: HL_TESTNET=true, EMS_DRY_RUN=true, risk 20, kill-switch 3R,
     state on `/data`. Apply → Deploy.

**5. Watch logs** — confirm the `[run]` config line, then a tick each :00/:30
   (fetch → evaluate → "no signal"). Validates the unattended scheduler.

**6. Soak dry ~1 day** — zero risk, no orders placed.

**7. ARM** — worker → Environment → set `EMS_DRY_RUN=false` → Save (auto-redeploys).
   Now it places real testnet orders when a signal fires. (H4 gate currently blocks
   longs until BTC H4 EMA20 > EMA100, so an entry won't fire until H4 turns up.)

**8. Verify first autonomous trade** — check entry + resting stop on the testnet UI.

**Later — Phase 5 mainnet:** only after a clean multi-day armed testnet soak. Flip
HL_TESTNET=false, transfer real USDC spot→perp in the UI (master-signed; agent can't),
tiny risk_usd, one mainnet dry-run, then arm live.

Heads-up: Render worker + persistent disk is **paid** (~$7/mo). Optional: run
`python -m ems_live.run` locally first to watch it live before paying.
