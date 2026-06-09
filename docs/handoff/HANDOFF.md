# Handoff

## Last Session Summary

**Generated the missing V3 H4-EMA slow=100 trades CSV**

User had only the slow=50 V3 Binance CSV; the robustness winner (slow=100) had no
export. Reran `ems.engine.simulate` with `h4_ema_slow=100` on cached Binance BTCUSDT,
wrote the 11-column Quantprove schema. Result matches robustness exactly: **487 trades,
WR 23.2%, total R 487.05, PF 2.63**.

Output: `trades_ems_v3_binance_h4ema100.csv` (committed `e97c27e`) + Desktop copy at
`C:\Users\chill\Desktop\trades_ems_v3_binance_h4ema100.csv`. Companion to the existing
slow=50 file (`trades_ems_v3_binance.csv`, 480 trades). No code changes.

**ACTION FOR NEXT SESSION:** resume the automated HL bot at **Phase 4, step 1**
(build `ems_live/run.py` entrypoint). See Next Steps below.

---

## Previous Session Summary

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

## Earlier Session Summary

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

Nothing mid-flight. Phase 3 complete and committed. Bot is armed-capable on testnet
but left in `dry_run=True` (safe default). Account funded + configured:
- Testnet/master address: `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`
- Testnet equity: **999 mock USDC** (Unified Account — usable for perps directly)
- Agent (API) wallet `ems-bot` authorized, BTC leverage set 3x isolated
- Credentials in gitignored `.env` (HL_MASTER_ADDRESS, HL_AGENT_KEY, HL_TESTNET=true)
- Mainnet: 9.8 USDC sits in spot (the deposit that unlocked the faucet gate)

---

## Parked / Unfinished

**EMS live bot — remaining phases:**
- **Phase 4: unattended deploy.** Add entrypoint `python -m ems_live.run`
  (load `.env` → `run_forever`); add a Render background worker to `render.yaml`
  with HL_* env vars; soak-test on live testnet bars for days to catch a real
  autonomous trade. Decide: local soak first vs straight to Render.
- **Missed-bar catch-up on restart** — runner currently resumes at next bar; if the
  worker is down across a `:30` H1-exit bar, that exit is skipped. Add catch-up.
- **max-daily-loss kill switch** — designed, not yet implemented.
- **Phase 5: mainnet.** Flip testnet=False, transfer real USDC spot→perp (master-
  signed, in UI — agent can't), tiny `risk_usd`, mainnet dry-run first, then live.

**EMS engine (pre-existing):**
- Parity check vs TradingView Pine strategy report
- Short-side extension — Samuel may have short rules, not discussed

**MCT engine (carried):**
- No-divergence test result outstanding
- `max_sweep_age_bars` not implemented
- `div_found` counter cosmetic bug
- RSI divergence 4-model logic unaudited

---

## Next Steps (tomorrow — exact order)

1. **Build the entrypoint** — `ems_live/run.py`: load `.env`, construct LiveConfig +
   PositionStore + LiveBroker, call `run_forever()`. Add a tiny `.env` loader
   (no new dependency) or add `python-dotenv`.
2. **Local soak test (testnet, dry_run=True first)** — run `python -m ems_live.run`
   on your machine for ~1 hour, watch it tick each bar, confirm scheduler wakes on
   :00/:30, fetches, evaluates, logs "no signal". Then flip `dry_run=False` and let
   it run on testnet to (eventually) catch a real crossover and trade autonomously.
3. **Deploy to Render worker** — add background worker service to `render.yaml`,
   set HL_MASTER_ADDRESS / HL_AGENT_KEY / HL_TESTNET env vars in Render dashboard
   (NOT committed), deploy, watch logs. Soak on testnet for several days.
4. **Add missed-bar catch-up + max-daily-loss kill switch** before mainnet.
5. **Phase 5 mainnet** — only after a clean multi-day testnet soak: transfer real
   USDC spot→perp in UI, set tiny risk_usd, run one mainnet dry-run (orders logged),
   then arm live.
