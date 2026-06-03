# Handoff

## Last Session Summary

**EMS live Hyperliquid bot: Phases 2 + 3 shipped — trades autonomously on TESTNET**

Built on top of last session's Phase 0+1 foundation. The bot can now run a full
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
