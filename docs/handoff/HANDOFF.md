# Handoff

## Last Session Summary

**Final model LOCKED = V3 (H4 EMA20>EMA100). HL bot converted to V3 + entrypoint built.**

Model-selection + bot work:
1. **Caught a definition mismatch:** coded "V3" = H4 EMA20 > EMA100 *confluence* (two
   EMAs); user's mental model was *price > a single H4 EMA*. Clarified explicitly.
2. **3 alt models** built per user spec (base / price>H4 EMA50 / price>H4 EMA100), all
   with H1 **EMA50** exit (≠ the prior EMA100 exit). Full Binance CSVs w/ Madrid
   timestamps. Results: base PF 1.92 (1132 tr), r1 PF 2.01 (686), r2 PF 2.30 (649).
   Harness `scripts/three_models.py`, commit `97a4954`.
3. **Cross-compared** vs confluence V3 (EMA100 exit). Confluence **V3 EMA20>EMA100**
   strongest: EV 1.00, PF 2.63, and the only robustness-validated one. (Caveat noted:
   exit EMA differs 100 vs 50, so it's a confounded comparison.)
4. **User decision:** final = **V3 confluence, H4 EMA20>EMA100, H1 EMA100 exit.**
   Reasoning: wins every quality metric; R2's lower historical max-DD is a mirage —
   Monte-Carlo p95 DD is ~equal (−50 vs −52 R); the extra 171 trades are dilutive.
5. **Bot → V3:** decider gains `check_h4_confluence` (mirrors `engine.simulate` H4
   branch exactly), applied after H1 trend, before SL; `build_ctx`/`replay` take an
   optional h4 frame; config `h4_filter=True, h4_ema_slow=100, EMA-Cross-H4F`; runner
   builds H4 from Binance H1. **V3 parity test green** (live replay == engine V3,
   byte-identical). 49 tests. Commit `55ce69c`.
6. **Phase 4 entrypoint** `ems_live/run.py` — `python -m ems_live.run [once]`. Dry tick
   on testnet OK; H4 gate currently **blocks longs** (BTC H4 EMA20 < EMA100). Commit
   `4b28b70`.

Cost-testing status (unchanged): fees tested once (0.20% round-trip, EV survives);
**swap/funding still unmodeled** — Madrid-timestamped CSVs ready to feed it.

---

## Previous Session Summary — V3 trade CSVs with Madrid timestamps (swap-cost prep)

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

## Older Session Summary — live bot Phases 2+3 (testnet lifecycle)

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

Nothing mid-flight. Bot is **V3, parity-locked, entrypoint built**, left in
`dry_run=True` (safe). Strategy frozen: V3 confluence, H4 EMA20>EMA100, H1 EMA100
exit, long-only. Ready for soak (local then Render). Account funded + configured:
- Testnet/master address: `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`
- Testnet equity: **~999 mock USDC** (Unified Account — usable for perps directly)
- Agent (API) wallet `ems-bot` authorized, BTC leverage set 3x isolated
- Credentials in gitignored `.env` (HL_MASTER_ADDRESS, HL_AGENT_KEY, HL_TESTNET=true)
- Mainnet: 9.8 USDC in spot (the deposit that unlocked the faucet gate)
- Run it: `python -m ems_live.run once` (single tick) / `python -m ems_live.run` (forever)

---

## Parked / Unfinished

**EMS live bot — remaining phases (entrypoint DONE; bot is V3):**
- **Render worker** — add a background worker to `render.yaml`
  (`startCommand: python -m ems_live.run`); set HL_MASTER_ADDRESS / HL_AGENT_KEY /
  HL_TESTNET=true / EMS_DRY_RUN=true as Render env vars (NOT committed); deploy,
  soak on testnet several days.
- **Missed-bar catch-up on restart** — runner resumes at next bar; if down across a
  `:30` H1-exit bar, that exit is skipped. Add catch-up.
- **max-daily-loss kill switch** — designed, not yet implemented.
- **Phase 5: mainnet.** Flip testnet=False, transfer real USDC spot→perp (master-
  signed, in UI — agent can't), tiny `risk_usd`, mainnet dry-run first, then live.

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

## Next Steps (exact order)

1. **Local soak** — `python -m ems_live.run` on your machine, `dry_run=True`, a few
   hours. Confirm it wakes on :00/:30, fetches, evaluates, logs. (H4 gate currently
   blocks longs — BTC H4 EMA20 < EMA100 — so expect "no signal" until H4 flips up.)
2. **Render worker** — add bg worker to `render.yaml`
   (`startCommand: python -m ems_live.run`), set HL_* + EMS_DRY_RUN=true env vars in
   the Render dashboard (NOT committed), deploy, soak on testnet several days.
3. **Add missed-bar catch-up + max-daily-loss kill switch** before arming.
4. **Arm testnet** — set `EMS_DRY_RUN=false`, let it catch a real autonomous trade,
   verify entry→stop→exit on the testnet UI.
5. **Phase 5 mainnet** — only after a clean multi-day testnet soak: transfer real
   USDC spot→perp in UI, tiny risk_usd, one mainnet dry-run, then arm live.
