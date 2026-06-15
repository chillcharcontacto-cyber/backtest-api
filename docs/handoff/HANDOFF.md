# Handoff

## Last Session Summary — Render deploy attempt (BLOCKED, resume at step 4)

No code shipped. Attempted to drive the Render Blueprint deploy (step 4) via the
Chrome extension. **Got as far as the repo-select page** (`dashboard.render.com/
select-repo?type=blueprint`) with `chillcharcontacto-cyber/backtest-api` listed and a
**Connect** button visible — but could not complete it.

**Blocker (important for next time):** Render's dashboard is a live SPA that never
reaches `document_idle`, so every Claude-in-Chrome DOM tool (`find`, `read_page`,
`screenshot`, `form_input`) times out at 45s. Blind pixel-clicks via the extension
landed nowhere reliable (coordinate scale couldn't be calibrated without an
in-Chrome screenshot). Desktop screenshots (computer-use) DO work for *seeing* the
page, but Chrome is read-tier so computer-use can't click.

**Working approach for next session:** hybrid — Claude watches via desktop
screenshots and calls each click; the USER clicks (they're at the keyboard). Or the
user just runs the 5-click Blueprint flow solo with the values below. Do NOT try to
fully auto-drive Render again — it doesn't work.

---

## Previous Session Summary — autonomy hardening (V3 bot safe to run unattended)

All CODE for full-autonomous testnet operation is done; `dry_run=True` default.
Commit `5a29d53`, 58 tests green.
1. **Per-tick exchange reconcile** (`runner.tick`) — detects the resting stop firing
   while running (queries exchange each tick; if flat → record −1R, go flat). Closed a
   real gap (stop fills were only seen on restart).
2. **Max-daily-loss kill switch** (`ems_live/daylimit.py`) — `DayLedger` tracks realized
   R per UTC day, halts entries at ≤ −`max_daily_loss_r` (default 3R; 0 disables).
3. **Render worker blueprint** (`render.yaml`) — `type: worker`,
   `python -m ems_live.run`, persistent disk `/data`; HL_* secrets `sync:false`.
4. **Entrypoint env** — `run.py` reads EMS_STATE_PATH + EMS_MAX_DAILY_LOSS_R.

---

## Earlier History (condensed — full detail in git + decisions-log)

- **V3 locked + bot→V3 + entrypoint** (`97a4954`,`55ce69c`,`4b28b70`): final model =
  V3 confluence H4 EMA20>EMA100, H1 EMA100 exit. Decider `check_h4_confluence` mirrors
  `engine.simulate`; V3 parity byte-identical; `ems_live/run.py` (`once`/forever). Three
  alt price-filter models built + rejected (`scripts/three_models.py`).
- **Robustness check** (`docs/EMS_V3_H4_robustness_recap.md`): V3 H4 slow 50 vs 100 →
  ROBUST (EV gap 4.2%, overlap 78.7%), locked 100.
- **Madrid-timestamped V3 CSVs** (`5179be5`) for swap-cost analysis (funding still
  unmodeled — open item).
- **Bot Phases 2+3** (`7759439`,`8e82fb8`): state machine + reconcile, scheduler,
  broker (market entry / resting stop / close), safety guards, stop-confirmed-or-flatten,
  HL price rounding (5 sig figs), agent is trade-only (can't withdraw). Full testnet
  lifecycle proven with real fake-money orders.

---

## Currently Working On

**Render deploy — mid-flight, resume at step 4.** Code 100% done; only the Render
dashboard deploy remains. Last session reached the repo-select page but couldn't
auto-drive Render's SPA (see Last Session Summary). Next session: use the
watch-and-guide hybrid (or user runs the 5-click flow solo) with the values below.
Bot stays `dry_run=True`. Account funded + configured:
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
   - **How to drive it:** Render's dashboard is a never-idle SPA — Claude-in-Chrome
     DOM tools all time out, so do NOT try to auto-drive. Use the hybrid: Claude
     watches via computer-use desktop screenshots and calls each click; USER clicks.
     Or user just does the ~5 clicks solo. Fast path to the repo step:
     navigate directly to `dashboard.render.com/select-repo?type=blueprint`.
   - Steps: that page → **Connect** on `chillcharcontacto-cyber/backtest-api` → Render
     reads `render.yaml`, shows `backtest-api` + new **`ems-live-bot`** worker.
   - Paste the two secrets it prompts for (`sync: false`):
     **HL_MASTER_ADDRESS** = `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`
     **HL_AGENT_KEY** = the testnet agent key (in local `.env`).
   - Presets already set: HL_TESTNET=true, EMS_DRY_RUN=true, risk 20, kill-switch 3R,
     state on `/data`. Apply → Deploy. (Worker + disk = paid ~$7/mo.)

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
