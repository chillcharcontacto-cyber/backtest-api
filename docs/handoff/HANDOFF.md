# Handoff

## Last Session Summary — FLAT card noise fix + strategy clarification

Small session on the live bot (already deployed + soaking).
- **FLAT card on gate-change only** (`36f5c36`) — the ⚪ FLAT status card was flooding
  Telegram every 30 min (48/day). Added `status_mode` (`EMS_STATUS_MODE`, default
  `change`): the card now sends ONLY when a gate flips (M30 bull, H1 vs ema50, H4
  allows), with a `Δ …` line naming what flipped. Last gate persisted to
  `<state>.status`; first card after boot fires once, then silent. Modes:
  `change` | `always` | `off`. 71 tests green.
- **Strategy mechanics clarified** (no code) — V3 entry = **1 trigger + 2 filters**:
  the M30 EMA20/50 bullish **cross is the trigger (event)**; H1 (close > ema50) and H4
  (ema20 > ema100) are **standing permission gates**. After H4 confirms (it lags,
  flips last), entry needs the next **fresh M30 cross** (a small M30 dip-and-recross) —
  NOT an H1 bear/bull cycle. Filters gate, the M30 cross fires.

---

## Previous Session Summary — Bot DEPLOYED & LIVE on Render + full monitoring 🟢

The EMS-V3 bot is now **running autonomously on Render** (testnet, `dry_run=True`,
no orders). Drove the Blueprint deploy via watch-and-guide (Claude sees via
computer-use desktop screenshots, user clicks — Render's never-idle SPA can't be
auto-driven). Then built the full monitoring/alerting stack and fixed a live bug.

Shipped this session:
1. **Deployed** the `ems-live-bot` worker from `render.yaml` (Blueprint). Live, running
   `python -m ems_live.run`, persistent disk `/data`. ~$7.25/mo.
2. **PYTHONUNBUFFERED=1** (`b13f88e`) — stream logs live (were buffered).
3. **Monitoring** (`7aae9b6`) — `ems_live/notify.py`: Telegram cards (🚀 start, ⚪ FLAT
   status each tick w/ M30·H1·H4 EMA standings, 🟢 ENTRY, 🔵 in-pos each H1 close,
   🔴/🔻 EXIT, ⚠️/🛑 BLOCKED) + healthchecks.io liveness ping. Env-gated, never raises.
4. **Minute heartbeat** (`a933908`) — ping every 60s → ~3-min hang detection
   (healthchecks Period 1m/Grace 2m). Render native alerts = instant on crash.
5. **EXIT deviation** (`065352c`) — card shows model R → net R after fees + deviation %
   (cost drag; `$ = R × risk_usd` with fixed-$ risk).
6. **Count-based kill switch** (`0e61319`) — halt after **10 losing trades/day** (R-based
   cap off; 3R was too tight for this low-WR/streaky strategy). NOTE: backtest had NO
   kill switch — live-only overlay.
7. **Binance 451 fix** (`4cdbe8c`) — live feed → `data-api.binance.vision` (Render's US
   IP was geo-blocked by api.binance.com). **Caught by the new Telegram alerting** —
   monitoring proved itself on day one.

Verified live: 🚀 + ⚪ FLAT cards arriving, feed flowing; all 3 gates currently closed
(BTC soft) so the bot correctly waits. 67 tests green.

Monitoring config (in Render env): Telegram bot token + chat `442557401`,
healthcheck `https://hc-ping.com/27d79596-d668-44e4-b8c8-991f6912ee9c`.

---

## Earlier History (condensed — full detail in git + decisions-log)

- **Render auto-drive BLOCKED:** Render dashboard is a never-idle SPA; all
  Claude-in-Chrome DOM tools time out. Use watch-and-guide (computer-use desktop
  screenshots + user clicks). `navigate` by URL works → jump to `/select-repo?type=blueprint`.
- **Autonomy hardening** (`5a29d53`): per-tick exchange reconcile (detects resting-stop
  fills live), kill switch, Render worker blueprint + disk, entrypoint env.
- **V3 locked + bot→V3** (`97a4954`,`55ce69c`,`4b28b70`): final model V3 confluence
  H4 EMA20>EMA100, H1 EMA100 exit; decider mirrors engine, V3 parity byte-identical.
  Three alt price-filter models built + rejected (`scripts/three_models.py`).
- **Robustness** (`docs/EMS_V3_H4_robustness_recap.md`): lock period 100.
- **Madrid CSVs** (`5179be5`): swap-cost prep (funding still unmodeled — open item).
- **Bot Phases 2+3** (`7759439`,`8e82fb8`): state machine, broker (market entry /
  resting stop / close), guards, stop-confirmed-or-flatten, 5-sig-fig px rounding,
  agent trade-only. Full testnet lifecycle proven.

---

## Currently Working On

**Soaking — bot is LIVE & autonomous on Render (testnet, dry_run, no orders).** Nothing
mid-flight. It posts ⚪ FLAT status each :30 and will fire 🟢 ENTRY when all 3 gates
align (M30 cross + H1 close > ema50 + H4 ema20 > ema100). Just let it run + watch Telegram.
- Render worker `ems-live-bot`, kill switch = **10 losses/day**, risk $20
- Testnet/master address: `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`, equity ~999 mock USDC
- Agent `ems-bot` authorized, BTC 3x isolated; creds in gitignored `.env`
- Mainnet: 9.8 USDC in spot (faucet-gate deposit)
- Local run: `python -m ems_live.run once` / `python -m ems_live.run`

---

## Parked / Unfinished

**EMS live bot — DEPLOYED & LIVE on testnet (dry_run); remaining = soak + mainnet:**
- **Missed-bar catch-up on restart** — runner resumes at next bar; if down across a
  `:30` H1-exit bar, that exit is skipped. Low priority — stop rests on exchange so
  positions stay protected; only the computed H1 exit can be missed.
- **Phase 5: mainnet.** Flip HL_TESTNET=false, transfer real USDC spot→perp (master-
  signed, in UI — agent can't), tiny `risk_usd`, mainnet dry-run first, then arm.
- **Slippage in deviation** — EXIT deviation is fee-only; fold in real fill slippage
  from exchange data for a true model-vs-actual.

DONE (no longer parked): Render deploy (LIVE), monitoring (Telegram+healthchecks),
count-based kill switch, per-tick reconcile, entrypoint, unbuffered logs, 451 fix.

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

## Next Steps — bot is LIVE on testnet; now SOAK then mainnet

1. **Soak (now, passive).** Let it run dry on testnet. Watch Telegram for ⚪ FLAT
   cards each :30, and the first 🟢 ENTRY → 🔵 in-pos → 🔴/🔻 EXIT cycle when the 3
   gates align. Cross-check that cycle against the Hyperliquid testnet UI (Positions /
   Trade History / Funding History). Goal: ≥1 real autonomous trade end-to-end.
2. **ARM testnet** — when you trust it, worker → Environment → `EMS_DRY_RUN=false` →
   Save. Real testnet orders fire on the next aligned signal.
3. **Mainnet prep** — fix capital first: 9.8 USDC can't support risk $20; deposit more
   or drop `EMS_RISK_USD` to ~$1–2. Funding/swap cost is real only on mainnet — watch
   Funding History (or build the historical-funding filter — parked).
4. **Phase 5 mainnet** — flip HL_TESTNET=false, transfer real USDC spot→perp in the UI,
   one mainnet dry-run, then arm.

Verify after any redeploy: 🚀 card reads `kill 10 losses/day`; ⚪ cards have real data
(no TICK ERROR).

**8. Verify first autonomous trade** — check entry + resting stop on the testnet UI.

**Later — Phase 5 mainnet:** only after a clean multi-day armed testnet soak. Flip
HL_TESTNET=false, transfer real USDC spot→perp in the UI (master-signed; agent can't),
tiny risk_usd, one mainnet dry-run, then arm live.

Heads-up: Render worker + persistent disk is **paid** (~$7/mo). Optional: run
`python -m ems_live.run` locally first to watch it live before paying.
