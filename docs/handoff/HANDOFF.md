# Handoff

## Last Session Summary — first testnet trade verified; WENT LIVE ON MAINNET 🎯

No code changes — verification + mainnet go-live.

**Verified the first real testnet trade (2026-07-12) end-to-end — execution is correct.**
Reconciled the Telegram cards against on-chain fills:
- Entry: on-time market long 0.06689 BTC @ 64,103, notional $4,288, **auto-leverage 5×**
  (the old $500 ceiling would have refused this) — the crossover bar closed 17:00 and it
  entered 17:00 (timing fix working; card labels it 16:30 = the bar's open time).
- Stop: resting structural stop placed at 63,787, **triggered**, closed the position.
- Result: **−1.19R** (−$24). The extra ~0.19R beyond −1R is **testnet thin-book slippage**
  (stop filled 63,745.9 vs 63,787 trigger; entry swept 8 fills). Mainnet's deep book fills
  far tighter — not a defect.
- 🔵 IN-POS cards fire once per H1 close by design (5-6 for a 6h trade) — explained.

**Found a P&L-reporting gap (staged):** SL exits are recorded/reported at the trigger
price (clean −1.00R), NOT the actual fill, so the card under-reports the loss by the
slippage (~$2.75 on that trade). Fix = read the real SL fill for the exit record.

**WENT LIVE ON MAINNET (real money):**
- Funded ~$328 USDC (in spot; unified account backs perp — verified read-only).
- Authorized a MAINNET agent `ems-bot-main` (`0xc07aA2354249ba34D7a4436fEDEC6864Dd07b8Fd`,
  valid ~180 days). Key is in Render env only (trade-only, cannot withdraw).
- Render env: `HL_TESTNET=false`, `EMS_RISK_USD=3`, `EMS_STATE_PATH=/data/ems_mainnet_state.json`,
  `EMS_DRY_RUN=false`. 🚀 card confirms `testnet=False dry_run=False risk $3.0`.
- Armed and waiting at **Stage 1** (H4 bull, H1 below ema50) — first real mainnet trade
  fires when H1 reclaims ema50 + a fresh M30 cross. $3 risk, 10-loss/day kill = −$30 max/day.

**UPDATE — first mainnet trade fired and WON (verified on-chain).** Trade #1 (07-14 13:00
→ 07-16 10:00, ~45h, H1_EMA100 exit): open 63,799 → close 64,079, 0.00294 BTC. Price P&L
+$0.82, fees −$0.16, **funding −$0.11** (45 hourly payments over the hold) → **real net +$0.55**.
The exit card showed +$0.69 / +0.23R net (**fee-only — funding not included**) and a −19.8%
deviation, which is just **fees as a % of a small +0.28R win** (fee ≈0.054R fixed by the
~1.6% stop; against a small R it's a big %, against +2R it'd be ~3%). Deep-book fills were
tight (no testnet slippage). A **2nd trade is currently OPEN** (07-17 20:00, 0.01626 BTC @
64,054). User declined adding funding to the card for now — left as a refinement.

---

## Previous Session Summary — verified the fixes in dry-run, ARMED testnet

No code changes — verification + arming only.
- **Reconciled "several trades this week" against on-chain reality.** Queried HL fills
  for the master `0x18ce…6964`: TESTNET has only the June-3 manual Phase-3 fills (14, all
  long) — NO autonomous trades; MAINNET fills are the user's OWN 2025 manual trades (10
  shorts / 2 longs, agent NOT authorized there) — not the bot. Conclusion: the bot placed
  no real orders; the "trades" were **🟡 DRY_RUN cards** (bot was still `dry_run=True`).
- **The dry-run cards VALIDATE both fixes** (post-`5054608`/`eb9f8f6`): e.g. 2026-07-07
  entry 63,916/sl 62,980 → $1,366 notional, **2× lev, risk $20** (old $500 ceiling would
  have refused this); 2026-07-09 entry 61,563/sl 61,371 → 0.31% tight stop, ~$6.4k
  notional (the exact case the ceiling killed) now sized fine. Entries at clean bar opens
  (on-time, not a bar late).
- **Confirmed Unified Account: spot USDC backs perp orders** — placed a live 0.0002 BTC
  testnet order that FILLED with funds in spot, then closed. So NO spot→perp transfer
  needed; the $998 spot is the trading margin (perp balance reads 0 cosmetically).
- **ARMED testnet:** user set `EMS_DRY_RUN=false`; 🚀 card confirms `dry_run=False`,
  risk $20, kill 10 losses/day. Gates currently H4✅ H1✅, M30 armed (stage 2) — next
  fresh M30 cross fires the FIRST real testnet trade.

---

## Earlier Session Summary — two live-execution bugs fixed (sizing + entry timing)

Two real bugs on the live-money path, both found, fixed, and adversarially reviewed.

**Bug 1 — sizing refused valid trades (`5054608`).** The first live signal was refused
("notional 3438 exceeds ceiling 500"). A 63-scenario audit (5-agent workflow) found the
root cause: a fixed $500 notional ceiling is mathematically incompatible with fixed-$
risk (notional = risk_usd/stop%), so it refused ~every valid trade. Replaced with
**equity-relative auto-leverage** (`plan_trade`): size stays `risk_usd/(entry-sl)` (risk
fixed); leverage = smallest that fits margin, **capped so liquidation stays beyond the
stop**; resizes down (flagged) only if the account can't carry it — never over-leverages,
never silently drops. Guards + broker made equity-relative + robust; ALL knobs env-tunable
(`EMS_MAX_LEVERAGE`/`MARGIN_BUFFER_FRAC`/`LIQ_SAFETY_MULT`/`MAX_RISK_BAND_PCT`/`MIN_RISK_PCT`/
`SLIPPAGE`/`HL_MIN_NOTIONAL`/`MAX_NOTIONAL_USD`). Adversarial review (15-agent, 12→6
confirmed) caught + fixed 4 follow-ons: plan flags infeasible on ultra-wide stops; LIVE
aborts if equity can't be read (no blind leverage); guarded flatten + state-saved-before-
stop (never orphan an unprotected long); partial-fill P&L uses actual filled risk.
Validated on real testnet numbers: the exact $3,448 trade → 4× lev, liq ~24% away, risk $20.

**Bug 2 — entry one bar late (`eb9f8f6`).** User spotted it on the chart: the bot fired
at the entry bar's CLOSE (30 min late) at a worse price. Backtest enters at open[i] on a
cross at i-1; the live tick checked cross[i-1] on the latest closed bar, waiting an extra
bar. Fix: `check_entry_live()` detects the crossover on the JUST-CLOSED bar and enters at
the live mid immediately (~open[i+1] ~ close[i] = the backtest price/instant). Parity test
proves `check_entry_live(i)` selects the identical trades as backtest `check_entry(i+1)`
(same SL/crossover/anchor) — only timing/price corrected, WHICH trades unchanged.

87 tests green. Bot still LIVE on testnet, `dry_run=True` (no orders); Render auto-redeploys.

---

## Older Session Summary — step-by-step ladder notifications + daily heartbeat

Telegram-notification UX work on the live bot (already deployed + soaking).
- **Step-by-step ladder** (`fdfe51b`, new default `EMS_STATUS_MODE=steps`) — top-down
  H4→H1→M30. A step message fires only when the setup STAGE changes, so lower-TF noise
  stays silent while a higher TF blocks:
    - stage 0 = H4 bearish (blocked) → silent on H1/M30
    - stage 1 = H4 bullish, waiting H1 → 🟠 "Step 1/3"
    - stage 2 = H4+H1 bullish, armed, waiting fresh M30 cross → 🟡 "Step 2/3"
    - M30 cross → 🟢 ENTRY. Regressions ping too (H4→bearish = 🔴 standing down).
  Plus a once-per-day 😴 "still alive" heartbeat showing H4/H1/M30 even during
  multi-day dead-flat stretches. Stage + last-heartbeat-day persist to `<state>.status`.
- Superseded the prior same-thread `change` mode (`36f5c36`, FLAT card only on any gate
  flip) — `steps` is the new default; `change`/`always`/`off` still available.
- **Strategy mechanics clarified** (no code) — V3 entry = **1 trigger + 2 filters**:
  M30 EMA20/50 bullish **cross = trigger (event)**; H1 (close>ema50) and H4
  (ema20>ema100) = **standing gates**. After H4 confirms (lags, flips last), entry needs
  the next **fresh M30 cross** (small M30 dip-and-recross), NOT an H1 bear/bull cycle.
- 75 tests green. Bot still LIVE on testnet, dry_run, soaking.

---

## Oldest Session Summary — Bot DEPLOYED & LIVE on Render + full monitoring 🟢

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

**LIVE ON MAINNET (real money) — waiting for the first real mainnet trade.** Render worker
`ems-live-bot`: `HL_TESTNET=false`, `EMS_DRY_RUN=false`, **risk $3/trade**, kill 10 losses/day,
`EMS_STATE_PATH=/data/ems_mainnet_state.json`. 🚀 card confirms `testnet=False dry_run=False
risk $3.0`.
- Master `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`; **~$328 USDC in SPOT** = the margin
  (Unified Account backs perp off spot).
- Mainnet agent **`ems-bot-main`** `0xc07aA2354249ba34D7a4436fEDEC6864Dd07b8Fd` authorized
  (~180 days; re-authorize before it lapses). Key in Render env only, trade-only (no withdraw).
- Auto-leverage: at $3 risk on $328, a 1% stop → 2×, tight stops → higher lev, all liq-safe.
- Testnet still exists (`ems-bot` agent, ~$998 spot) but the worker now points at mainnet.
- Currently **Stage 1** (H4 bull, H1 below ema50) — no trade until H1 reclaims ema50 + M30 cross.

**Mainnet execution PROVEN:** trade #1 closed a verified win (real net +$0.55 after fees +
funding); a 2nd trade is currently open. Deep-book fills tight. Just let it run + watch
Telegram; to check any trade, pull on-chain fills + `userFunding` for `0x18ce…6964` on MAINNET.

Reconciliation gotchas learned: the exit card is **fee-only** (excludes funding — matters on
long holds, e.g. −$0.11 over 45h) and books SL exits at the trigger, not the fill. Deviation%
is inflated on small-R trades (fee is a big % of a small move) — read the $ not the %.

---

## Parked / Unfinished

**EMS live bot — LIVE ON MAINNET; open refinements (none blocking):**
- **Exit card is fee-only — add funding + actual SL fill.** Two accuracy gaps now that it's
  live: (a) SL exits book the TRIGGER price, not the actual fill (under-reports a slipped
  stop); (b) net P&L/deviation EXCLUDE perp funding, which is real on long holds (−$0.11 over
  45h on trade #1). Fix: read the real SL fill for the exit record AND pull `userFunding` for
  the hold window so the card/ledger show true net. Highest-value refinement now real money.
- **IN-POS card throttle (optional)** — fires once per H1 close; user may want it only on a
  big R move or near the exit trigger (a long hold = many cards).
- **Staged edge-robustness (63-scenario audit, not blocking common trades):** in-bar retries
  for a transient Binance/HL feed error; SL-adapt fallback to the basis-adjusted Binance stop
  when HL candles are empty; tz hardening in sl_adapter; feasibility/no-retry on a single-bar
  refusal. Full list: audit output `tasks/wbwomb00s.output`.
- **Missed-bar catch-up on restart** — if down across a `:30` H1-exit bar, that exit is
  skipped (stop still rests on exchange, so protected).

DONE (no longer parked): mainnet go-live, Render deploy, monitoring, count kill switch,
per-tick reconcile, entrypoint, unbuffered logs, 451 fix, equity-relative auto-leverage,
entry-timing fix, Unified-Account spot-as-perp-margin (verified — no transfer needed).

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

⭐ **USER REMINDER (next session):** start tracking the EMS indicator on **GU (GBP/USD)
from Jan 5th** onward, and keep tracking the indicator. (User's words: "get it from
jan 5th on GU and keep tracking the indicator" — clarify exact intent: forward-track
the live indicator on GBPUSD, and/or backtest GBPUSD from 2026-01-05.)

1. **Soak (now, passive).** Let it run dry on testnet. Watch Telegram for the
   step-by-step ladder (🟠/🟡 step cards on gate flips, daily 😴 heartbeat), and the
   first 🟢 ENTRY → 🔵 in-pos → 🔴/🔻 EXIT cycle when the 3 gates align. Cross-check
   that cycle against the Hyperliquid testnet UI (Positions / Trade History / Funding
   History). Goal: ≥1 real autonomous trade end-to-end.
2. **ARM testnet** — when you trust it, worker → Environment → `EMS_DRY_RUN=false` →
   Save. Real testnet orders fire on the next aligned signal.
3. **Mainnet prep** — fix capital first: 9.8 USDC can't support risk $20; deposit more
   or drop `EMS_RISK_USD` to ~$1–2. Funding/swap cost is real only on mainnet — watch
   Funding History (or build the historical-funding filter — parked).
4. **Phase 5 mainnet** — flip HL_TESTNET=false, transfer real USDC spot→perp in the UI,
   one mainnet dry-run, then arm.

Verify after any redeploy: 🚀 card reads `kill 10 losses/day`; step/heartbeat cards
have real data (no TICK ERROR).
