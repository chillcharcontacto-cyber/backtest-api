# Handoff

## Last Session Summary — Tier-4 ACTIVATED: brain live + thin-client published + signing key rotated clean

**Tier-4 is now fully live — no repo code changes; all activation (Render + GitHub + verification).**

- **Brain deployed + serving:** `https://ems-brain.onrender.com` (the standalone `ems-brain` Free
  web service). Signs license-gated EMS decisions; verified end-to-end.
- **Signing key rotated to a CLEAN one.** The first TWO generated keys were exposed in Render
  screenshots, so it was rotated twice; the LIVE signer is now
  **`0xd0B67b43ce1459381871aF5b64FBB47CC4404513`** (that key's private half was never
  screenshotted). `EMS_BRAIN_SIGNING_KEY` on the `ems-brain` service holds it. Lesson: after
  generating a key, copy the private half straight into Render — never screenshot the terminal.
- **Thin-client repo PUBLISHED public:** `github.com/chillcharcontacto-cyber/ems-thin-client`
  (branch `main`), brain URL + clean signer baked into `render.yaml`/`.env.example`, Deploy-to-
  Render button targeting the repo. Committed with the GitHub **noreply** email
  (`273226247+chillcharcontacto-cyber@users.noreply.github.com`) to pass email-privacy (GH007
  blocks a real email in a public push).
- **Verified end-to-end with the SHIPPED client code:** `thin.brain_client.BrainClient` against
  the live brain returns a signed decision that verifies against the clean signer (account +
  nonce + expiry checked); a wrong-signer client is rejected. Test key `EMS-test-01` bound to the
  master (`0x18ce…964`) for verification — harmless; remove/keep as desired.

Tier-4 is ready for real partners; onboarding is now a per-partner operator recipe (Next Steps /
`docs/BRAIN_OPERATIONS.md`). The "rotate before first partner" reminder is DONE (live key is clean).

---

## Previous Session Summary — 429 rate-limit fix: hardened retry + missed-entry catch-up + throttled alerts

**Shipped (`45e4a1e`, deployed to the worker). The live bot's OPEN trade is untouched — the
in-position management code didn't change; still riding (+$310 as of 08-27).**

**Diagnosed the 429 TICK-ERROR pattern** (user reported 8 in a day): it is NOT the bot and NOT
HL being down. HL is fine from other IPs (12/12 OK from a home IP), and the worker makes only
~1 HL call per 30-min tick — so it isn't over-calling. It's **Render's shared outbound IP
being rate-limited by Hyperliquid** — outside the bot's control, and harmless to an open
position (the stop rests on the exchange). The one real risk was a 429 landing on an ENTRY bar
→ a missed trade.

**Fix — 3 parts, 103 tests green (`tests/test_live_catchup.py` new):**
1. **Shared hardened retry** — new `ems_live/nethttp.py` used by BOTH `broker.py` and `feed.py`:
   6 tries, exp backoff + jitter (~45s), so a rate-limit is ridden out at EVERY HL touchpoint
   (candles, SL adapt, orders). Reads retry 429+5xx; orders retry **429 only** (rejected
   pre-matching → no double-fill; an ambiguous 5xx is never resent).
2. **Missed-entry catch-up** — the entry path is extracted to `runner._attempt_entry()`; a 429
   during entry now **DEFERS** (saves a `<state>.pending` sidecar) and the next bar retries while
   the signal is still fresh (≤ `CATCHUP_MAX_BARS`=3 bars) and valid (guard re-checks price vs
   stop). An ambiguous NON-429 order error aborts with NO pending → a trade is never double-
   entered. Near-zero missed entries from a rate-limit (not literally zero — a block spanning the
   entry bar AND the catch-up window would still miss; unavoidable for any bot).
3. **Throttled alerts** — a 429/5xx tick error sends **ONE Telegram card per day** + a running
   count (`<state>.429`); any NON-429 error stays loud. Kills the spam, keeps the signal.

Rule of thumb kept: a one-off 429 = noise (harmless, self-heals); a NON-429 TICK ERROR = real,
flag it. `is_rate_limit()` in `nethttp.py` is what routes them.

---

## Session — partner distribution built: kit → Tier-4 protected bot (brain + thin client) + 429 fix

**Update (2026-08-26) — TIER-4 ACTIVATED: brain deployed + verified live.** Created the brain's
web service on Render — it never existed (only the worker did), and the blueprint had reserved
the name `backtest-api`, so it's a **standalone Free web service named `ems-brain`** →
**`https://ems-brain.onrender.com`** (from `main`, `uvicorn api:app`, Oregon, health `/health`,
auto-deploy on commit). Set env `EMS_BRAIN_SIGNING_KEY` + `EMS_LICENSE_KEYS =
{"EMS-test-01":"0x18ce…964"}` (a TEST key bound to the master, for verification only). **Verified
end-to-end:** `POST /ems/decision` → HTTP 200 → strategy ran on live candles (returned `none` =
no signal this bar), and the signature recovers to operator signer
**`0x35657b4d9790ff143c050556B2b89aC38eDfe3fb`**, account-bound + nonce'd + unexpired. Whole chain
works (license → candles → strategy → signed decision → verifiable). ⚠️ **The private signing key
was shown in a Render screenshot → it is in this chat → ROTATE it (new keypair → update
`EMS_BRAIN_SIGNING_KEY` → send the new signer address) BEFORE issuing any real partner key.**
Remaining (see Next Steps): rotate key → prep+publish the thin-client repo → smoke-test a thin
client dry-run → issue real partner keys (replacing EMS-test-01).

**Update (2026-08-23) — no code changes.** A 2nd 429 TICK ERROR arrived (Sun 08-23 07:30).
Full on-chain health check: bot fine — BTC long still open **+$248** (0.02151 @ 64,477, mark
~76k), the ONE reduce-only stop resting @ 64,337, no dupes/orphans, **the 429 caused zero
trades**, liq 52,360 (far). Confirmed the **429 retry fix IS deployed** (user got the 🚀 card
on 08-21) — so this was just a rate-limit burst that outlasted the ~3.5s retry window (rare).
**Decision: leave 429 alerting AS-IS** — the TICK ERROR card is a catch-all that also catches
real errors, so it's not silenced. A single 429 is noise; a *pattern* (many/day) is signal;
the user watches for the pattern manually and pings if it recurs. (A "smart" alerting change —
silence single 429s, alert on a daily threshold, keep non-429 errors loud — was designed and
**deferred**; see Parked.)

**Big build session. Live bot untouched operationally; all new work is partner-facing +
one resilience fix. Everything shipped to `main` + two sibling deliverable folders.**

**429 retry/backoff fix (`f39a432`, deployed).** A live tick threw a TICK ERROR from a
one-off HTTP 429 (Hyperliquid's CloudFront edge rate-limiting a poll). Position was never
at risk (resting stop on the exchange). Wrapped every HL SDK call in `_retry()` with
exponential backoff — reads retry on 429+5xx (idempotent), writes on **429 only** (rejected
before matching → no double-fill risk). A transient blip now self-heals in-tick instead of
skipping a bar. `ems_live/broker.py`; unit-tested; 87→ still green.

**Partner replication — the goal: give trading partners the same autonomous EMS bot.**
Worked through it end to end:
- **Correct topology = each partner runs their OWN Render + OWN Hyperliquid.** Hosting for
  them scales cost per partner AND makes you hold their keys — rejected. Self-serve = $0 to
  the user, full isolation, no custody.
- **Built the Tier-1 kit** `ems-live-bot/` (sibling folder): the monolith bot trimmed to a
  clean, publishable template (no strategy files removed — it HAS the strategy) + Deploy-to-
  Render + SETUP.md for non-coders. Verified zero personal data. **Superseded by Tier-4
  below** for redistribution control; keep only as the simple unprotected option.
- **User wanted redistribution control** ("they can't hand files to anyone without my
  approval"). Analyzed 5 tiers (source check → obfuscate → encrypted-with-remote-key →
  **server-side brain**). Honest limit: you can't make working code un-copyable, but you CAN
  gate USE. Chose **Tier-4: keep the strategy server-side; ship only the hands.**

**Tier-4 BUILT (3 stages, all tested):**
- **Stage 1 — brain (`e78d547`, on `main`):** `POST /ems/decision` added to the existing
  `api.py` web service (so hosting is ~$0 — it rides the already-deployed `backtest-api`
  service). Given a partner's license key + state, it runs EMS V3 on fresh candles (reuses
  the real `decider` → no drift) and returns a **signed, account-bound, 120s-expiring**
  instruction. License = approved key BOUND to one HL address (env `EMS_LICENSE_KEYS`);
  signing via `eth_account` (no new dep, env `EMS_BRAIN_SIGNING_KEY`). `ems_brain.py` +
  `scripts/gen_brain_keys.py`. **8 tests.** DORMANT until both env vars are set → safe live.
- **Stage 2 — thin client (`ems-thin-client/` sibling folder):** a full partner bot with
  **ZERO strategy** — calls the brain, verifies 4 ways (signature=operator, account=mine,
  nonce, not-expired), executes on the partner's HL account. Keeps all safety (save-before-
  stop, guarded flatten, kill switch, equity-abort, the 429-retry broker). **No pandas/numpy,
  no strategy names** (exit labels genericized to STOP/TREND_EXIT). **15 integration tests.**
  Not committed to backtest-api (it's a separate PUBLIC repo the user publishes).
- **Stage 3 — operator playbook (`6cacb76`, on `main`):** `docs/BRAIN_OPERATIONS.md` —
  issue/revoke keys, activate, smoke-test. Private (not in the public thin-client repo).

**110 tests total** (95 backtest-api + 15 thin client). **Activation deferred to next
kickoff** (see Next Steps — only the user can do it: generate keys, set env vars, publish).

⚠️ **Git hazard this session:** the `backtest-api` main clone dir is checked out on the
**`multitimeframe`** branch (a side-chat's Pine work) and had that session's **uncommitted
doc WIP**. All EMS/brain work was done on `main` via **throwaway `git worktree`s** to avoid
touching it. Next session: do EMS work from a main worktree (or `git switch main` only if the
dir is clean), never commit EMS changes onto `multitimeframe`.

---

## Session — risk raised $3 → $6/trade; account funded to ~$1,279; open trade verified

**No code changes — a live config change + verification (~2026-08-20, a month after the
ops_kit session below).**

**Raised EMS risk per trade $3 → $6.** Done on Render (worker `ems-live-bot` →
Environment → `EMS_RISK_USD` 3→6 → **Save and deploy** — env-var change needs a restart,
no rebuild). Confirmation is the 🚀 card reading `risk $6.0` after redeploy. `EMS_RISK_USD`
is read only at ENTRY sizing, so the change applies to the NEXT trade only.

**Account was funded up** from ~$325 to **~$1,279** (perp $455 + spot $824), via
Phantom → KuCoin → Arbitrum → Hyperliquid (walked the user through that route this
session). At $6 risk on ~$1.3k, auto-leverage sizing is very comfortable; daily kill-switch
max loss is now 10 × $6 = **−$60/day** (was −$30).

**Verified the open trade on-chain (read-only).** BTC long **0.02151 @ 64,477**, resting
**Stop Market at 64,337** (reduce-only) = confirmed **$3.01 risk** (pre-change sizing).
With BTC at ~72,800 it was **+~$179 unrealized** (deep runner) — will exit on **H1 close <
H1 EMA100** (price is $8k above the structural stop, so the EMA100 cross is the realistic
exit). This trade is UNAFFECTED by the $6 change: its size/SL/stop are locked on the
exchange + `/data`; a redeploy `OK_RESUME`s it; exit P&L is computed off stored size/entry/
SL, not `risk_usd`. Redeploy is safe (stop rests on the exchange through the restart).

Since the ops_kit session (~07-21): the bot kept trading unattended — 08-07→08-10 long
(−$1.47), then the 08-19 long above. Running clean.

⚠️ **`render.yaml` intentionally holds SAFE DEFAULTS (testnet=true, dry_run=true,
risk=20), NOT the live values.** The live mainnet config (testnet=false, dry_run=false,
risk=6, state=/data/ems_mainnet_state.json) lives ONLY in the Render dashboard. Do **not**
"fix" render.yaml to the live values — a fresh blueprint sync is meant to come up safe. If
you ever re-sync the blueprint, re-apply the mainnet dashboard overrides afterward.

---

## Session — mainnet reconciled + H1-intrabar variation REJECTED + ops_kit shipped for the swing-bot

**Live account reconciled (read-only). Bot is FLAT; 2 mainnet trades done, net −$3.10.**

| # | in → out | hold | price P&L | fees | real net |
|---|---|---|---|---|---|
| 1 | 63,799 → 64,079 | 45h | +$0.82 (+0.28R) | −$0.16 | **+$0.55** |
| 2 | 64,054 → 63,887 | 3h | −$2.72 (−0.90R) | −$0.90 | **−$3.61** |

Equity **$328.06 → $324.96** (−1%). Funding over the window −$0.14 (48 events).
Trade #2 exited 23:11 (mid-bar, *above* its stop) → the resting stop triggered on a wick and
**filled better than −1R** (−0.90R). Execution is healthy; no defect.

⚠️ **Live data exposed a real cost: fee drag scales inversely with stop width.** Because
`notional = risk / stop%`, a tight stop means a huge notional and therefore huge fees
*relative to risk*. Trade #2 (**0.29% stop**) paid **$0.90 = 0.30R in fees**; trade #1
(1.6% stop) paid only 0.05R. Formula: `fee_R = 0.09 / stop%` at HL's 0.09% round-trip taker.
Median stop across the backtest is 0.82% → typical fee ≈ 0.14R. Not yet acted on — see
Next Steps #1.

**Tested a user-proposed variation → REJECTED on the numbers (`e8ee4d7`, `29778b3`).**
Question: must the M30 cross wait for an H1 candle to *close* above EMA50 (locked model = yes),
or should it fire as soon as price is above EMA50 intrabar? Built
`scripts/variation_h1_intrabar.py` — only the price compared against EMA50 changes (the EMA
level itself is the last closed H1's in both, since EMAs update on close); every other gate
identical. Data refreshed through **2026-07-21** (previous CSVs stopped 05-12).

| | trades | EV | net EV¹ | net total | net maxDD |
|---|---|---|---|---|---|
| **V3 locked** | 498 | +0.960R | **+0.823R** | **+409.9R** | −42.9R |
| V3 intrabar | 518 | +0.917R | +0.779R | +403.6R | −41.4R |

¹net of real HL fees, `fee_R = 0.09/stop%`

**Why it loses — displacement, not signal quality.** 492 trades are byte-identical. The
variation adds 26 weak trades (+0.269R net EV) and **loses 6 of the baseline's biggest
winners** (+2.222R net EV) — entering intrabar occupies the single position slot, so the
better signal an hour later gets skipped. Net −6.3R over 9 years. **Locked model retained;
the live bot was not touched.**

CSVs (repo root + Desktop): `trades_v3_h1intrabar_to_now.csv`, `trades_v3_locked_to_now.csv`.

**Shipped `ops_kit/` — the EMS monitoring+safety system, packaged to port to a 2nd bot
(`d59c53d`).** The user is automating another Render bot (**swing-bot**, RD3I USDJPY on a
**cTrader** account) and asked for "the same monitoring." Surfaced that "heartbeat +
Telegram" is only 2 of **5 ops layers**; the real gaps are risk/state/execution-safety.
`ops_kit/` (10 files, 845 lines, at this repo's root) extracts the venue-agnostic parts:
- **Verbatim, zero coupling:** `daylimit.py` (kill switch), `position.py` (state + boot
  reconcile 4-case matrix), `monitor.py` (Telegram + healthchecks senders + generic
  cards), `loop.py` (run_forever: reconcile→sleep+ping→tick→repeat, "one bad tick never
  kills the worker").
- **The seam:** `broker_protocol.py` — the exact method surface the loop calls; implement
  once per venue, reuse everything else.
- **Deploy + brief:** `render.worker.yaml`, `.env.example`, and `PORT_BRIEF.md` (the
  handover doc: verbatim-vs-adapt table, cTrader specifics, build order, 8-point
  acceptance checklist). `monitor.fmt_exit` already carries a swap/funding line — closing
  the gap EMS's own exit card still has. Generic modules smoke-tested (kill switch,
  reconcile, formatters, scheduler math). Nothing in `ems_live/` touched.

**Delivered it to swing-bot (a MERGE, not greenfield).** swing-bot
(`chillcharcontacto-cyber/swing-bot`, local `…/TradingEdgeLabs/strategies/swing`, branch
`main`) already has the cTrader broker + OAuth + `token_refresh`, `sizing.py`, `dry_run`,
`state` (4 files) and `reconcile` (2 files). Verified the genuine gaps (0 files reference
them): **Telegram, healthchecks, daily kill switch**, and **no persistent disk in its
render.yaml → state wiped on every redeploy** (a latent bug). Copied `ops_kit/` into the
swing repo (uncommitted), gave a merge-aware build prompt, and handed off to the swing-bot
Claude session, which confirmed its cwd + that `ops_kit/PORT_BRIEF.md` is present
(12042 bytes, byte-identical) and is proceeding read-before-wire. Note broker exposes
`open_positions()` (plural/list) vs the contract's `open_position()` (None|dict) — small
shape adapt flagged.

---

## Session — first testnet trade verified; WENT LIVE ON MAINNET 🎯

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

## Earlier Session Summary — verified the fixes in dry-run, ARMED testnet

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

## Session — two live-execution bugs fixed (sizing + entry timing)

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

## Session — step-by-step ladder notifications + daily heartbeat

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

## Session — Bot DEPLOYED & LIVE on Render + full monitoring 🟢

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

**LIVE ON MAINNET (real money) — running unattended, one big trade open.** Render worker
`ems-live-bot`: `HL_TESTNET=false`, `EMS_DRY_RUN=false`, **risk $6/trade** (raised from $3
this session), kill 10 losses/day (= −$60/day max), `EMS_STATE_PATH=/data/ems_mainnet_state.json`.
Verify the 🚀 card reads `risk $6.0` after the redeploy.
- Master `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`; **~$1,414 equity** (as of 08-23, grown
  with the open winner). Funded up from ~$325 via Phantom→KuCoin→Arbitrum→HL. Unified Account.
- Mainnet agent **`ems-bot-main`** `0xc07aA2354249ba34D7a4436fEDEC6864Dd07b8Fd` authorized
  (~180 days from 2026-07-13; re-authorize before it lapses). Key in Render env only, trade-only.
- **OPEN trade (as of 08-27):** BTC long 0.02151 @ 64,477, resting Stop Market 64,337 (= the
  old $3 risk), **+~$248 unrealized** at BTC ~76k — a deep runner still riding to the H1<EMA100
  exit (stop sits ~15% below price). Sized at $3 (opened 08-19, before the change); next entry
  sizes at $6. Health-checked clean 08-23 (see Last Session Update).
- Auto-leverage: at $6 risk on ~$1.3k, a ~1% stop → ~1×, tight stops → higher lev, all liq-safe.
- Testnet still exists (`ems-bot` agent, ~$998 spot) but the worker points at mainnet.

**Mainnet execution PROVEN.** Prior closes: #1 +$0.55, #2 −$3.61, 08-07→08-10 −$1.47; current
08-19 long deep in profit. Deep-book fills tight. Let it run + watch Telegram; to check any
trade, pull on-chain fills + `userFunding` for `0x18ce…6964` on MAINNET.

Reconciliation gotchas learned: the exit card is **fee-only** (excludes funding — matters on
long holds, e.g. −$0.11 over 45h) and books SL exits at the trigger, not the fill. Deviation%
is inflated on small-R trades (fee is a big % of a small move) — read the $ not the %.
**Fee drag is inversely proportional to stop width** (`fee_R = 0.09/stop%`): a 0.29% stop
costs 0.30R in fees, a 1.6% stop only 0.05R. Live-confirmed on trade #2.

**Model is settled — do not re-litigate without new data.** The H1-intrabar variation was
tested this session and rejected (see Last Session Summary). Waiting for the H1 *close* is
load-bearing: it protects the single position slot for the biggest winners.

**`ops_kit/` shipped here; the swing-bot merge runs in that repo's own session.** For
backtest-api nothing is mid-flight — `ops_kit/` is the canonical source of the shared ops
tooling and is done. The actual integration (adding Telegram + healthchecks + kill switch +
a persistent disk into swing-bot's `live/`) is being done by the swing-bot Claude session,
tracked in swing-bot's own handoff, not here. This repo just hosts the kit.

---

## Parked / Unfinished

**EMS live bot — LIVE ON MAINNET; open refinements (none blocking):**
- **Smart 429 alerting (designed, deferred 08-23).** Today's fix: in `run_forever`'s tick
  handler, detect a transient-429 exhaustion and (a) log it quietly instead of a Telegram
  TICK ERROR, (b) count them and alert once if >~5/day (pattern = signal), (c) keep every
  NON-429 error loud. User chose to leave alerting as-is for now and watch the pattern
  manually — build only if the blip cards become annoying or a pattern shows.
- **Fee-drag study → possibly raise `min_risk_pct`.** Live trade #2 paid 0.30R in fees on a
  0.29% stop. `fee_R = 0.09/stop%`, so sub-0.5% stops cost >0.18R each. Open question: does
  filtering out the tightest stops raise *net* EV, or are those trades net-positive anyway?
  The harness in `scripts/variation_h1_intrabar.py` already computes net-of-fee EV per bucket
  — sweep `min_risk_pct` over e.g. 0.1/0.3/0.5/0.75/1.0% and compare netEV/netTotal/netDD.
  Careful: raising it also removes trades, so judge on net total R and DD, not EV alone.
- **H1-intrabar variation — TESTED AND REJECTED (2026-07-21), do not redo.** Result and
  reasoning in the Last Session Summary. One follow-up never run: test the 26 intrabar-only
  signals as an *additive* entry that fires **only when flat**, which removes the displacement
  cost and isolates whether those signals have standalone edge (+0.269R net EV suggests weak
  but positive).
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

## Next Steps — bot is LIVE on mainnet and settled; work is now analysis + reporting

Deploy/arm/mainnet steps are all DONE. Nothing is blocking; the bot runs unattended.

⭐ **TIER-4 IS LIVE (activation COMPLETE 08-27).** Brain `https://ems-brain.onrender.com` (clean
signer `0xd0B67b43ce1459381871aF5b64FBB47CC4404513`); public thin-client repo
`github.com/chillcharcontacto-cyber/ems-thin-client`. Only remaining action is **onboarding real
partners, on demand** (per-partner operator recipe — full detail in `docs/BRAIN_OPERATIONS.md`):
1) Send the partner the repo link + "start SETUP.md, do the Hyperliquid part first."
2) They create + fund their HL account → send you their **main address** (the exact one they
   deploy with as `HL_MASTER_ADDRESS`; the license binds to it, a mismatch is denied).
3) Mint a key: `py -c "import secrets; print('EMS-'+secrets.token_hex(12))"`.
4) Render → `ems-brain` → Environment → `EMS_LICENSE_KEYS` → add `,"THEIR-KEY":"0xTheirAddress"`
   inside the `{}` → Save and deploy. (Claude can hand you the exact JSON value to avoid a fumble.)
5) Send the partner their key + the repo link; they finish SETUP.md and arm.
Revoke = remove their line → Save and deploy (their bot stops next tick).
NOTE: the live brain is the standalone `ems-brain` service (NOT the render.yaml `backtest-api`
web block — that was never deployed; created manually to avoid a blueprint re-sync resetting the
worker). Deliverables: brain on `main` (`ems_brain.py`, `/ems/decision`); thin client
`…/TradingEdgeLabs/ems-thin-client`; Tier-1 kit `…/ems-live-bot` = superseded unprotected option.

⭐ **USER REMINDER (outstanding 2 sessions now):** start tracking the EMS indicator on
**GU (GBP/USD) from Jan 5th** onward, and keep tracking it. (User's words: "get it from
jan 5th on GU and keep tracking the indicator".) **Ask what they mean before building**:
forward-track the live indicator on GBPUSD, and/or backtest GBPUSD from 2026-01-05?
Note GU is **forex → Twelve Data** (`TWELVEDATA_API_KEY`), not the Binance/crypto feed,
so this needs a new data path in the EMS stack.

1. **Fee-drag study (recommended first).** Sweep `min_risk_pct` (0.1/0.3/0.5/0.75/1.0%)
   and compare **net** EV / total R / maxDD after `fee_R = 0.09/stop%`. Motivated by live
   trade #2 burning 0.30R on a 0.29% stop. Decide from net total R + DD, not EV.
2. **Exit-card accuracy** — fold **funding** + the **actual SL fill** into the exit record
   so the card, deviation %, and the kill-switch ledger reflect true net. Highest-value
   *reporting* fix now that it's real money.
3. **Optional:** test the 26 intrabar-only signals as an additive entry that fires only
   when flat (isolates their standalone edge without the displacement cost).
4. **Passive, ongoing:** let it run, watch Telegram, reconcile each closed trade on-chain
   (fills + `userFunding` for `0x18ce…6964`, MAINNET).
5. **swing-bot ops merge (cross-repo, low-touch here):** the swing-bot session is wiring
   `ops_kit/` in. If the user asks about it from THIS repo, the kit source is `ops_kit/`
   + `PORT_BRIEF.md`; the real gaps it fills there are Telegram + healthchecks + kill
   switch + a persistent disk. If `ops_kit/` itself needs a fix, patch it here (canonical)
   and the user re-copies. Don't do swing-bot's integration from this session.

Verify after any redeploy: 🚀 card reads `testnet=False dry_run=False risk $6.0` and
`kill 10 losses/day`; step/heartbeat cards have real data (no TICK ERROR).
Re-authorize the mainnet agent before its ~180-day expiry (authorized 2026-07-13).
