# Decisions Log

A running list of architectural and product decisions, newest first.

---

## 2026-08-27

**Tier-4 activated end-to-end — brain live, thin-client repo public, verified with the shipped code**
Finished Tier-4: the brain runs on the standalone `ems-brain` Free web service
(`https://ems-brain.onrender.com`), and the partner bot is a PUBLIC repo
(`github.com/chillcharcontacto-cyber/ems-thin-client`) with the brain URL + operator signer baked
into render.yaml + the Deploy-to-Render button. Verification standard adopted: don't consider a
release done until the SHIPPED client code proves it — used `thin.brain_client.BrainClient`
against the live brain and confirmed a real signed decision verifies against the operator signer
(account + nonce + expiry) and that a wrong-signer client is rejected. Onboarding is now a
per-partner operator recipe (license key bound to the partner's exact HL address), not a build.

**A signing key that touches a screenshot/log is compromised — rotate, and never screenshot it**
The signing key was rotated TWICE during activation because the first two private keys appeared in
Render screenshots the user shared. Decision/lesson: treat any key that has been screenshotted,
pasted into chat, or logged as compromised and rotate it before it guards anything real; generate
the keypair and copy the private half **directly** into the host env, never screenshotting the
terminal. The live signer `0xd0B67b43ce1459381871aF5b64FBB47CC4404513` was generated this way (its
private half never left the user's machine). Also: public-repo pushes must use the GitHub
**noreply** email (`{id}+{user}@users.noreply.github.com`) or GH007 blocks them under email
privacy — standardize on noreply for any repo the user publishes.

**429 spam diagnosed as Render shared-IP rate-limiting — mitigate, don't chase a "true zero"**
Frequent 429 TICK ERRORs (8/day) were diagnosed as HL rate-limiting Render's shared outbound IP
(HL is fine from other IPs; the worker makes ~1 HL call/tick, so it isn't over-calling). It's
outside the bot's control and harmless to an OPEN position (the stop rests on the exchange). The
only real exposure was a 429 landing on an ENTRY bar → a missed trade. Decision: mitigate at the
client (hardened retry + catch-up + throttle) rather than pay for a Render dedicated IP — and be
honest that a client cannot guarantee ZERO missed entries (if the exchange refuses your IP for
the whole window, any bot misses).

**Retry lives in one shared module; reads vs writes differ; entries defer only on 429**
Extracted the HTTP retry into `ems_live/nethttp.py` (used by broker.py AND feed.py) so a
rate-limit is ridden out at EVERY exchange touchpoint (candles, SL adaptation, orders), not just
the SDK calls. Hardened to 6 tries with backoff + jitter (~45s, well inside a 30-min bar; jitter
de-syncs from a shared-IP limit). READS retry on 429+5xx; WRITES (orders) retry on **429 only** —
a 429 is rejected before matching (safe to resend) whereas a 5xx could hide a lost success.
Consistent rule reused by the missed-entry catch-up: an entry is DEFERRED-and-retried only on a
429 (definitely not executed); an ambiguous non-429 order error ABORTS with no pending, so a
trade is never double-entered.

**Missed-entry catch-up via a bounded pending-entry, not a decider look-back**
Chose to persist a `<state>.pending` sidecar when an entry defers, and retry it on subsequent
flat ticks while the crossover is ≤ CATCHUP_MAX_BARS (3) old and still valid (guard_order
re-checks price vs stop), rather than widening the decider's look-back (which would touch the
backtest-shared predicates and risk parity drift). The entry path was extracted verbatim into
`runner._attempt_entry()` so the normal path is unchanged (parity + 103 tests confirm) and the
catch-up is a thin additive layer.

**429 tick alerts are throttled to one/day, but never blanket-silenced**
A 429/5xx tick error now sends ONE Telegram card per UTC day plus a running count sidecar; any
NON-429 error stays loud immediately. This kills the per-blip spam while preserving the signal
if the rate-limit pattern worsens — consistent with the 2026-08-23 decision not to blanket-mute
the catch-all TICK ERROR card.

---

## 2026-08-26

**Brain runs as a standalone `ems-brain` web service, NOT the blueprint's `backtest-api` one**
Activating Tier-4 revealed the `backtest-api` web service defined in render.yaml was never
actually deployed (only the `ems-live-bot` worker existed). Rather than re-sync the blueprint to
create it — which would reset the live worker's dashboard overrides back to render.yaml's SAFE
DEFAULTS (testnet/dry-run/$20) and flip the real bot off — created the brain as a **separate,
manually-configured Free web service** named `ems-brain` (`https://ems-brain.onrender.com`, from
`main`, `uvicorn api:app`). Decision: the live brain is `ems-brain`, independent of the blueprint;
the render.yaml `backtest-api` web block stays unused. Free tier is fine (cold-start ~30-60s is
nothing at 30-min bars; the endpoint self-warms on the first thin-client call each cycle).

**Verified the full Tier-4 chain before trusting it, and required a key rotation after exposure**
Confirmed end-to-end on the live brain: `POST /ems/decision` with a bound license → HTTP 200 → the
strategy ran on live candles → the returned decision's signature recovers to the operator's signer
address, account-bound + nonce'd + unexpired. Only after that proof is the brain considered
"working." Separately: the private `EMS_BRAIN_SIGNING_KEY` was captured in a Render screenshot
(thus in the chat transcript), so decided it MUST be rotated before issuing any real partner key —
a signing key that has touched any log/screenshot is treated as compromised, even if the practical
attack (MITM + resign) is hard. Rotation = new keypair, update the env var, ship the new signer
address in the thin client.

---

## 2026-08-23

**Keep the 429 TICK-ERROR alert as-is — do not blanket-silence it**
After a second harmless 429 alert, decided NOT to silence 429 TICK-ERROR cards, because that
card is a **catch-all** for any per-bar failure (feed/broker/logic errors too) — muting it
would hide real problems. A *single* 429 is noise (HL edge rate-limiting; the bot polls only
once per 30-min bar, so it isn't the cause and it self-corrects next bar with the position
protected by the resting stop). A *pattern* of 429s would be signal (HL limit change / a
loop). Resolution: leave alerting unchanged for now; the user watches for a daily pattern
manually. A "smart" handler (quiet-log single 429s, alert on a daily threshold, keep non-429
errors loud) is designed and parked — build only if the noise or a pattern warrants it.

---

## 2026-08-21

**Retry Hyperliquid calls on transient 429/5xx — reads freely, writes on 429 only**
A live 429 (HL's CloudFront edge rate-limiting a poll) aborted a whole tick. Decision: wrap
every HL SDK call in a backoff-retry, but split the policy by idempotency — READS retry on
429+5xx (safe to repeat), WRITES (orders) retry on **429 only** because a 429 is rejected
before the matching engine (cannot double-execute), whereas a 5xx could mask a lost success
response and resending could double-fill. A genuine outage still surfaces (final attempt
raises → the tick handler retries next bar). Rationale: turn a transient blip into a silent
in-tick self-heal without ever risking a duplicate order.

**Partners get the bot self-serve (own Render + own Hyperliquid), never operator-hosted**
For giving trading partners the same EMS bot: each partner runs their OWN Render worker on
their OWN HL account. Rejected operator-hosting because cost scales per partner AND the
operator would hold everyone's agent keys (de-facto fund operator = custody + liability).
Self-serve = $0 to the operator, full isolation, operator never touches partner funds/keys.
The operator's job is only to make setup easy (a template repo + a non-coder SETUP).

**Protecting the strategy: gate USE, not copying — chose a server-side brain (Tier-4)**
The user wanted partners unable to redistribute a working bot without approval. Established
the hard truth: code that runs on someone's machine can't be made un-copyable (all DRM is
just cost-raising). The only robust lever is to NOT ship the secret — run the strategy on a
server the operator controls. Decision: build Tier-4 — a **brain** (holds the strategy, signs
decisions) + a **thin client** (holds no strategy, only executes signed, account-bound
decisions). A stripped or forwarded client is inert without an approved license key the
operator issues and binds to one HL account; revocation is instant. Accepted residual leak:
partners can observe the signals they receive and infer the *idea* over time (unavoidable —
the trades are visible on their own account), but they cannot run an autonomous bot for
anyone without the operator's live server saying yes.

**Brain colocates in the existing web service (~$0); thin client carries zero strategy**
Implementation choices for Tier-4: (1) the brain is one new endpoint (`/ems/decision`) on the
already-deployed `backtest-api` web service — no new host, ~$0 (free tier's ~30-60s cold start
is negligible at 30-min bars). (2) Signing uses `eth_account` (already a dependency via the HL
SDK — no new library); the client verifies signature + account-binding + nonce + expiry and
fails safe (does nothing) on any mismatch or if the brain is unreachable. (3) The license map
`{key: bound_HL_address}` lives in a web-service env var — add a line to approve, remove to
revoke. (4) The brain reuses the existing `decider` predicates, so brain decisions are
identical to the monolith bot (no strategy drift), and the thin client needs no pandas/numpy
and no strategy names (exit labels genericized). The strategy edge never leaves the operator.

---

## 2026-08-20

**Raised live risk per trade $3 → $6 after funding the account to ~$1,279**
The mainnet account was funded up from ~$325 to ~$1,279 (Phantom→KuCoin→Arbitrum→HL), so
doubled the per-trade risk from $3 to $6 (Render `EMS_RISK_USD`). Rationale: $6 on ~$1.3k
keeps auto-leverage sizing well within liq-safe bounds, and the strategy's edge is validated
enough (live + 9y backtest) to scale risk with the larger balance. Consequence noted: the
count-based kill switch now caps a bad day at 10 × $6 = −$60 (was −$30). `EMS_RISK_USD` is
read only at entry sizing, so the change applies to the NEXT trade; the open trade keeps its
$3 sizing and exits normally. Applied via dashboard **Save and deploy** (restart to re-read
the env; no rebuild needed since no code changed).

**`render.yaml` holds SAFE DEFAULTS, not live values — dashboard is the source of truth**
Confirmed/formalized the deploy pattern: the blueprint ships `HL_TESTNET=true`,
`EMS_DRY_RUN=true`, `EMS_RISK_USD=20` on purpose so a fresh blueprint deploy comes up SAFE
(testnet, no orders). The live mainnet config (testnet=false, dry_run=false, risk=6, mainnet
state path) exists ONLY as Render dashboard overrides. Decision: do NOT sync render.yaml to
the live values — that would make a fresh/accidental blueprint deploy go straight to mainnet
real-money. Trade-off accepted: the repo file doesn't reflect production, so a blueprint
re-sync requires re-applying the dashboard overrides afterward (documented in the handoff).

---

## 2026-07-21

**A live trading bot needs 5 ops layers, not 2 — extracted them as a portable `ops_kit/`**
When scoping "the same monitoring as EMS" for a second bot, named the full operational
surface so the copy isn't half-safe: (1) liveness (dead-man's-switch ping + host crash
alerts), (2) observability (Telegram trade cards), (3) risk (daily-loss kill switch),
(4) state & recovery (atomic JSON on a persistent disk + boot reconcile vs broker truth),
(5) execution safety (save-state-before-stop, never orphan an unprotected position, size
off actual fills). "Heartbeat + Telegram" is only layers 1–2; the forgotten 3–5 are what
make a bot safe to leave alone with money. Decision: package the venue-agnostic parts of
`ems_live/` as a reusable `ops_kit/` (daylimit, position, monitor senders, the loop, a
broker Protocol, render worker template, and a PORT_BRIEF) rather than re-describing them
in prose each time. The kit lives in backtest-api as the canonical source; other bots copy
it and implement only the broker seam. Rationale: reference code the receiving session
adapts beats a spec it has to reverse-engineer, and it forces parity with a system already
running real money.

**Port to a new venue = implement one broker Protocol; everything else is reused**
Chose a single adaptation seam (`ops_kit/broker_protocol.py`) as the only venue-specific
surface. The loop, reconcile, kill switch, and monitoring never import a venue SDK — they
only call the Protocol's methods. Consequence for cTrader (swing-bot): its equity-relative
AUTO-LEVERAGE (a Hyperliquid-perp concept) does NOT port — cTrader leverage is broker-set
and there's no per-order liquidation price, so `plan_trade`'s liq math is dropped and
replaced with plain forex volume-from-risk sizing; `update_leverage`/`set_margin_mode`
become no-ops. Costs on the exit card become commission + swap instead of taker fee +
funding. The generic `fmt_exit` was given a `swap` parameter up front so the new bot is
truthful about overnight cost from day 1 — a gap EMS's own card still has.

**swing-bot integration is a MERGE, not a greenfield build**
Inspected swing-bot before handing off: it already has the cTrader broker + OAuth +
token_refresh, sizing, dry_run, and state+reconcile under `live/`. The verified gaps
(0 files) are Telegram, healthchecks, and a daily kill switch — plus its render.yaml has
no persistent disk, so its existing state code is wiped on every redeploy (a latent bug).
Decision: instruct the swing-bot session to MERGE the missing layers into its existing
`live/runner.py` and add a persistent disk, NOT to rebuild the broker/state it already has,
and to reconcile its state machine against `ops_kit/position.py`'s 4-case matrix. Rationale:
telling it to "build the whole thing" would have it duplicate or fight working code.

---

**H1 trend filter keeps the wait-for-close requirement — intrabar variation REJECTED**
Tested taking the M30 cross as soon as price is above the H1 EMA50, instead of requiring the
last H1 candle to have CLOSED above it (`scripts/variation_h1_intrabar.py`, data through
2026-07-21). Net of real HL fees the locked model wins: **+409.9R vs +403.6R** (netEV +0.823R
vs +0.779R) across 9 years. The mechanism matters more than the size of the gap: 492 of the
trades are byte-identical, and the variation's 26 extra trades are weak (+0.269R net EV) while
it **loses 6 of the baseline's biggest winners** (+2.222R net EV). Entering intrabar fills the
single position slot early, so the stronger signal an hour later gets skipped — a
**displacement** cost, not a signal-quality difference. Conclusion: waiting for the H1 close is
load-bearing (it rations a scarce position slot toward higher-quality setups), so the locked V3
stands and the live bot was not modified. Follow-up left open: test those 26 signals as an
additive entry that fires only when flat, which would isolate their standalone edge.

**Fee drag is inversely proportional to stop width — a real, live-confirmed cost**
Because fixed-$ risk implies `notional = risk / stop%`, a tighter stop mechanically means a
larger notional and therefore larger fees *relative to risk*: `fee_R = 0.09 / stop%` at HL's
0.09% round-trip taker. Mainnet trade #2 (0.29% stop) paid $0.90 = **0.30R** in fees, versus
0.05R for trade #1's 1.6% stop. Decision: adopt `fee_R = 0.09/stop%` as the standard cost line
in all EMS analysis (the older flat 0.20% round-trip assumption was conservative by comparison),
and open a study on whether raising `min_risk_pct` to filter the tightest stops improves net
EV — deliberately NOT changing the live parameter until that study is run, since raising it also
removes trades and must be judged on net total R and drawdown, not EV alone.

**Backtest CSVs must be regenerated to the actual current date, not a hardcoded one**
The variation run initially hardcoded an end date a few days stale, silently cutting the
"until now" CSVs short. Caught and refetched. Rule going forward: any "to now" export states
its true data range in the output, and the range gets verified against the real current date
before the numbers are reported.

---

## 2026-07-17

**First mainnet trades reconciled: card P&L is fee-only; deviation% misleads on small R**
Trade #1 (mainnet, ~45h hold) closed a real +$0.55 win, but exposed two reporting facts to
keep in mind: (1) the exit card's net P&L/deviation are **fee-only** — they exclude perp
**funding**, which was −$0.11 over the 45h hold and is real on long holds; (2) the deviation%
metric is **inflated for small-R trades** because the ~fixed per-trade fee (≈0.05R at a ~1.6%
stop) is a large fraction of a small win (−19.8% on a +0.28R exit) but a tiny fraction of a
big one — so read the **dollar** cost, not the %, on small trades. Decision: leave the card
as-is for now (user opted not to add funding yet), but the funding + actual-SL-fill additions
are the top reporting refinement (see handoff Parked). Deep-book mainnet fills were tight (no
testnet-style slippage), validating the testnet→mainnet execution-quality expectation.

---

## 2026-07-13

**Went LIVE on mainnet at $3 risk after verifying the first testnet execution**
The first real testnet trade (2026-07-12) executed correctly end-to-end (on-time entry,
resting structural stop triggered, position closed), confirming both the auto-leverage
sizing and entry-timing fixes on-chain. The −1.19R (vs −1.00R) was testnet thin-book
slippage, which mainnet's deep book largely removes — so testnet-vs-mainnet execution
quality favors mainnet, and the go/no-go was made on that basis. Funded ~$328, authorized
a dedicated MAINNET agent (`ems-bot-main`, trade-only, ~180d), and armed with
`EMS_RISK_USD=3` and a mainnet-specific `EMS_STATE_PATH` so testnet history doesn't bleed
into the live ledger. Risk sized to the account: $3/trade ≈ 1%, 10-loss/day kill = −$30.

**Known reporting gap accepted for go-live: SL exits book the trigger price, not the fill**
The bot records a stopped exit at the SL trigger (clean −1.00R) rather than the actual
(possibly slipped) fill, so realized P&L/ledger/deviation slightly understate a slipped
stop. Accepted as non-blocking for mainnet (deep-book slippage is small) but flagged as
the top refinement now that it's real money — the fix reads the real SL fill for the
exit record so P&L and the kill switch reflect reality.

---

## 2026-07-10

**Testnet Unified Account: spot USDC is the perp trading margin — no transfer needed**
The testnet master holds ~$998 in SPOT and $0 in perp `marginSummary` (cosmetic under a
Unified Account). Verified empirically: a live 0.0002 BTC perp market order FILLED with
funds in spot, then closed. So the bot needs NO spot→perp transfer step on testnet; the
spot balance backs the perp order directly, and `account_value()` (perp+spot) already
reports the right tradable equity for auto-leverage.

**Armed testnet (`EMS_DRY_RUN=false`) to get the first real autonomous execution**
After confirming the dry-run cards reproduce correct auto-leverage sizing (e.g. a 0.31%
tight-stop trade that the old $500 ceiling would have refused now sizes at ~8×, risk $20)
and correct on-time entries, flipped the worker to armed. The bot now places real
(fake-money) testnet orders on the next aligned signal. On-chain reconciliation of that
first fill (entry/stop/size/leverage) is the remaining end-to-end proof.

**Sizing is equity-relative auto-leverage, not fixed leverage + a dollar ceiling**
A fixed-$ risk system produces notional = risk_usd/stop%, which is large for tight
stops — so any fixed-dollar notional ceiling (the old $500) or fixed leverage fights the
model and refuses valid trades. Decision: `size = risk_usd/(entry-sl)` is the sole risk
control and is never scaled to satisfy a guard; instead LEVERAGE adapts per trade =
smallest that fits margin (`ceil(notional/(equity*buffer))`) capped so the isolated
liquidation stays beyond the structural stop (`lev <= 1/(stop%*liq_safety_mult+maint)`).
If the account can't carry full size liq-safely, resize DOWN (flagged) — never
over-leverage, never silently drop. Everything reads live `account_value()` so testnet
and mainnet behave identically relative to their own equity, and every knob is env-tunable.
This is the ONLY sanctioned deviation from exact risk_usd (an under-capitalized account),
and it is explicit. Leverage is not extra risk here: the stop caps loss at risk_usd and
liquidation is kept beyond the stop.

**Live entry fires on the just-closed crossover bar, not one bar later**
The backtest enters at open[i] on a cross at bar i-1. Acting on closed bars, the live
tick must therefore enter the instant the CROSSOVER bar closes (that bar is the latest
closed bar) — at the live mid ~ open[i+1] ~ close[i] — not wait for the next bar to close
(which fired ~30 min late at a worse fill). `check_entry_live()` implements this and is
proven (parity test) to select the identical trades as the backtest's `check_entry(i+1)`
— same SL/crossover/anchor — changing only the timing/price, never which trades. The
offline backtest (replay/simulate) is unchanged; live gets its own correctly-timed entry.

**Adversarial review is mandatory before deploying live-money path changes**
The sizing change was audited (63 scenarios, 5 agents) and then adversarially reviewed
(15 agents, 12→6 confirmed) before commit; the review caught 4 real follow-on bugs
(silent unsafe 1x on ultra-wide stops, blind leverage on equity-read failure, orphaned
unprotected position on a failed flatten, partial-fill P&L overstatement) that were
fixed pre-deploy. Policy: money-path changes get an audit + adversarial verify, not just
unit tests.

---

## 2026-06-27

**Status notifications follow the setup hierarchy (H4→H1→M30 ladder), not raw flips**
Reporting every gate flip still surfaced lower-timeframe noise while a higher TF was
blocking longs. Changed the default to `status_mode=steps`: a "stage" is computed
top-down (0 H4 bearish/blocked, 1 H4 ok/waiting H1, 2 armed/waiting M30 cross) and a
step message is sent only when the STAGE changes. So while H4 is bearish, H1/M30 churn
is silent; the bot only pings as the setup actually climbs (or regresses) the ladder,
ending in the 🟢 entry. Rationale: a notification is only useful if it moves you closer
to (or further from) an entry given the higher timeframes. Plus a once-per-UTC-day
"still alive" heartbeat so multi-day flat stretches still confirm liveness on Telegram.
Supersedes the prior `change` mode (kept as an option). Persisted to `<state>.status`.

**FLAT status card is event-driven (on gate flip), not time-driven**
The ⚪ FLAT card was sent every tick (48/day) — pure noise. Changed to `status_mode`
(default `change`): send only when a gate's boolean flips (M30 ema20>ema50, H1 close>
ema50, H4 ema20>ema100), with a `Δ` line naming the flip. The last gate-state persists
to a `<state>.status` sidecar so change is detected across the stateless-per-tick loop.
A card is only useful when the market actually moves a gate (setup building/breaking);
otherwise stay silent. Modes `change`/`always`/`off` via `EMS_STATUS_MODE`.

---

## 2026-06-25

**Bot deployed LIVE on Render (testnet, dry_run); monitoring is mandatory before arming**
The EMS-V3 bot now runs autonomously on a Render worker. Monitoring shipped alongside,
not after: Telegram event cards (entry/exit/status/blocked) + healthchecks.io liveness.
Rationale proven same day — the Binance-451 production bug surfaced on the phone in
seconds via the new Telegram alerting instead of failing silently.

**Live feed uses data-api.binance.vision, not api.binance.com (Render US IP → 451)**
`api.binance.com` geo-blocks US IPs (Render) with HTTP 451. The live feed (`ems_live/
feed.py`) switched to Binance's public market-data mirror `data-api.binance.vision`
(same klines, no geo-block, no key). The backtest fetcher (`ems/data.py`) stays on
api.binance.com since it runs from the user's own IP.

**Kill switch is count-based (10 losing trades/day), not R-based**
The R-based daily cap (3R) tripped on normal variance for this low-WR (~23%), streaky
strategy (backtest longest losing streak 14–17) and would clip the tested edge. Switched
the default to a losing-trade count (10/day, ≈ −10R worst case) which only catches a
genuinely abnormal day. Either limit can be set via env; R-based now defaults off.
Reminder logged in handoff: the backtest had NO kill switch — this is a live-only
safety overlay, so live can diverge from backtested results on extreme days.

**EXIT reporting separates model R from net R (cost deviation)**
Each close reports the clean model R, the after-fee net R, and the deviation % (how far
costs moved the result). With fixed-$ risk, gross $ = model_r × risk_usd exactly, so
deviation currently isolates fee drag (bigger on tight stops = more notional per 1R);
slippage from real fills is a parked enhancement.

**Liveness = minute heartbeat + Render native alerts (two failure modes)**
Crash/exit → Render's native notifications (instant) + auto-restart. Hang (alive but
frozen) → only a heartbeat catches it, so the loop pings healthchecks every 60s
(healthchecks Period 1m/Grace 2m → ~3-min detection). Heartbeat is silent (no log
spam) and never touches trading logic. Tick errors go to Telegram, NOT marked as
healthchecks-down (process is alive and will retry).

---

## 2026-06-16

**Render dashboard cannot be auto-driven by Claude-in-Chrome — use watch-and-guide**
The Render dashboard is a single-page app with persistent connections, so it never
reaches `document_idle`. Every Claude-in-Chrome DOM/screenshot tool (`find`,
`read_page`, `screenshot`, `form_input`) times out at 45s, and blind pixel-clicks
can't be calibrated without an in-Chrome screenshot. computer-use desktop screenshots
DO render the page (good for seeing), but Chrome is read-tier so computer-use cannot
click. Decision: for Render (and similar never-idle SPAs), do NOT attempt full
automation — Claude watches via desktop screenshots and the user performs the clicks.
`navigate` (URL changes) still works, so jump straight to
`dashboard.render.com/select-repo?type=blueprint` to skip the button clicks.

---

## 2026-06-11

**Per-tick exchange reconcile — the exchange is checked every tick, not just on boot**
The resting stop lives on the exchange, so when it fires the bot must learn of it. The
live `tick()` now queries the exchange position at the top of the IN_POSITION branch;
if flat, the stop fired → record −1R and go flat. Without this a long-running process
would never notice a stop fill (it only reconciled on restart) and would keep managing
a phantom position. The designed SL outcome is −1R, so that is what's booked.

**Max-daily-loss kill switch — circuit breaker before any unattended arming**
`DayLedger` accumulates realized R per UTC day (rolls at the date boundary). New
entries halt for the rest of the day once realized R ≤ −`max_daily_loss_r` (default
3R; 0 disables). This is a hard precondition for running armed on Render — a bad run
of stops can't bleed indefinitely. State persisted as a JSON sidecar next to the
position state.

**Render deployment is a `worker` with a persistent disk; secrets stay in the dashboard**
The bot is a background `worker` (no HTTP port), not a web service. A 1 GB disk at
`/data` persists the position state + day-ledger across deploys (Render's default FS is
ephemeral; without the disk a redeploy mid-position loses SL/exit context — the resting
stop still protects the position, but management degrades). HL_MASTER_ADDRESS and
HL_AGENT_KEY are `sync: false` so they are entered in the dashboard and never committed.
`EMS_DRY_RUN=true` is the committed default; arming is a deliberate dashboard flip.

---

## 2026-06-10

**Final live model LOCKED: V3 confluence (H4 EMA20 > EMA100), H1 EMA100 exit**
After comparing six variants — confluence V3 (EMA20>slow, slow∈{50,100}) vs three
price-above-single-H4-EMA models (base / EMA50 / EMA100, all on an EMA50 H1 exit) —
the confluence **V3 EMA20>EMA100** wins every quality metric (EV 1.00, PF 2.63) and
is the only robustness-validated one. R2 (price>H4 EMA100) had a shallower historical
max-DD, but that edge is a mirage: Monte-Carlo p95 drawdowns are ~equal (−50 vs −52 R)
— the extra 171 trades just smooth the path without adding net value. Structural
rationale: requiring fast-EMA-above-slow waits for a *confirmed* trend; price barely
poking one average lets in the trades that net negative. The HL bot runs this model.

**Clarified a long-standing definition gap: "V3" = confluence, not price>EMA**
The coded V3 filter is `H4 EMA20 > H4 EMA(slow)` (two H4 EMAs), NOT "price above a
single H4 EMA" as previously described in passing. All committed V3 backtests/CSVs and
the robustness check are the confluence version. The price>single-EMA variant was
tested separately (`scripts/three_models.py`) and rejected per the decision above.

**Bot strategy in the live decider mirrors the engine, enforced by a parity test**
The H4 confluence gate added to `ems_live/decider.py` (`check_h4_confluence`) mirrors
`ems.engine.simulate`'s H4 branch exactly (last closed H4, `EMA_fast > EMA_slow`,
applied after the H1 trend filter and before the SL anchor). `replay()` now accepts an
h4 frame and a new parity test asserts live-V3 == engine-V3 byte-identical. The
single-source-of-truth invariant (live can't drift from backtest) now covers V3.

---

## 2026-06-05

**Cost model status: only flat trading fees tested; funding/swap is unmodeled**
The only cost ever applied to EMS is a flat 0.20% round-trip (fees/spread) in the
robustness check, where EV survived (net 0.65–0.70 R/trade). Perp funding/swap has
never been modeled and is duration-dependent — EMS holds long (avg ~33h, max
290–362h), so the long-hold tail could lose meaningful R to funding. Decision: treat
funding as the next cost test; exported V3 trade CSVs with exact Madrid open/close
timestamps + `sl_pct` so any cost% converts to R, ready to join a HL funding series.

**EMS V3 H4-EMA period: locked at 100 (robustness-validated)**
2-value robustness check (not a sweep) on the coded V3 confluence filter
(H4 EMA fast=20 > slow), slow period 50 vs 100, on cached Binance BTCUSDT.
Result ROBUST: EV gap 4.2%, EV-minus-top5 gap 11.6%, DD-duration gap 11.4%,
trade overlap 78.7% — all inside thresholds (≤15% gaps, ≥70% overlap). The period
is cosmetic, not structural; the filter is a real edge. Locked **slow=100** (shorter
DD duration 62 vs 70; also marginally better EV/PF/total R/Sortino). EV survives a
0.20% round-trip cost (net 0.65–0.70 R/trade). Known caveat (both periods): edge is
fat-tail/convexity driven — EV-minus-top10 ≈ 0.06–0.09 R, skew ~6.7, payoff ~8.7.
Harness: `scripts/robustness_h4_v3.py`; recap: `docs/EMS_V3_H4_robustness_recap.md`.

---

## 2026-06-04

**Exchange is the source of truth; local state reconciles on every boot**
The bot keeps a local JSON `PositionState`, but on each start it reconciles against
the live exchange position via a 4-case matrix (flat/flat, inpos/inpos resume,
inpos/exch-flat = closed-while-down, flat/exch-pos = unexpected). The exchange wins;
local state only supplies the SL/signal context the exchange does not store. Prevents
the bot acting on a stale view after a crash or a stop that fired while it was down.

**Stop is a reduce-only trigger resting on the exchange, not bot-monitored**
The structural stop is placed as a reduce-only stop-market order on Hyperliquid at
entry time. It survives bot downtime — protection does not depend on the process
being alive. The H1 EMA100 exit, by contrast, is computed and so does require the bot
to be running at `:30` bars (catch-up on restart is a parked refinement).

**Stop-confirmed-or-flatten invariant**
A fill without a working stop is the worst state. If `place_stop()` throws after a
market entry fills, the runner immediately `market_close()`s. The bot never holds an
unprotected position by design.

**Hyperliquid price rounding: 5 significant figures (caught live on testnet)**
HL perp prices must have ≤5 significant figures AND ≤(6−szDecimals) decimals. First
testnet stop used 6 sig figs (63522.9) and was rejected "Invalid TP/SL price". Fixed
`round_px()` to the sig-fig rule (BTC ~65k → integer prices). Validates the decision
to debug on testnet before risking real money.

**Unified Account: tradable equity = perp + spot collateral**
The testnet account has Unified Account enabled, so spot USDC backs perp trades and
the perp `marginSummary.accountValue` reads 0. No spot→perp transfer is needed (and
the agent cannot do one anyway — fund moves are master-signed). `account_value()`
sums perp + spot USDC to report true tradable equity.

**Agent key is trade-only — confirmed empirically**
On testnet the agent successfully set leverage (a trading action) but was rejected on
`usd_class_transfer` (a fund move, keyed to the agent's own address). Confirms the
safety claim: a leaked/buggy agent key can trade the account but cannot move or
withdraw funds. Max loss from any bot bug is capped at the account balance.

---

## 2026-06-02

**EMS V2 chosen as the live-trading strategy (not V3)**
Live Hyperliquid bot is built on EMS V2, long-only. V3's H4 filter stays a backtest
option only for now. Keeps the first live system simpler.

**Live bot venue + execution: Hyperliquid BTC perp, testnet first**
Trade BTC perp on Hyperliquid. Always start on testnet, flip to mainnet via a single
config flag only after a clean testnet lifecycle + a mainnet dry-run. Render worker
for 24/7 hosting (matches existing API platform).

**Hybrid data model: Binance signals, Hyperliquid-adapted SL**
Entry, H1 trend filter, and H1 EMA100 exit are computed on Binance candles (what V2
was backtested on). The structural SL is identified on Binance (anchor bar), but the
stop PRICE is read from Hyperliquid candles over the same anchor→crossover timestamp
range. Reason: the bot executes on Hyperliquid, whose lows differ from Binance by a
small basis (measured ~+38 USD on mainnet); placing the stop at the Binance low would
be the wrong price on the venue actually traded.

**Sizing: fixed $ risk per trade (hot-adjustable)**
`size = risk_usd / (entry − sl)`. Flat dollar risk per trade, changeable anytime via
config. Chosen over %-equity for simplicity in the first live version.

**One source of truth for entry/exit rules (decider.py + parity test)**
Live predicates and the backtest must never drift. `ems_live.decider.replay()`
reproduces `ems.engine.simulate()` trade-for-trade; `tests/test_live_parity.py`
enforces it (synthetic seeds + a real 171-trade Binance slice, byte-identical). Any
future rule change must keep this test green.

**Safety architecture for live orders (to build before mainnet)**
API/agent wallet cannot withdraw funds (protocol-level) — worst-case bug caps loss at
the HL account balance, never the external wallet. On top of that: hard guards
(`assert sl < entry`, sane risk band, absolute `max_notional` ceiling, leverage cap +
isolated margin, stop-confirmed-or-flatten, one-position-only, max-daily-loss kill
switch) and a mainnet dry-run (orders logged, not sent) before any real-size trading.

**Testnet faucet is gated behind a mainnet deposit**
Hyperliquid's testnet faucet only pays addresses with prior mainnet deposit history
(anti-bot). To validate on testnet we must first make a small (~5–10 USDC) real
mainnet deposit on the same address, which is recoverable. Accepted this cost to keep
the testnet-first safety path.

---

## 2026-05-13

**EMS V2: adopted Samuel's canonical rules as the authoritative spec**
Six engine changes to align with `RULES_EMS_AND_H4.md`: 500-bar warmup, unlimited SL lookback, H1 exit fires only at `:30` M30 bars (second half of H1 window), gap-down SL always fills at `sl_price` with `r_multiple=-1.00`, exit reason labels changed to `STRUCTURAL_SL`/`H1_EMA100`, NaN guards added. Samuel's rules are now the ground truth; any future engine logic questions defer to that doc.

**EMS V3: H4 EMA20/50 confluence filter added as opt-in flag**
H4 filter enabled via `--h4-filter` CLI flag, `cfg.h4_filter=True`. H4 built from H1 resample (not fetched separately — avoids extra API calls and stays consistent with H1 data). Lookup formula `(entry_ts - 4h).floor('4h')` confirmed against Samuel's worked examples. Results show PF improves from 2.20→2.48 (Binance) and 2.63→2.94 (Bitstamp) at cost of ~51% fewer trades. V3 ships as an opt-in, not the default.

**Output schema: Samuel's 11-column Quantprove format**
Replaced 4-column R-only CSV with 11-column schema: `trade_id, strategy, date, time, pair, direction, result, rr, duration, sl_size, exit_reason`. This matches Samuel's journal format for direct import/comparison. Price columns (entry_price, sl_price, exit_price) intentionally excluded from CSV — R-multiple is the only metric that matters for journal review.

**SL lookback cap of 20 bars is equivalent to unlimited on this dataset**
Tested LB20 vs LB∞ in-memory — results identical (976/978 trades, same PF). Every crossover in 2017–2026 BTC data finds its structural SL anchor within 20 bars. Decision: keep `lookback=None` (unlimited) as the engine default. The 20-bar cap adds no filtering value and might incorrectly block valid setups on other assets or timeframes.

---

## 2026-05-14

**Bitstamp API paginates backwards, not forwards**
Bitstamp OHLC endpoint ignores `start` param and returns 1000 bars ending at `end`. Forward pagination (advance `start`) fails -- always returns same newest 1000 bars. Fix: walk `end` backwards (set `end = first_bar_timestamp - 1`) until full date range is covered. Sort ascending after accumulation.

**CLI extended with `--exchange` flag instead of separate script**
Added `--exchange binance|bitstamp` to existing `cli.py` rather than a separate entry point. Dispatch to `fetch_ohlcv` or `fetch_ohlcv_bitstamp` based on flag. Keeps one command surface, auto-sets default symbol per exchange.

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
