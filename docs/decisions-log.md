# Decisions Log

A running list of architectural and product decisions, newest first.

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
