# Live-Bot Ops Kit — Port Brief

**What this is.** The complete monitoring + safety system from the EMS bot (live on
Hyperliquid via a Render worker), packaged so a second bot — **yours, on Render,
trading a cTrader account** — gets the *exact same* operational behavior. This folder
ships the venue-agnostic code as drop-in `.py` files, and this brief tells you what to
copy verbatim, what to re-author for cTrader, and in what order.

**How to use it (for the receiving Claude Code session).**
1. Copy this whole `ops_kit/` folder into the new bot's repo.
2. Read this file top to bottom before writing anything.
3. Follow **§7 Build Order**. Copy the verbatim files as-is; implement the cTrader
   pieces against the contract in `broker_protocol.py`.
4. Verify against **§8 Acceptance Checks** — that's the definition of "same as EMS."

Do not invent a different structure. The whole point is parity with a system that is
already running real money.

---

## 1. The system is 5 layers, not 2

The user's mental model was "heartbeat + Telegram." That's two of five. A bot you can
leave alone with real money needs all five:

| # | Layer | What it does | Where |
|---|-------|--------------|-------|
| 1 | **Liveness** | Is the process breathing? Dead-man's switch + crash alerts | `monitor.ping_health` + Render native alerts |
| 2 | **Observability** | Trade-follow cards: start / entry / in-pos / exit / step-ladder / heartbeat | `monitor.py` formatters + Telegram |
| 3 | **Risk** | Kill switch: halt after N losing trades/day | `daylimit.py` |
| 4 | **State & recovery** | Survive restarts; reconcile local vs broker truth | `position.py` |
| 5 | **Execution safety** | Never orphan an unprotected position; size off real fills | `loop.py` + your broker adapter |

Layers 1 and 2 are what was asked for. **Layers 3–5 are the actual gap** — they're the
difference between "monitored" and "safe to leave alone."

---

## 2. Verbatim vs. adapt

| File | Port | Notes |
|------|------|-------|
| `daylimit.py` | ✅ **verbatim** | Pure R-based kill switch. Zero venue coupling. |
| `position.py` | ✅ **verbatim** | State + reconcile matrix. Only needs `broker.open_position()`. |
| `monitor.py` **senders** | ✅ **verbatim** | `send_telegram`, `ping_health`. |
| `monitor.py` **formatters** | 🟡 adapt wording | Generic templates already (symbol + price-format). Reword cards to your strategy's TFs/terms. |
| `loop.py` | ✅ **~verbatim** | `run_forever`/`boot_reconcile`/scheduler. You pass in `tick_fn`. |
| `broker_protocol.py` | 🔨 **implement** | The seam. Write ONE class satisfying it against cTrader Open API. |
| `render.worker.yaml` | 🟡 adapt keys | Rename service/env; keep disk + `PYTHONUNBUFFERED` + `sync:false`. |
| **strategy `tick_fn`** | 🔨 **your logic** | Entry/exit/sizing/orders for your strategy. The only large new write. |

**Rule:** everything except the broker adapter and your `tick_fn` is reused. If you find
yourself rewriting `daylimit`/`position`/the loop, stop — you're diverging from parity.

---

## 3. cTrader ≠ Hyperliquid — the real differences

Hyperliquid is a crypto **perp DEX**; cTrader is a **forex/CFD broker**. The monitoring,
kill switch, state, and reconcile layers don't care. Four things genuinely differ:

**a) Broker API & auth.** HL = one agent key signing REST calls. cTrader = **OAuth2 +
Open API** (protobuf over TLS; the `ctrader-open-api` Python package, Twisted-based, or
FIX). You register an app at **connect.spotware.com** → `client_id`/`client_secret` →
OAuth consent → `access_token` + `refresh_token`; the account is a `ctidTraderAccountId`.
Request the **trading** scope. **The access token expires** — the bot (or you) must
refresh it before it lapses; there is no 180-day agent like HL. All of these are Render
secrets (`sync:false`). Like HL agents, Open API trading scope can trade but not withdraw.

**b) Sizing — DROP the auto-leverage.** EMS's `plan_trade()` computes per-order leverage
and a liquidation price because HL perps expose both. **cTrader does not** — leverage is
fixed by the account/symbol and the broker runs margin-call/stop-out. So replace the whole
liquidation machinery with plain forex sizing:

```
stop_dist   = abs(entry - sl)                     # in price
volume_units = risk_ccy / (stop_dist * pip_value_per_unit_of_base)
volume       = round_to_step(volume_units, symbol.volume_step)
reject if volume < symbol.min_volume  or  margin_required(volume) > free_margin
```

Keep the *spirit* of the guards (`guard_order`): fixed-risk size, a min/max stop-width
band, a min-volume floor. Just no `lev_liqsafe`/`maint_frac`/`coin_max_leverage`. In the
adapter, make `update_leverage`/`set_margin_mode` no-ops and `coin_max_leverage`/
`maint_margin_frac` return `1`/`0.0`.

> ⚠️ Verify cTrader's **volume convention** for your symbol from the API (`ProtoOASymbol`:
> `lotSize`, `minVolume`, `stepVolume`, `pipPosition`) — do not hard-code lot math. Get
> `pip_value` right or every position is mis-sized. Sanity-check the first live volume by
> hand against a known lot value.

**c) Costs — commission + swap, not taker fee + funding.** The exit card's cost line
becomes `fees = commission_per_side*2` and `swap = overnight_financing_over_hold`
(cTrader `ProtoOAExecutionEvent` / deal history gives commission & swap per position).
Spread is already baked into the fill price. `monitor.fmt_exit` already has a `swap`
parameter — pass the real swap; that makes the card **truthful about overnight cost from
day one** (a gap EMS's own card still has).

**d) Market hours.** Forex closes weekends/holidays. The loop handles this for free — it
keeps sleeping+pinging; `tick_fn` just finds no fresh bar and returns. Don't add special
handling; do make sure your candle source returns cleanly across the gap.

---

## 4. The broker contract (the seam)

Everything venue-specific hides behind `broker_protocol.Broker`. Implement exactly these
methods against cTrader Open API — signatures and return shapes matter because the loop
and `reconcile` depend on them:

- `open_position() -> None | {"size","entry_px","unrealized_pnl"}` — from
  `ProtoOAReconcileReq`. **Source of truth for reconcile.**
- `account_value() -> float` — equity/free margin. If unreadable on the LIVE path, the
  loop **aborts the entry** rather than sizing blind. Preserve that.
- `mid_price() -> float` — latest bid/ask mid.
- `market_entry(size) -> {"avg_px","filled"}` — **return the actually filled size**; the
  loop sizes the stop and P&L off `filled`, never the request.
- `place_stop(trigger_px, size) -> int | -1 | None` — attach SL to the position
  (`ProtoOAAmendPositionSLTPReq`) or a stop order. `-1` = already through (immediate −1R),
  `None` = failed (loop flattens + guards).
- `market_close()` — `ProtoOAClosePositionReq`.
- `cancel_order(oid)` — `ProtoOACancelOrderReq`.
- `update_leverage(lev)` / `set_margin_mode()` — **no-ops** on cTrader.
- `coin_max_leverage` (→ `1`) / `maint_margin_frac` (→ `0.0`) — unused on cTrader.

The dangerous edges are already handled in the loop (see `loop.py` + EMS
`runner.py::tick`), you just have to honor the return contract:
- **State is saved BEFORE the stop is placed** → a crash after entry can't hide a live
  position.
- **Stop placement retries; if it can't rest, the loop flattens**; if the flatten *also*
  fails it keeps `IN_POSITION` and screams `⛔ UNPROTECTED POSITION` to Telegram rather
  than recording a false flat.
- **Partial fills** use `filled`, so 1R is the real dollar risk taken.

---

## 5. Kill switch, state, reconcile — reuse as-is

- **Kill switch** (`daylimit.py`): call `record(...)` on every close, `is_halted(...)`
  before every entry. EMS uses `max_daily_losses=10` (≈ −10R/day cap). The backtest had
  NO kill switch — this is a live-only overlay; keep it.
- **State** (`position.py`): one atomic JSON on the **persistent disk** (`/data/...`).
  `PositionStore.save` is an atomic replace — don't "simplify" it to a plain write.
- **Reconcile** (`position.py::reconcile`): run at **boot** (and you may run per-tick).
  The 4-case matrix (flat/flat, in/in resume, in/flat = closed-while-down, flat/in =
  unexpected) is what lets a Render redeploy mid-trade recover instead of double-entering.

---

## 6. Environment & deploy

- Use `render.worker.yaml` as the worker block. **Keep**: `type: worker`, the `/data`
  disk, `PYTHONUNBUFFERED=1`, and `sync:false` on every secret.
- `.env.example` is the local template (env must WIN over `.env` — use `setdefault`, see
  EMS `run.py::load_env`).
- **Turn on BOTH liveness legs:** healthchecks.io (`HEALTHCHECK_URL`, Period 1m / Grace
  2m to match the minute heartbeat) **and** Render Dashboard → Notifications →
  deploy-failed + crash. healthchecks catches hangs; Render catches crashes.
- **Safety ladder for going live** (do not skip): `dry_run=true` on a **demo** account →
  watch the cards for a full entry→exit cycle → `dry_run=false` on demo (real demo
  orders) → only then a small-risk **live** account. EMS went testnet-dry → testnet-armed
  → mainnet $3; mirror that.

---

## 7. Build order

1. **Copy verbatim:** `daylimit.py`, `position.py`, `monitor.py`, `loop.py`,
   `broker_protocol.py` into the bot package. Wire a config object with the fields
   `loop.py` reads (see its docstring).
2. **Broker adapter:** implement `Broker` against cTrader Open API (OAuth connect,
   reconcile, market order + read fill, attach SL, close, cancel). Unit-test
   `open_position()` returns the right shape and `market_entry()` returns real `filled`.
3. **Sizing:** forex volume-from-risk (§3b) with `ProtoOASymbol` metadata; guard band +
   min-volume + free-margin check. Hand-verify the first volume.
4. **Strategy `tick_fn`:** port your entry/exit logic. Mirror EMS `runner.py::tick`'s
   shape: reconcile-aware, kill-switch-gated, save-state-before-stop, notify on every
   branch. Costs into `fmt_exit` = commission + swap.
5. **Monitoring wording:** adapt the cards to your strategy's timeframes/terms. Keep
   senders untouched.
6. **Deploy:** worker block, disk, secrets in dashboard, both liveness legs on.
7. **Prove it** against §8 on a demo account before any live money.

---

## 8. Acceptance checks — "same as EMS" means all of these pass

- [ ] 🚀 start card fires on boot with `testnet/dry_run/risk/kill` shown correctly.
- [ ] healthchecks shows a ping at least every minute; killing the worker turns it red
      within the grace window; Render emails on a forced crash/deploy-fail.
- [ ] A full **dry-run** entry→in-pos→exit cycle posts the right Telegram cards on a
      demo account, and the exit card's net R = model R − (commission+swap)/risk.
- [ ] Kill switch: after `max_daily_losses` losing trades in a UTC day, new entries are
      refused with a 🛑 card; it resets next UTC day.
- [ ] Restart mid-trade (redeploy while a demo position is open): boot reconcile logs
      `OK_RESUME` and the bot keeps managing the *same* position — no double entry.
- [ ] Force `place_stop` to fail once: the bot flattens and posts `STOP FAILED →
      FLATTENED` (and `⛔ UNPROTECTED POSITION` only if the flatten also fails). It never
      records a false flat while a position is live.
- [ ] `account_value()` unreadable on the live path → entry is **aborted**, not sized blind.
- [ ] Partial fill → P&L and the resting stop use the **actual filled** size.

If every box is checked, the new bot has the EMS operational system, adapted to cTrader.
```
Reference implementation to read alongside this brief (in the EMS repo):
  ems_live/runner.py   — tick(), guard_order(), _notify_exit(), boot_reconcile(), run_forever()
  ems_live/broker.py   — the Hyperliquid Broker (your cTrader adapter mirrors this)
  ems_live/notify.py   — full card formatters (yours mirror these)
  ems_live/run.py      — env wiring / entrypoint
  render.yaml          — the live worker block
```
