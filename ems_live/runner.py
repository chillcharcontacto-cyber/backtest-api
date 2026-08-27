"""
Live runner — orchestrates the bar-close loop.

Flow per closed M30 bar:
  FLAT        -> check_entry on latest bar; if signal: adapt SL to HL, size it,
                 run safety guards, then place (or, in dry_run, just log)
  IN_POSITION -> SL is a resting stop on the exchange (placed at entry).
                 At :30 bars, evaluate the H1 EMA100 exit; if triggered, close.

Sizing: size = risk_usd / (entry - sl_hl)   (fixed-$ risk).

SAFETY: every order path goes through guard_order(). In dry_run (default) NO order
methods are called at all — the bot computes and logs the intended action. Phase 3
implements broker order methods and flips dry_run=False on testnet first.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ems.indicators import (add_emas, add_h1_emas, mark_crossovers,
                            build_h4, add_h4_emas)
from .config import LiveConfig
from .decider import build_ctx, check_entry, check_entry_live, check_h1_exit, check_sl_hit
from .feed import fetch_binance_recent
from .sl_adapter import adapt_sl_to_hl
from .position import (PositionState, PositionStore, reconcile,
                       FLAT, IN_POSITION)
from .daylimit import DayLedgerStore, record as ledger_record, is_halted
from .nethttp import is_rate_limit
from . import notify


# --------------------------------------------------------------------------- #
#  Sizing + guards                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Order:
    side:    str      # "BUY" | "SELL"
    size:    float    # coin units
    kind:    str      # "MARKET_ENTRY" | "STOP" | "MARKET_CLOSE"
    price:   Optional[float] = None   # trigger for STOP


@dataclass
class TradePlan:
    size:     float
    leverage: int
    notional: float
    feasible: bool
    resized:  bool
    reason:   str


def compute_size(entry: float, sl: float, risk_usd: float) -> float:
    """Fixed-$ risk position size in coin units."""
    risk_per_unit = entry - sl
    if risk_per_unit <= 0:
        raise ValueError(f"non-positive risk per unit: entry={entry} sl={sl}")
    return risk_usd / risk_per_unit


def plan_trade(entry: float, sl: float, risk_usd: float, equity: float,
               cfg: LiveConfig, coin_max_lev: int, maint_frac: float) -> TradePlan:
    """
    Equity-relative sizing/leverage. Size is ALWAYS risk_usd/(entry-sl) — dollar
    risk is fixed. Leverage is the SMALLEST that fits margin, capped so the
    liquidation price stays beyond the structural stop (never over-leverage).

    liquidation distance (long, isolated) = 1/lev - maint_frac; we require it to
    exceed stop_dist * liq_safety_mult, so:
        lev <= 1 / (stop_pct_frac*liq_safety_mult + maint_frac)   (lev_liqsafe)
    margin fit: lev >= ceil(notional / (equity*margin_buffer_frac))   (lev_fit)

    If lev_fit > min(lev_liqsafe, coin_max): the account can't take full size
    safely -> resize DOWN (reduced risk, flagged) rather than over-leverage or
    silently drop.
    """
    import math
    risk_per_unit = entry - sl
    if risk_per_unit <= 0 or entry <= 0:
        return TradePlan(0.0, 1, 0.0, False, False, "SL not below entry")

    size = risk_usd / risk_per_unit
    notional = size * entry
    stop_pct = risk_per_unit / entry
    usable = max(equity, 0.0) * cfg.margin_buffer_frac

    max_lev = coin_max_lev if cfg.max_leverage <= 0 else min(cfg.max_leverage, coin_max_lev)
    max_lev = max(1, max_lev)

    denom = stop_pct * cfg.liq_safety_mult + maint_frac
    lev_liqsafe = int(math.floor(1.0 / denom)) if denom > 0 else max_lev
    if lev_liqsafe < 1:
        # even 1x can't keep liquidation the required margin beyond this (very wide)
        # stop — refuse rather than silently ship an under-buffered trade.
        return TradePlan(0.0, 1, 0.0, False, False,
                         f"stop too wide for a liquidation-safe entry ({stop_pct*100:.1f}%)")
    hard_max = max(1, min(max_lev, lev_liqsafe))

    lev_fit = int(math.ceil(notional / usable)) if usable > 0 else 10**9
    lev_fit = max(1, lev_fit)

    if lev_fit <= hard_max:
        return TradePlan(size, min(max(lev_fit, 1), hard_max), notional, True, False, "ok")

    # Cannot fit full risk while staying liquidation-safe -> resize down.
    max_notional = usable * hard_max
    size2 = max_notional / entry if entry > 0 else 0.0
    feasible = size2 > 0
    return TradePlan(size2, hard_max, size2 * entry, feasible, True,
                     "resized down to fit margin while keeping liquidation beyond the stop")


def guard_order(entry: float, sl: float, size: float, cfg: LiveConfig):
    """
    Last-line sanity guards (band + min/optional-max notional + positive size).
    Margin/leverage affordability is handled by plan_trade(); NO fixed-$ notional
    ceiling here unless cfg.max_notional_usd > 0 (off by default).
    """
    if not (sl < entry):
        return False, f"SL not below entry (sl={sl}, entry={entry})"

    risk_pct = (entry - sl) / entry * 100.0
    if risk_pct < cfg.min_risk_pct:
        return False, f"stop too tight: {risk_pct:.3f}% < {cfg.min_risk_pct}%"
    if risk_pct > cfg.max_risk_band_pct:
        return False, f"stop too wide: {risk_pct:.3f}% > {cfg.max_risk_band_pct}%"

    if size <= 0:
        return False, f"non-positive size {size}"

    notional = size * entry
    if notional < cfg.hl_min_notional_usd:
        return False, f"notional ${notional:.2f} below HL min ${cfg.hl_min_notional_usd}"
    if cfg.max_notional_usd > 0 and notional > cfg.max_notional_usd:
        return False, f"notional {notional:.2f} exceeds ceiling {cfg.max_notional_usd}"

    return True, "ok"


# --------------------------------------------------------------------------- #
#  Data load                                                                   #
# --------------------------------------------------------------------------- #

def load_signal_frames(cfg: LiveConfig):
    """
    Fetch + indicator-decorate the Binance signal frames.
    Returns (m30, h1, h4). h4 is None unless cfg.h4_filter (V3), in which case it
    is built from H1 with EMA(fast)/EMA(slow) — same construction as the backtest.
    """
    m30 = fetch_binance_recent(cfg.binance_symbol, "30m", cfg.lookback_m30)
    h1  = fetch_binance_recent(cfg.binance_symbol, "1h",  cfg.lookback_h1)
    m30 = mark_crossovers(add_emas(m30, cfg.ema_fast, cfg.ema_slow))
    h1d = add_h1_emas(h1, cfg.h1_trend_ema, cfg.h1_exit_ema)

    h4 = None
    if cfg.h4_filter:
        h4 = add_h4_emas(build_h4(h1), cfg.h4_ema_fast, cfg.h4_ema_slow)
    return m30, h1d, h4


# --------------------------------------------------------------------------- #
#  One tick (testable)                                                         #
# --------------------------------------------------------------------------- #

def tick(cfg: LiveConfig, store: PositionStore, broker, log=print) -> PositionState:
    """
    Evaluate the most recent closed bar once and act. Returns the new state.
    `broker` may be None in pure dry_run flows that never touch the exchange.
    """
    state = store.load()
    m30, h1, h4 = load_signal_frames(cfg)
    ctx = build_ctx(m30, h1, h4)
    i = len(m30) - 1
    t = ctx.m30_times[i]
    today = str(t.date())
    log(f"[tick] bar={t}  status={state.status}")

    # ---------------- IN POSITION ----------------
    if state.status == IN_POSITION:
        # 1) Did the resting stop fire? Exchange is the source of truth.
        if not cfg.dry_run and broker is not None:
            if broker.open_position() is None:
                log("  position gone on exchange (resting stop fired) — recording -1R")
                _notify_exit(cfg, broker, state, "STRUCTURAL_SL",
                             state.sl_price, -1.00, t, today, log)
                return store_flat(store)
        else:
            # dry: simulate intrabar stop hit on the bar low
            if check_sl_hit(ctx, i, state.sl_price):
                log("  DRY_RUN: simulated SL hit -> -1R")
                _notify_exit(cfg, broker, state, "STRUCTURAL_SL",
                             state.sl_price, -1.00, t, today, log)
                return store_flat(store)

        # 2) H1 EMA100 exit
        ex = check_h1_exit(ctx, i, state.entry_price, state.sl_price)
        if ex is None:
            log("  in position — no exit this bar (stop resting on exchange)")
            # H1-close update (at :30 M30 bars, when the H1 just closed)
            _notify_inpos(cfg, ctx, i, state, t, log)
            return state
        log(f"  H1_EMA100 exit signal @ {ex.exit_price:.2f} (r={ex.r_multiple})")
        if cfg.dry_run:
            log("  DRY_RUN: would market_close + cancel resting stop")
        else:
            broker.market_close()
            if state.stop_oid and state.stop_oid > 0:
                try:
                    broker.cancel_order(state.stop_oid)
                except Exception as e:
                    log(f"  (stop cancel after close failed, likely already gone: {e!r})")
        _notify_exit(cfg, broker, state, "H1_EMA100", ex.exit_price,
                     ex.r_multiple, ex.exit_time, today, log)
        return store_flat(store)

    # ---------------- FLAT: catch up a deferred entry, else check for a new one ----
    # First, if a prior bar's entry was DEFERRED by a rate-limit, retry it now (as long
    # as it's still fresh + valid). This is the missed-entry catch-up.
    caught = _try_pending_entry(cfg, store, broker, ctx, i, today, log)
    if caught is not None:
        return caught

    # LIVE timing: crossover on the JUST-CLOSED bar i -> enter at market NOW
    # (~ backtest open[i+1]), not one bar late.
    sig = check_entry_live(ctx, i, cfg, ctx.thirty_min)
    if sig is None:
        log("  flat — no entry signal")
        _notify_status(cfg, ctx, i, m30, t, log)   # market follow-up while flat
        return state

    result = _attempt_entry(cfg, store, broker, t, sig.anchor_time, sig.crossover_time,
                            sig.sl_price, sig.entry_price, today, log)
    return result if result is not None else state


def store_flat(store: PositionStore) -> PositionState:
    s = PositionState()
    store.save(s)
    return s


# --------------------------------------------------------------------------- #
#  Entry execution + missed-entry catch-up                                     #
# --------------------------------------------------------------------------- #

CATCHUP_MAX_BARS = 3     # retry a deferred entry while its crossover is <= this many M30 bars old


def _pending_path(cfg) -> str:
    return cfg.state_path + ".pending"


def _save_pending(cfg, anchor_time, crossover_time, sl_binance, bar_t):
    data = {"anchor_time": str(anchor_time), "crossover_time": str(crossover_time),
            "sl_binance": float(sl_binance), "bar_time": str(bar_t)}
    try:
        p = _pending_path(cfg)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(p + ".tmp", p)
    except Exception:
        pass


def _load_pending(cfg):
    p = _pending_path(cfg)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _clear_pending(cfg):
    try:
        p = _pending_path(cfg)
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass


def _defer_or_abort(cfg, exc, stage, anchor_time, crossover_time, sl_binance, t, log):
    """A transient rate-limit -> save a pending entry so the next bar retries it. Any
    other (permanent) error -> clear pending and abort with a loud card."""
    if is_rate_limit(exc):
        _save_pending(cfg, anchor_time, crossover_time, sl_binance, t)
        log(f"  ENTRY DEFERRED — {stage} rate-limited; will retry next bar")
        return None
    _clear_pending(cfg)
    log(f"  ENTRY ABORTED — {stage}: {exc!r}")
    notify.send_telegram(notify.fmt_blocked("ENTRY ABORTED", f"{stage}: {exc!r}"))
    return None


def _try_pending_entry(cfg, store, broker, ctx, i, today, log):
    """If a previously-deferred entry is pending AND still fresh, retry it now. Validity
    (price still above the stop) is enforced downstream by guard_order in _attempt_entry;
    a broken setup clears the pending. Returns the new state on a fill, else None."""
    pend = _load_pending(cfg)
    if not pend:
        return None
    t = ctx.m30_times[i]
    try:
        age = int((t - pd.Timestamp(pend["crossover_time"])) / ctx.thirty_min)
    except Exception:
        age = 10**6
    if age < 0 or age > CATCHUP_MAX_BARS:
        log(f"  pending entry stale (age {age} bars) — dropping")
        _clear_pending(cfg)
        return None
    log(f"  CATCH-UP: retrying deferred entry (crossover {pend['crossover_time']}, age {age} bars)")
    return _attempt_entry(cfg, store, broker, t, pd.Timestamp(pend["anchor_time"]),
                          pd.Timestamp(pend["crossover_time"]), pend["sl_binance"],
                          ctx.m30_closes[i], today, log)


def _attempt_entry(cfg, store, broker, t, anchor_time, crossover_time, sl_binance,
                   entry_est, today, log):
    """
    Execute an entry for a detected signal, or DEFER it on a transient rate-limit so the
    next bar retries (missed-entry catch-up). Returns the new IN_POSITION state on a live
    fill, else None. Permanent rejections (kill switch / infeasible / guard / dry-run)
    clear any pending; only rate-limited HL failures defer. A 429 on the ORDER is safe to
    retry (rejected before matching); an ambiguous non-429 order error aborts (no double
    fill). Faithful to the original inline entry path — plus the defer/catch-up.
    """
    if _halted(cfg, today):
        _clear_pending(cfg)
        log(f"  KILL SWITCH active — no entries today ({today})")
        notify.send_telegram(notify.fmt_blocked("KILL SWITCH", f"daily loss limit hit, halted today ({today})"))
        return None

    # adapt SL to Hyperliquid candles over the Binance anchor range
    try:
        sl_hl = adapt_sl_to_hl(anchor_time, crossover_time, cfg.coin, cfg.hl_api_url)
    except Exception as e:
        return _defer_or_abort(cfg, e, "SL adaptation", anchor_time, crossover_time, sl_binance, t, log)

    entry = entry_est
    if broker is not None:
        try:
            entry = broker.mid_price()
        except Exception:
            pass   # fall back to the estimate

    equity = None
    coin_max = cfg.max_leverage or 40
    maint = 0.0125
    if broker is not None:
        coin_max = broker.coin_max_leverage
        maint = broker.maint_margin_frac
        try:
            equity = broker.account_value()
        except Exception as e:
            log(f"  equity read failed ({e!r})")
            if not cfg.dry_run:
                return _defer_or_abort(cfg, e, "equity read", anchor_time, crossover_time, sl_binance, t, log)

    if equity is not None and equity > 0:
        plan = plan_trade(entry, sl_hl, cfg.risk_usd, equity, cfg, coin_max, maint)
    elif not cfg.dry_run:
        _clear_pending(cfg)
        log("  ENTRY ABORTED — equity unavailable, cannot size safely")
        notify.send_telegram(notify.fmt_blocked("ENTRY ABORTED", "equity read unavailable — skipping to avoid unsafe leverage"))
        return None
    else:
        try:
            sz = compute_size(entry, sl_hl, cfg.risk_usd)
        except ValueError as e:
            _clear_pending(cfg)
            log(f"  ENTRY ABORTED — sizing error: {e}")
            notify.send_telegram(notify.fmt_blocked("ENTRY ABORTED", f"sizing error: {e}"))
            return None
        plan = TradePlan(sz, cfg.leverage, sz * entry, True, False, "dry-run no-equity fallback")

    if not plan.feasible:
        _clear_pending(cfg)
        log(f"  ENTRY ABORTED — {plan.reason}")
        notify.send_telegram(notify.fmt_blocked("ENTRY ABORTED", plan.reason))
        return None

    ok, reason = guard_order(entry, sl_hl, plan.size, cfg)
    if not ok:
        _clear_pending(cfg)
        log(f"  ENTRY REFUSED by guard: {reason}")
        notify.send_telegram(notify.fmt_blocked("ENTRY REFUSED", reason))
        return None

    if plan.resized:
        note = (f"account can't carry full risk with a liq-safe stop; sized down to "
                f"{plan.size:.6f} BTC (lev {plan.leverage}x)")
        log(f"  RISK RESIZED — {note}")
        notify.send_telegram(notify.fmt_blocked("RISK RESIZED", note))

    log(f"  ENTRY SIGNAL entry~{entry:.2f} hl_sl={sl_hl:.2f} size={plan.size:.6f} "
        f"notional={plan.notional:.2f} lev={plan.leverage}x resized={plan.resized}")

    if cfg.dry_run:
        _clear_pending(cfg)
        log("  DRY_RUN: would update_leverage + market_entry + place_stop")
        notify.send_telegram("🟡 DRY_RUN entry signal\n"
                             + notify.fmt_entry(t, entry, sl_hl, plan.size, cfg.risk_usd, plan.leverage))
        return None

    # --- LIVE ---
    try:
        broker.update_leverage(plan.leverage)
    except Exception as e:
        return _defer_or_abort(cfg, e, "leverage set", anchor_time, crossover_time, sl_binance, t, log)

    try:
        fill = broker.market_entry(plan.size)
    except Exception as e:
        return _defer_or_abort(cfg, e, "market_entry", anchor_time, crossover_time, sl_binance, t, log)

    _clear_pending(cfg)   # committed — a position is opening; never retry this signal
    actual_entry = fill["avg_px"]
    actual_size = fill["filled"]
    log(f"  FILLED entry @ {actual_entry:.2f} sz={actual_size}")

    new = PositionState(
        status=IN_POSITION,
        entry_time=str(t), entry_price=actual_entry, sl_price=sl_hl, size=actual_size,
        anchor_time=str(anchor_time), crossover_time=str(crossover_time),
        stop_oid=None,
    )
    store.save(new)

    stop_oid = None
    for attempt in range(3):
        try:
            stop_oid = broker.place_stop(sl_hl, actual_size)
            break
        except Exception as e:
            log(f"  place_stop attempt {attempt + 1}/3 failed: {e!r}")

    if stop_oid == -1:
        log("  stop triggered on placement (price already through SL) — recording -1R")
        _notify_exit(cfg, broker, new, "STRUCTURAL_SL", sl_hl, -1.00, t, today, log)
        return store_flat(store)

    if stop_oid is None:
        log("  STOP PLACEMENT FAILED after retries — flattening")
        try:
            broker.market_close()
            notify.send_telegram(notify.fmt_blocked("STOP FAILED → FLATTENED", "could not place stop"))
            return store_flat(store)
        except Exception as e:
            log(f"  FLATTEN ALSO FAILED ({e!r}) — position OPEN & UNPROTECTED, keeping state")
            notify.send_telegram(notify.fmt_blocked(
                "⛔ UNPROTECTED POSITION",
                "stop AND flatten both failed — CHECK MANUALLY. Bot keeps managing it."))
            return new

    new.stop_oid = stop_oid
    store.save(new)
    log(f"  POSITION OPEN entry={actual_entry:.2f} stop={sl_hl:.2f} sz={actual_size} "
        f"lev={plan.leverage}x oid={stop_oid}")
    notify.send_telegram(notify.fmt_entry(t, actual_entry, sl_hl, actual_size,
                                          cfg.risk_usd, plan.leverage))
    return new


def _note_rate_limit(cfg, log):
    """Throttle 429/5xx tick alerts: send ONE Telegram card per UTC day + keep a running
    daily count in a sidecar (so a worsening pattern is still visible without per-blip
    spam). These are harmless and self-healing (retried in-tick; entries catch up)."""
    p = cfg.state_path + ".429"
    today = str(pd.Timestamp.utcnow().date())
    data = {"day": today, "count": 0}
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("day") == today:
                data = prev
    except Exception:
        pass
    data["day"] = today
    data["count"] = int(data.get("count", 0)) + 1
    try:
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(p + ".tmp", p)
    except Exception:
        pass
    log(f"  [rate-limit] 429/5xx #{data['count']} today (throttled)")
    if data["count"] == 1:
        notify.send_telegram(
            "⏳ Hyperliquid rate-limit (429) today — harmless. The bot rides it out and "
            "catches up any missed entry. Further 429 cards suppressed today (count in the log).")


# --------------------------------------------------------------------------- #
#  Daily-loss kill switch helpers                                              #
# --------------------------------------------------------------------------- #

def _ledger_store(cfg: LiveConfig) -> DayLedgerStore:
    return DayLedgerStore(cfg.state_path + ".day")


def _record(cfg: LiveConfig, r: float, today: str, log=print):
    s = _ledger_store(cfg)
    led = ledger_record(s.load(), r, today)
    s.save(led)
    log(f"  day P&L: {led.realized_r:+.2f}R  {led.losses} loss / {led.trades} trade(s)  day={led.day}")
    if is_halted(led, today, cfg.max_daily_loss_r, cfg.max_daily_losses):
        log(f"  >>> DAILY LOSS LIMIT HIT ({led.losses} losses, {led.realized_r:+.2f}R) "
            f"— entries halted today")
    return led


def _halted(cfg: LiveConfig, today: str) -> bool:
    return is_halted(_ledger_store(cfg).load(), today,
                     cfg.max_daily_loss_r, cfg.max_daily_losses)


# --------------------------------------------------------------------------- #
#  Telegram notifications                                                      #
# --------------------------------------------------------------------------- #

def _notify_exit(cfg, broker, state, reason, exit_px, r, exit_time, today, log):
    """Record the closed trade in the day-ledger and push a Telegram EXIT card.

    Deviation = how far the realized (after-fee) R lands from the model R, as a
    percent of |model_r|. With fixed-$ risk, gross $ = model_r * risk_usd exactly,
    so the only drift here is fees (slippage on real fills can be added later from
    exchange data). Negative deviation = costs ate into the result.
    """
    led = _record(cfg, r, today, log)            # ledger keeps model R
    size = state.size or 0.0
    entry = state.entry_price or 0.0
    sl = state.sl_price or 0.0
    fees = cfg.taker_fee * (size * entry + size * exit_px)

    # $ P&L off the ACTUAL filled size (a partial fill risks < risk_usd), so 1R here
    # is the real dollar risk of the position, not the nominal risk_usd.
    actual_risk = size * (entry - sl) if (entry > sl and size > 0) else cfg.risk_usd
    model_r = r
    gross_pnl = model_r * actual_risk
    net_pnl = gross_pnl - fees
    net_r = net_pnl / actual_risk if actual_risk else 0.0
    deviation = ((net_r - model_r) / abs(model_r) * 100.0) if model_r else 0.0

    pct = None
    if broker is not None:
        try:
            av = broker.account_value()
            if av and av > 0:
                pct = net_pnl / av * 100.0
        except Exception:
            pass
    try:
        dur_h = (pd.Timestamp(exit_time) - pd.Timestamp(state.entry_time)).total_seconds() / 3600
    except Exception:
        dur_h = 0.0

    notify.send_telegram(notify.fmt_exit(
        reason, exit_px, model_r, net_r, net_pnl, pct, fees, deviation, dur_h,
        led.realized_r))
    return led


def _gather(ctx, i):
    """EMA/price snapshot at bar i for the status + in-position cards. None if gaps."""
    t = ctx.m30_times[i]
    h1_open = t.floor("h") - ctx.one_hour
    h1i = ctx.h1_time_idx.get(h1_open)
    h4l = (t - ctx.four_hours).floor("4h")
    h4i = ctx.h4_time_idx.get(h4l) if ctx.h4_time_idx is not None else None
    if h1i is None or h4i is None:
        return None
    return {
        "h1_close": ctx.h1_closes[h1i],
        "h1_e50":   ctx.h1_ema_trend[h1i],
        "h1_e100":  ctx.h1_ema_exit[h1i],
        "h4_e20":   ctx.h4_ema_fast[h4i],
        "h4_e100":  ctx.h4_ema_slow[h4i],
    }


def _status_store_path(cfg) -> str:
    return cfg.state_path + ".status"


def _load_status(cfg) -> dict:
    p = _status_store_path(cfg)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_status(cfg, data: dict):
    p = _status_store_path(cfg)
    try:
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(p + ".tmp", p)
    except Exception:
        pass


def _notify_status(cfg, ctx, i, m30, t, log):
    """
    Flat-state notifications.

    status_mode='steps' (default): top-down ladder H4 -> H1 -> M30. A step message
    is sent only when the setup STAGE changes (so lower-TF noise stays silent while a
    higher TF blocks), plus one daily 'still alive' heartbeat. Stage:
      0 H4 bearish (blocked) · 1 H4 ok/waiting H1 · 2 armed/waiting M30 cross.
    status_mode='change'/'always': legacy per-flip / per-tick FLAT card.
    status_mode='off': nothing.
    """
    mode = getattr(cfg, "status_mode", "steps")
    if mode == "off":
        return
    snap = _gather(ctx, i)
    if snap is None:
        return
    try:
        m30_e20 = float(m30["ema_fast"].iloc[-1])
        m30_e50 = float(m30["ema_slow"].iloc[-1])
        cross = bool(ctx.m30_cross[i - 1])
    except Exception:
        return

    h4 = snap["h4_e20"] > snap["h4_e100"]
    h1 = snap["h1_close"] > snap["h1_e50"]
    m30b = m30_e20 > m30_e50
    today = str(t.date())
    data = _load_status(cfg)

    if mode == "steps":
        stage = notify.stage_of(h4, h1)
        send_hb, send_stage = notify.steps_plan(
            data.get("stage"), data.get("hb_day"), stage, today)
        _save_status(cfg, {**data, "stage": stage, "hb_day": today})
        if send_hb:
            notify.send_telegram(notify.fmt_heartbeat(h4, h1, m30b))
        if send_stage:
            notify.send_telegram(notify.fmt_stage(stage, h4, h1, m30b))
        return

    # legacy change / always
    gate = {"m30": int(m30b), "h1": int(h1), "h4": int(h4)}
    last = data.get("gate") if mode == "change" else None
    send, prefix = notify.status_decision(last, gate, mode)
    if mode == "change":
        _save_status(cfg, {**data, "gate": gate})
    if not send:
        return
    notify.send_telegram(prefix + notify.fmt_status(
        t, m30_e20, m30_e50, cross, snap["h1_close"], snap["h1_e50"],
        snap["h4_e20"], snap["h4_e100"]))


def _notify_inpos(cfg, ctx, i, state, t, log):
    """In-position update on each H1 close (:30 M30 bars) — distance to exit trigger."""
    if t.minute != 30:
        return
    h1_open = t - ctx.thirty_min
    idx = ctx.h1_time_idx.get(h1_open)
    if idx is None:
        return
    h1_close = ctx.h1_closes[idx]
    h1_e100 = ctx.h1_ema_exit[idx]
    entry = state.entry_price or 0.0
    sl = state.sl_price or 0.0
    denom = entry - sl
    r_now = (h1_close - entry) / denom if denom else 0.0
    notify.send_telegram(notify.fmt_inpos(t, h1_close, entry, sl, r_now, h1_e100))


# --------------------------------------------------------------------------- #
#  Boot reconcile                                                              #
# --------------------------------------------------------------------------- #

def boot_reconcile(cfg: LiveConfig, store: PositionStore, broker, log=print) -> PositionState:
    # set leverage + isolated margin once (live only)
    if broker is not None and not cfg.dry_run:
        try:
            broker.set_margin_mode()
            log(f"[boot] margin mode set: {cfg.leverage}x "
                f"{'isolated' if cfg.isolated_margin else 'cross'}")
        except Exception as e:
            log(f"[boot] set_margin_mode failed: {e!r}")

    local = store.load()
    exch = broker.open_position() if broker else None
    new, events = reconcile(local, exch)
    for ev in events:
        log(f"[reconcile] {ev.kind}: {ev.detail}  {ev.payload or ''}")
    store.save(new)
    return new


# --------------------------------------------------------------------------- #
#  Scheduler                                                                   #
# --------------------------------------------------------------------------- #

def seconds_until_next_bar(now_utc: pd.Timestamp, buffer_sec: int) -> float:
    """Seconds from now until the next M30 boundary (:00/:30) + buffer."""
    minute = now_utc.minute
    next_min = 30 if minute < 30 else 60
    next_boundary = now_utc.floor("h") + pd.Timedelta(minutes=next_min)
    delta = (next_boundary - now_utc).total_seconds() + buffer_sec
    return max(delta, 1.0)


def run_forever(cfg: LiveConfig, store: PositionStore, broker, log=print):
    """Long-running loop for the Render worker. Reconciles, then ticks each bar."""
    import time
    log(f"=== EMS LIVE runner start — testnet={cfg.testnet} dry_run={cfg.dry_run} ===")
    kill_bits = []
    if cfg.max_daily_losses > 0:
        kill_bits.append(f"{cfg.max_daily_losses} losses")
    if cfg.max_daily_loss_r > 0:
        kill_bits.append(f"{cfg.max_daily_loss_r}R")
    kill_s = "/".join(kill_bits) or "off"
    notify.send_telegram(
        f"🚀 EMS-V3 bot started\ntestnet={cfg.testnet}  dry_run={cfg.dry_run}\n"
        f"risk ${cfg.risk_usd}  kill {kill_s}/day  H4 {cfg.h4_ema_fast}/{cfg.h4_ema_slow}")
    boot_reconcile(cfg, store, broker, log)
    notify.ping_health()                  # liveness from boot
    while True:
        now = pd.Timestamp.utcnow()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        sleep_s = seconds_until_next_bar(now, cfg.poll_buffer_sec)
        log(f"[sleep] {sleep_s:.0f}s until next bar")

        # Sleep in heartbeat-sized chunks, pinging each chunk so healthchecks
        # knows the process is alive even between 30-min bars (hang detection).
        remaining = sleep_s
        while remaining > 0:
            chunk = min(cfg.heartbeat_sec, remaining)
            time.sleep(chunk)
            notify.ping_health()          # heartbeat (silent; no log)
            remaining -= chunk

        try:
            tick(cfg, store, broker, log)
            notify.ping_health()          # liveness: tick completed OK
        except Exception as e:   # never let one bad tick kill the worker
            # process is still alive (heartbeat continues). A rate-limit (429/5xx) is
            # harmless + self-healing (retries in-tick, catches up missed entries), so
            # it is THROTTLED to one card/day; any OTHER error stays loud (real bug).
            log(f"[ERROR] tick failed: {e!r}")
            if is_rate_limit(e):
                _note_rate_limit(cfg, log)
            else:
                notify.send_telegram(notify.fmt_blocked("TICK ERROR", repr(e)))
