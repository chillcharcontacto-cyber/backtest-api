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
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ems.indicators import (add_emas, add_h1_emas, mark_crossovers,
                            build_h4, add_h4_emas)
from .config import LiveConfig
from .decider import build_ctx, check_entry, check_h1_exit, check_sl_hit
from .feed import fetch_binance_recent
from .sl_adapter import adapt_sl_to_hl
from .position import (PositionState, PositionStore, reconcile,
                       FLAT, IN_POSITION)
from .daylimit import DayLedgerStore, record as ledger_record, is_halted
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


def compute_size(entry: float, sl: float, risk_usd: float) -> float:
    """Fixed-$ risk position size in coin units."""
    risk_per_unit = entry - sl
    if risk_per_unit <= 0:
        raise ValueError(f"non-positive risk per unit: entry={entry} sl={sl}")
    return risk_usd / risk_per_unit


def guard_order(entry: float, sl: float, size: float, cfg: LiveConfig):
    """
    Return (ok: bool, reason: str). Refuse the trade on any violation.
    These are the last-line protections against a sizing/SL bug opening a
    runaway position.
    """
    if not (sl < entry):
        return False, f"SL not below entry (sl={sl}, entry={entry})"

    risk_pct = (entry - sl) / entry * 100.0
    if risk_pct < cfg.min_risk_pct:
        return False, f"stop too tight: {risk_pct:.3f}% < {cfg.min_risk_pct}%"
    if risk_pct > cfg.max_risk_band_pct:
        return False, f"stop too wide: {risk_pct:.3f}% > {cfg.max_risk_band_pct}%"

    notional = size * entry
    if notional > cfg.max_notional_usd:
        return False, (f"notional {notional:.2f} exceeds ceiling "
                       f"{cfg.max_notional_usd}")
    if size <= 0:
        return False, f"non-positive size {size}"

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

    # ---------------- FLAT: check entry ----------------
    sig = check_entry(ctx, i, cfg)
    if sig is None:
        log("  flat — no entry signal")
        _notify_status(cfg, ctx, i, m30, t, log)   # market follow-up while flat
        return state

    # kill switch — halt new entries once today's loss limit is hit
    if _halted(cfg, today):
        log(f"  KILL SWITCH active — daily loss limit reached, no entries today ({today})")
        notify.send_telegram(notify.fmt_blocked(
            "KILL SWITCH", f"daily loss limit hit, halted today ({today})"))
        return state

    # adapt SL to Hyperliquid candles over the Binance anchor range
    try:
        sl_hl = adapt_sl_to_hl(sig.anchor_time, sig.crossover_time,
                               cfg.coin, cfg.hl_api_url)
    except RuntimeError as e:
        log(f"  ENTRY ABORTED — SL adaptation failed: {e}")
        notify.send_telegram(notify.fmt_blocked("ENTRY ABORTED", f"SL adaptation failed: {e}"))
        return state

    entry = sig.entry_price
    try:
        size = compute_size(entry, sl_hl, cfg.risk_usd)
    except ValueError as e:
        log(f"  ENTRY ABORTED — sizing error: {e}")
        notify.send_telegram(notify.fmt_blocked("ENTRY ABORTED", f"sizing error: {e}"))
        return state

    ok, reason = guard_order(entry, sl_hl, size, cfg)
    if not ok:
        log(f"  ENTRY REFUSED by guard: {reason}")
        notify.send_telegram(notify.fmt_blocked("ENTRY REFUSED", reason))
        return state

    log(f"  ENTRY SIGNAL  entry~{entry:.2f}  binance_sl={sig.sl_price:.2f}  "
        f"hl_sl={sl_hl:.2f}  size={size:.6f}  notional={size*entry:.2f}")

    if cfg.dry_run:
        log("  DRY_RUN: would market_entry(size) then place_stop(hl_sl, size)")
        notify.send_telegram("🟡 DRY_RUN entry signal\n"
                             + notify.fmt_entry(t, entry, sl_hl, size, cfg.risk_usd, cfg.leverage))
        return state

    # --- LIVE ---
    fill = broker.market_entry(size)
    actual_entry = fill["avg_px"]
    log(f"  FILLED entry @ {actual_entry:.2f} sz={fill['filled']}")

    # stop-confirmed-or-flatten: a fill without a working stop is the worst state
    try:
        stop_oid = broker.place_stop(sl_hl, size)
    except Exception as e:
        log(f"  STOP PLACEMENT FAILED ({e!r}) — FLATTENING immediately")
        broker.market_close()
        notify.send_telegram(notify.fmt_blocked("STOP FAILED → FLATTENED", repr(e)))
        return store_flat(store)

    if stop_oid == -1:
        log("  stop triggered on placement (price already through SL) — position closed")
        notify.send_telegram(notify.fmt_blocked("STOP TRIGGERED ON PLACEMENT",
                                                "price already through SL — closed"))
        return store_flat(store)

    new = PositionState(
        status=IN_POSITION,
        entry_time=str(t), entry_price=actual_entry, sl_price=sl_hl, size=size,
        anchor_time=str(sig.anchor_time), crossover_time=str(sig.crossover_time),
        stop_oid=stop_oid,
    )
    store.save(new)
    log(f"  POSITION OPEN  entry={actual_entry:.2f} stop={sl_hl:.2f} oid={stop_oid}")
    notify.send_telegram(notify.fmt_entry(t, actual_entry, sl_hl, size,
                                          cfg.risk_usd, cfg.leverage))
    return new


def store_flat(store: PositionStore) -> PositionState:
    s = PositionState()
    store.save(s)
    return s


# --------------------------------------------------------------------------- #
#  Daily-loss kill switch helpers                                              #
# --------------------------------------------------------------------------- #

def _ledger_store(cfg: LiveConfig) -> DayLedgerStore:
    return DayLedgerStore(cfg.state_path + ".day")


def _record(cfg: LiveConfig, r: float, today: str, log=print):
    s = _ledger_store(cfg)
    led = ledger_record(s.load(), r, today)
    s.save(led)
    log(f"  day P&L: {led.realized_r:+.2f}R over {led.trades} trade(s)  day={led.day}")
    if is_halted(led, today, cfg.max_daily_loss_r):
        log(f"  >>> DAILY LOSS LIMIT HIT ({led.realized_r:+.2f}R) — entries halted today")
    return led


def _halted(cfg: LiveConfig, today: str) -> bool:
    return is_halted(_ledger_store(cfg).load(), today, cfg.max_daily_loss_r)


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
    fees = cfg.taker_fee * (size * entry + size * exit_px)

    model_r = r
    gross_pnl = model_r * cfg.risk_usd
    net_pnl = gross_pnl - fees
    net_r = net_pnl / cfg.risk_usd if cfg.risk_usd else 0.0
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


def _notify_status(cfg, ctx, i, m30, t, log):
    """Flat-state market follow-up (every tick) — EMA standings across timeframes."""
    snap = _gather(ctx, i)
    if snap is None:
        return
    try:
        m30_e20 = float(m30["ema_fast"].iloc[-1])
        m30_e50 = float(m30["ema_slow"].iloc[-1])
        cross = bool(ctx.m30_cross[i - 1])
    except Exception:
        return
    notify.send_telegram(notify.fmt_status(
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
    notify.send_telegram(
        f"🚀 EMS-V3 bot started\ntestnet={cfg.testnet}  dry_run={cfg.dry_run}\n"
        f"risk ${cfg.risk_usd}  kill {cfg.max_daily_loss_r}R  H4 {cfg.h4_ema_fast}/{cfg.h4_ema_slow}")
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
            # process is still alive (heartbeat continues) — surface the logic
            # error via Telegram; do NOT mark healthchecks down for a retryable tick.
            log(f"[ERROR] tick failed: {e!r}")
            notify.send_telegram(notify.fmt_blocked("TICK ERROR", repr(e)))
