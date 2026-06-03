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

from ems.indicators import add_emas, add_h1_emas, mark_crossovers
from .config import LiveConfig
from .decider import build_ctx, check_entry, check_h1_exit
from .feed import fetch_binance_recent
from .sl_adapter import adapt_sl_to_hl
from .position import (PositionState, PositionStore, reconcile,
                       FLAT, IN_POSITION)


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
    """Fetch + indicator-decorate the Binance signal frames."""
    m30 = fetch_binance_recent(cfg.binance_symbol, "30m", cfg.lookback_m30)
    h1  = fetch_binance_recent(cfg.binance_symbol, "1h",  cfg.lookback_h1)
    m30 = mark_crossovers(add_emas(m30, cfg.ema_fast, cfg.ema_slow))
    h1  = add_h1_emas(h1, cfg.h1_trend_ema, cfg.h1_exit_ema)
    return m30, h1


# --------------------------------------------------------------------------- #
#  One tick (testable)                                                         #
# --------------------------------------------------------------------------- #

def tick(cfg: LiveConfig, store: PositionStore, broker, log=print) -> PositionState:
    """
    Evaluate the most recent closed bar once and act. Returns the new state.
    `broker` may be None in pure dry_run flows that never touch the exchange.
    """
    state = store.load()
    m30, h1 = load_signal_frames(cfg)
    ctx = build_ctx(m30, h1)
    i = len(m30) - 1
    t = ctx.m30_times[i]
    log(f"[tick] bar={t}  status={state.status}")

    # ---------------- IN POSITION: manage H1 exit ----------------
    if state.status == IN_POSITION:
        ex = check_h1_exit(ctx, i, state.entry_price, state.sl_price)
        if ex is None:
            log("  in position — no H1 exit this bar (stop resting on exchange)")
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
        return store_flat(store)

    # ---------------- FLAT: check entry ----------------
    sig = check_entry(ctx, i, cfg)
    if sig is None:
        log("  flat — no entry signal")
        return state

    # adapt SL to Hyperliquid candles over the Binance anchor range
    try:
        sl_hl = adapt_sl_to_hl(sig.anchor_time, sig.crossover_time,
                               cfg.coin, cfg.hl_api_url)
    except RuntimeError as e:
        log(f"  ENTRY ABORTED — SL adaptation failed: {e}")
        return state

    entry = sig.entry_price
    try:
        size = compute_size(entry, sl_hl, cfg.risk_usd)
    except ValueError as e:
        log(f"  ENTRY ABORTED — sizing error: {e}")
        return state

    ok, reason = guard_order(entry, sl_hl, size, cfg)
    if not ok:
        log(f"  ENTRY REFUSED by guard: {reason}")
        return state

    log(f"  ENTRY SIGNAL  entry~{entry:.2f}  binance_sl={sig.sl_price:.2f}  "
        f"hl_sl={sl_hl:.2f}  size={size:.6f}  notional={size*entry:.2f}")

    if cfg.dry_run:
        log("  DRY_RUN: would market_entry(size) then place_stop(hl_sl, size)")
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
        return store_flat(store)

    if stop_oid == -1:
        log("  stop triggered on placement (price already through SL) — position closed")
        return store_flat(store)

    new = PositionState(
        status=IN_POSITION,
        entry_time=str(t), entry_price=actual_entry, sl_price=sl_hl, size=size,
        anchor_time=str(sig.anchor_time), crossover_time=str(sig.crossover_time),
        stop_oid=stop_oid,
    )
    store.save(new)
    log(f"  POSITION OPEN  entry={actual_entry:.2f} stop={sl_hl:.2f} oid={stop_oid}")
    return new


def store_flat(store: PositionStore) -> PositionState:
    s = PositionState()
    store.save(s)
    return s


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
    boot_reconcile(cfg, store, broker, log)
    while True:
        now = pd.Timestamp.utcnow()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        sleep_s = seconds_until_next_bar(now, cfg.poll_buffer_sec)
        log(f"[sleep] {sleep_s:.0f}s until next bar")
        time.sleep(sleep_s)
        try:
            tick(cfg, store, broker, log)
        except Exception as e:   # never let one bad tick kill the worker
            log(f"[ERROR] tick failed: {e!r}")
