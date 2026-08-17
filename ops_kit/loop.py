"""
Generic live loop: boot-reconcile → sleep-to-next-bar (pinging) → tick → repeat.
[PORTS ALMOST VERBATIM — this is the EMS run_forever with the strategy factored out]

What this gives you for free, on any venue:
  * boot reconciliation against the broker (survives Render restarts/redeploys)
  * a minute heartbeat to healthchecks.io between bars (hang detection ~3 min)
  * a 🚀 start card + liveness ping from boot
  * the golden rule: ONE bad tick NEVER kills the worker — the exception is caught,
    surfaced to Telegram as a TICK ERROR, and the loop continues (healthchecks stays
    green because the process is alive; only a real hang trips the dead-man's switch).

You provide:
  * `tick_fn(cfg, store, broker, log) -> None` — your strategy's per-bar logic
    (entry/exit/sizing/orders). This is the ONLY strategy-specific piece.
  * a broker satisfying broker_protocol.Broker
  * cfg with: strategy_name, testnet, dry_run, risk_usd, max_daily_losses,
    max_daily_loss_r, poll_buffer_sec, heartbeat_sec, isolated_margin, leverage
  * bar_minutes: your signal timeframe in minutes (EMS = 30)

Weekend/session gaps (forex): the loop keeps sleeping+pinging across closed markets;
tick_fn simply finds no fresh signal and returns. Nothing special needed.
"""
import pandas as pd

from . import monitor
from .position import PositionStore, reconcile


def seconds_until_next_bar(now_utc: pd.Timestamp, bar_minutes: int, buffer_sec: int) -> float:
    """Seconds from now until the next bar boundary + buffer. Handles any bar size
    that divides the hour (30, 15, 5) and multi-hour bars alike."""
    step = pd.Timedelta(minutes=bar_minutes)
    epoch = now_utc.floor("D")
    elapsed = now_utc - epoch
    n = int(elapsed / step) + 1
    next_boundary = epoch + n * step
    delta = (next_boundary - now_utc).total_seconds() + buffer_sec
    return max(delta, 1.0)


def boot_reconcile(cfg, store: PositionStore, broker, log=print):
    """Set margin mode once (live only), then reconcile local state vs broker truth."""
    if broker is not None and not cfg.dry_run:
        try:
            broker.set_margin_mode()
            log(f"[boot] margin mode set: {getattr(cfg, 'leverage', '?')}x "
                f"{'isolated' if getattr(cfg, 'isolated_margin', True) else 'cross'}")
        except Exception as e:
            log(f"[boot] set_margin_mode failed: {e!r}")

    local = store.load()
    broker_pos = broker.open_position() if broker else None
    new, events = reconcile(local, broker_pos)
    for ev in events:
        log(f"[reconcile] {ev.kind}: {ev.detail}  {ev.payload or ''}")
    store.save(new)
    return new


def _kill_desc(cfg) -> str:
    bits = []
    if getattr(cfg, "max_daily_losses", 0) > 0:
        bits.append(f"{cfg.max_daily_losses} losses")
    if getattr(cfg, "max_daily_loss_r", 0) > 0:
        bits.append(f"{cfg.max_daily_loss_r}R")
    return "/".join(bits) or "off"


def run_forever(cfg, store: PositionStore, broker, tick_fn, bar_minutes: int, log=print):
    """Long-running loop for the Render worker."""
    import time
    log(f"=== {cfg.strategy_name} LIVE start — testnet={cfg.testnet} dry_run={cfg.dry_run} ===")
    monitor.send_telegram(monitor.fmt_started(
        cfg.strategy_name, cfg.testnet, cfg.dry_run, f"${cfg.risk_usd}", _kill_desc(cfg)))
    boot_reconcile(cfg, store, broker, log)
    monitor.ping_health()                       # liveness from boot

    while True:
        now = pd.Timestamp.utcnow()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        sleep_s = seconds_until_next_bar(now, bar_minutes, cfg.poll_buffer_sec)
        log(f"[sleep] {sleep_s:.0f}s until next bar")

        # Sleep in heartbeat-sized chunks, pinging each chunk so healthchecks knows the
        # process is alive even between bars (hang detection).
        remaining = sleep_s
        while remaining > 0:
            chunk = min(cfg.heartbeat_sec, remaining)
            time.sleep(chunk)
            monitor.ping_health()               # heartbeat (silent)
            remaining -= chunk

        try:
            tick_fn(cfg, store, broker, log)
            monitor.ping_health()               # liveness: tick completed OK
        except Exception as e:                  # never let one bad tick kill the worker
            log(f"[ERROR] tick failed: {e!r}")
            monitor.send_telegram(monitor.fmt_blocked("TICK ERROR", repr(e)))
            # NOTE: deliberately DO NOT ping /fail here — a retryable logic error is not
            # a liveness failure. The process is alive; the next bar retries.
