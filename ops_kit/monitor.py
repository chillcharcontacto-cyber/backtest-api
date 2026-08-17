"""
Monitoring: Telegram event push + healthchecks.io liveness ping.  [SENDERS VERBATIM]

The two SENDERS (send_telegram, ping_health) are 100% venue-agnostic — copy as-is.
The FORMATTERS below are generic templates (symbol string + price-format callable,
no hard-coded BTC / EMA ladder). Adapt the trade cards to your strategy's language;
keep the contract:
  * senders NEVER raise (a monitoring failure must not break the trading loop)
  * env is read at CALL time (not import) so the .env loader runs first
  * formatters are PURE (no I/O) so they unit-test without a network

Env vars (all optional — each feature no-ops if unset):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   -> event messages
  HEALTHCHECK_URL                        -> dead-man's-switch ping each tick/heartbeat
"""
import os

import requests


# --------------------------------------------------------------------------- #
#  Senders (side-effecting, env-gated, NEVER raise)   [copy verbatim]          #
# --------------------------------------------------------------------------- #

def send_telegram(msg: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception:
        pass  # monitoring must never crash the bot


def ping_health(suffix: str = "") -> None:
    """Ping healthchecks.io. suffix '/fail' or '/log' supported by the service."""
    url = os.environ.get("HEALTHCHECK_URL")
    if not url:
        return
    try:
        requests.get(url + suffix, timeout=10)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Generic formatters (pure) — adapt wording to your strategy                  #
# --------------------------------------------------------------------------- #

def _p(px, fmt):
    """Format a price with the venue's precision. fmt is a format spec, e.g. ',.1f'
    for BTC or ',.5f' for a forex pair."""
    return format(px, fmt)


def fmt_started(strategy, testnet, dry_run, risk, kill_desc, extra="") -> str:
    return (f"🚀 {strategy} bot started\n"
            f"testnet={testnet}  dry_run={dry_run}\n"
            f"risk {risk}  kill {kill_desc}/day{(' ' + extra) if extra else ''}")


def fmt_entry(strategy, t, symbol, entry, sl, size, risk, px_fmt=",.1f") -> str:
    return (f"🟢 ENTRY  {strategy}\n{t} UTC\n{symbol}\n"
            f"entry {_p(entry, px_fmt)}  sl {_p(sl, px_fmt)}\n"
            f"size {size}  risk {risk}")


def fmt_exit(reason, symbol, exit_px, model_r, net_r, net_pnl, pct,
             fees, swap, deviation_pct, dur_h, day_r, px_fmt=",.1f") -> str:
    """
    model_r      : clean R from price levels (costs ignored)
    net_r        : realized R after ALL costs (= model_r - (fees+swap)/risk)
    swap         : overnight financing / perp funding over the hold (signed).
                   Pass 0.0 if not applicable. (EMS's own card omits this today —
                   the generic kit includes it so the new bot is truthful from day 1.)
    deviation_pct: how far net moved from model due to costs, as % of |model_r|
    """
    is_stop = "SL" in reason.upper() or "STOP" in reason.upper()
    emoji = "🔻" if is_stop else "🔴"
    pct_s = f"{pct:+.2f}%" if pct is not None else "n/a"
    swap_s = f"  swap {swap:+.2f}" if swap else ""
    return (f"{emoji} EXIT ({reason})  {symbol}\n"
            f"exit {_p(exit_px, px_fmt)}\n"
            f"R {model_r:+.2f} → net {net_r:+.2f}  (dev {deviation_pct:+.1f}%)\n"
            f"net P&L {net_pnl:+,.2f}  {pct_s}\n"
            f"fees ~{fees:.2f}{swap_s}  dur {dur_h:.0f}h\n"
            f"day: {day_r:+.2f}R")


def fmt_blocked(kind, detail) -> str:
    """Covers KILL SWITCH / ENTRY ABORTED / RISK RESIZED / STOP FAILED / TICK ERROR /
    UNPROTECTED POSITION — anything the loop needs to surface. 🛑 for kill, ⚠️ else."""
    emoji = "🛑" if "KILL" in kind.upper() else "⚠️"
    return f"{emoji} {kind}: {detail}"


def fmt_heartbeat(gates_desc: str) -> str:
    """Daily 'still alive' when nothing is happening. gates_desc = your TF snapshot,
    e.g. 'H4 ▲ bull · H1 ▽ bear · M30 ▲ bull'."""
    return ("😴 Still alive, waiting for an entry.\n"
            f"{gates_desc}\n"
            "Nothing to do — will ping on the next step.")


# --- Step ladder (generic N-stage version of EMS's H4→H1→M30) --------------- #

def steps_plan(last_stage, hb_day, stage, today):
    """Pure: (send_daily_heartbeat, send_stage_change). Send a step message only when
    the setup stage changes; send one heartbeat per new UTC day."""
    return (hb_day != today, last_stage != stage)


def status_decision(last_gate, gate, mode):
    """
    Legacy flat-status card control (used by mode 'change'/'always').
    gate/last_gate = dict of gate_name -> 0/1. mode: 'off'|'always'|'change'.
    Returns (send: bool, prefix: str).
    """
    if mode == "off":
        return False, ""
    if mode == "always":
        return True, ""
    if last_gate == gate:
        return False, ""
    prefix = ""
    if last_gate is not None:
        flips = [f"{k} {'▲' if gate[k] else '▽'}"
                 for k in gate if last_gate.get(k) != gate[k]]
        if flips:
            prefix = "Δ " + ", ".join(flips) + "\n"
    return True, prefix
