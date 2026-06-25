"""
Monitoring: Telegram event push + healthchecks.io liveness ping.

All network calls are best-effort and wrapped — a notification failure must NEVER
break the trading loop. Env is read at CALL time (not import) so run.py's .env
loader has populated it first.

Env vars (all optional — each feature no-ops if unset):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   -> event messages
  HEALTHCHECK_URL                        -> dead-man's-switch ping each tick

Formatters are pure (no I/O) so they're unit-testable.
"""
import os

import requests


# --------------------------------------------------------------------------- #
#  Senders (side-effecting, env-gated, never raise)                            #
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
#  Formatters (pure)                                                           #
# --------------------------------------------------------------------------- #

def fmt_entry(t, entry, sl, size, risk_usd, leverage) -> str:
    notional = size * entry
    return (f"🟢 ENTRY  EMS-V3\n"
            f"{t} UTC\n"
            f"entry {entry:,.1f}  sl {sl:,.1f}\n"
            f"size {size:.5f} BTC  risk ${risk_usd:.0f}\n"
            f"notional ${notional:,.0f}  lev {leverage}x")


def fmt_exit(reason, exit_px, r, pnl_usd, pct, fees_usd, dur_h, day_r) -> str:
    emoji = "🔻" if reason == "STRUCTURAL_SL" else "🔴"
    pct_s = f"{pct:+.2f}%" if pct is not None else "n/a"
    return (f"{emoji} EXIT ({reason})\n"
            f"exit {exit_px:,.1f}  R {r:+.2f}\n"
            f"P&L {pnl_usd:+,.2f}$  {pct_s}\n"
            f"fees ~${fees_usd:.2f}  dur {dur_h:.0f}h\n"
            f"day: {day_r:+.2f}R")


def fmt_blocked(kind, detail) -> str:
    emoji = "🛑" if "KILL" in kind.upper() else "⚠️"
    return f"{emoji} {kind}: {detail}"


def fmt_status(t, m30_e20, m30_e50, cross, h1_close, h1_e50,
               h4_e20, h4_e100) -> str:
    m30_dir = "▲" if m30_e20 > m30_e50 else "▽"
    h1_dir = "▲ above" if h1_close > h1_e50 else "▽ below"
    h4_dir = "▲ allows" if h4_e20 > h4_e100 else "▽ blocks"
    return (f"⚪ FLAT  {t}\n"
            f"M30 ema20 {m30_e20:,.0f}/ema50 {m30_e50:,.0f} {m30_dir}  "
            f"cross:{'YES' if cross else 'no'}\n"
            f"H1  close {h1_close:,.0f}/ema50 {h1_e50:,.0f} {h1_dir}\n"
            f"H4  ema20 {h4_e20:,.0f}/ema100 {h4_e100:,.0f} {h4_dir} long")


def fmt_inpos(t, h1_close, entry, sl, r_now, h1_e100) -> str:
    dist = h1_close - h1_e100
    return (f"🔵 IN POS  {t}\n"
            f"H1 close {h1_close:,.0f}\n"
            f"entry {entry:,.0f}  sl {sl:,.0f}  R now {r_now:+.2f}\n"
            f"exit if H1<EMA100 ({h1_e100:,.0f}) — {dist:+,.0f} away")
