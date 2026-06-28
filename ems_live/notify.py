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


def fmt_exit(reason, exit_px, model_r, net_r, net_pnl, pct, fees_usd,
             deviation_pct, dur_h, day_r) -> str:
    """
    model_r     : clean R from price levels (fees/slippage ignored)
    net_r       : realized R after fees  (= model_r - fees/risk_usd)
    deviation_pct: how far net moved from model due to costs, as % of |model_r|
                   (negative = costs ate into the result; the usual case)
    """
    emoji = "🔻" if reason == "STRUCTURAL_SL" else "🔴"
    pct_s = f"{pct:+.2f}%" if pct is not None else "n/a"
    return (f"{emoji} EXIT ({reason})\n"
            f"exit {exit_px:,.1f}\n"
            f"R {model_r:+.2f} → net {net_r:+.2f}  (dev {deviation_pct:+.1f}%)\n"
            f"net P&L {net_pnl:+,.2f}$  {pct_s}\n"
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


def status_decision(last_gate, gate, mode):
    """
    Decide whether to send a FLAT status card and an optional change-prefix.
    gate / last_gate = {"m30":0/1, "h1":0/1, "h4":0/1} (1 = bullish/allows).
    mode: "off" (never) | "always" (every tick) | "change" (only when a gate flips).
    Returns (send: bool, prefix: str).
    """
    if mode == "off":
        return False, ""
    if mode == "always":
        return True, ""
    # change mode
    if last_gate == gate:
        return False, ""
    prefix = ""
    if last_gate is not None:
        flips = [f"{label} {'▲' if gate[k] else '▽'}"
                 for k, label in (("m30", "M30"), ("h1", "H1"), ("h4", "H4"))
                 if last_gate.get(k) != gate[k]]
        if flips:
            prefix = "Δ " + ", ".join(flips) + "\n"
    return True, prefix


def _arrow(b):
    return "▲" if b else "▽"


def stage_of(h4: bool, h1: bool) -> int:
    """
    Setup ladder stage (top-down). M30 is the trigger inside stage 2, not its own stage.
      0 = H4 bearish (blocked at the top)
      1 = H4 bullish, H1 bearish (waiting on H1)
      2 = H4 + H1 bullish (armed, waiting for a fresh M30 cross)
    """
    if not h4:
        return 0
    if not h1:
        return 1
    return 2


def steps_plan(last_stage, hb_day, stage, today):
    """Pure: (send_daily_heartbeat, send_stage_change)."""
    return (hb_day != today, last_stage != stage)


def fmt_stage(stage, h4, h1, m30) -> str:
    if stage == 0:
        return (f"🔴 H4 bearish — longs blocked. Standing down.\n"
                f"H4 {_arrow(h4)}  (will ping when H4 flips bullish)")
    if stage == 1:
        return (f"🟠 Step 1/3 — H4 bullish ✅\n"
                f"H4 {_arrow(h4)} · H1 {_arrow(h1)} · M30 {_arrow(m30)}\n"
                f"Now waiting on H1 (close > ema50).")
    return (f"🟡 Step 2/3 — H4 ✅ + H1 ✅ — ARMED\n"
            f"H4 {_arrow(h4)} · H1 {_arrow(h1)} · M30 {_arrow(m30)}\n"
            f"Waiting for a fresh M30 cross → 🟢 entry.")


def fmt_heartbeat(h4, h1, m30) -> str:
    bull = lambda b: "▲ bull" if b else "▽ bear"
    return ("😴 Still alive, waiting for an entry.\n"
            f"H4 {bull(h4)} · H1 {bull(h1)} · M30 {bull(m30)}\n"
            "Nothing to do — will ping on the next step.")


def fmt_inpos(t, h1_close, entry, sl, r_now, h1_e100) -> str:
    dist = h1_close - h1_e100
    return (f"🔵 IN POS  {t}\n"
            f"H1 close {h1_close:,.0f}\n"
            f"entry {entry:,.0f}  sl {sl:,.0f}  R now {r_now:+.2f}\n"
            f"exit if H1<EMA100 ({h1_e100:,.0f}) — {dist:+,.0f} away")
