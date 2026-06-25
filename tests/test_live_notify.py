"""Pure-formatter tests for ems_live.notify (no network)."""
from ems_live import notify


def test_fmt_entry_has_key_fields():
    s = notify.fmt_entry("2026-06-25 18:30", 65432.0, 64610.0, 0.0061, 20.0, 3)
    assert "ENTRY" in s
    assert "65,432" in s and "64,610" in s
    assert "0.00610 BTC" in s
    assert "risk $20" in s
    assert "lev 3x" in s
    # notional = size*entry ≈ 399
    assert "399" in s


def test_fmt_exit_tp_with_deviation():
    # model +2.16, net +2.07 after fees -> dev -4.2%
    tp = notify.fmt_exit("H1_EMA100", 67210.0, 2.16, 2.07, 41.4, 0.41, 1.8, -4.2, 14, 1.16)
    assert "🔴" in tp and "H1_EMA100" in tp
    assert "R +2.16 → net +2.07" in tp
    assert "dev -4.2%" in tp
    assert "+41.40$" in tp and "+0.41%" in tp
    assert "day: +1.16R" in tp


def test_fmt_exit_sl_with_deviation():
    # user's example: model -1.0, net -1.1 ($-22), dev -10%
    sl = notify.fmt_exit("STRUCTURAL_SL", 64610.0, -1.0, -1.1, -22.0, -0.22, 2.0, -10.0, 1, -1.0)
    assert "🔻" in sl
    assert "R -1.00 → net -1.10" in sl
    assert "dev -10.0%" in sl
    assert "-22.00$" in sl


def test_fmt_exit_pct_none():
    s = notify.fmt_exit("H1_EMA100", 67000.0, 1.0, 0.95, 19.0, None, 1.0, -5.0, 5, 1.0)
    assert "n/a" in s


def test_fmt_blocked_killswitch_emoji():
    assert "🛑" in notify.fmt_blocked("KILL SWITCH", "halted")
    assert "⚠️" in notify.fmt_blocked("ENTRY REFUSED", "stop too wide")


def test_fmt_status_directions():
    s = notify.fmt_status("2026-06-25 18:30", 65100, 64800, False,
                          65200, 64900, 62300, 67300)
    assert "FLAT" in s
    assert "cross:no" in s
    assert "▲ above" in s          # h1 close above ema50
    assert "▽ blocks long" in s    # h4 ema20 < ema100


def test_fmt_inpos_distance():
    s = notify.fmt_inpos("2026-06-25 18:30", 66800, 65432, 64610, 1.66, 66950)
    assert "IN POS" in s
    assert "R now +1.66" in s
    # distance = 66800 - 66950 = -150
    assert "-150 away" in s


def test_senders_noop_without_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("HEALTHCHECK_URL", raising=False)
    # must not raise when unconfigured
    notify.send_telegram("hello")
    notify.ping_health()
