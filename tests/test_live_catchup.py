"""
Missed-entry catch-up + rate-limit handling for the live runner.

Validates the refactored entry path (_attempt_entry) and the deferral/retry logic:
  - a normal entry fills, places the stop, and clears any pending
  - a 429 on the ORDER defers (saves pending) instead of aborting -> next bar retries
  - an ambiguous NON-429 order error aborts (no double-fill), pending NOT saved
  - permanent rejections (kill switch / guard) clear pending and never enter
  - _try_pending_entry retries a fresh pending, drops a stale one
  - _note_rate_limit sends ONE card/day then stays silent (count kept)
"""
import pandas as pd
import pytest

from ems_live.config import LiveConfig
from ems_live.position import PositionState, IN_POSITION
from ems_live import runner
from ems_live import nethttp


class RateLimit(Exception):
    def __init__(self): self.status_code = 429


class Boom(Exception):          # ambiguous non-rate-limit error
    pass


def test_is_rate_limit():
    assert nethttp.is_rate_limit(RateLimit()) is True
    assert nethttp.is_rate_limit(Boom()) is False
    class R:  # requests-style
        class response: status_code = 503
    assert nethttp.is_rate_limit(R()) is True


class FakeStore:
    def __init__(self, s=None): self.s = s or PositionState()
    def load(self): return self.s
    def save(self, s): self.s = s


class FakeBroker:
    def __init__(self, equity=1000.0, fail_on=None, exc=RateLimit):
        self.equity = equity; self.fail_on = set(fail_on or []); self.exc = exc
        self.calls = []; self.coin_max_leverage = 40; self.maint_margin_frac = 0.0125
    def _f(self, name):
        if name in self.fail_on: raise self.exc()
    def account_value(self): self._f("account_value"); return self.equity
    def mid_price(self): return 64000.0
    def update_leverage(self, l): self._f("update_leverage"); self.calls.append(("lev", l))
    def market_entry(self, sz): self._f("market_entry"); self.calls.append(("entry", sz)); return {"avg_px": 64000.0, "filled": sz}
    def place_stop(self, sl, sz): self.calls.append(("stop", sl, sz)); return 111
    def market_close(self): self.calls.append(("close",))
    def cancel_order(self, o): pass


@pytest.fixture(autouse=True)
def _no_net(monkeypatch):
    # adapt_sl_to_hl hits HL candles — stub it to a fixed stop.
    monkeypatch.setattr(runner, "adapt_sl_to_hl", lambda a, c, coin, url: 63000.0)


def _cfg(tmp_path, dry=False):
    return LiveConfig(dry_run=dry, risk_usd=6.0, state_path=str(tmp_path / "s.json"))


T = pd.Timestamp("2026-08-27 12:00:00+00:00")
ANCHOR = pd.Timestamp("2026-08-27 11:00:00+00:00")
CROSS = pd.Timestamp("2026-08-27 11:30:00+00:00")


def test_entry_fills_and_clears_pending(tmp_path):
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker()
    out = runner._attempt_entry(cfg, store, broker, T, ANCHOR, CROSS, 63000.0, 64000.0, "2026-08-27", print)
    assert out is not None and out.status == IN_POSITION and out.stop_oid == 111
    entry_calls = [c for c in broker.calls if c[0] == "entry"]
    assert entry_calls and entry_calls[0][1] == pytest.approx(out.size)
    assert runner._load_pending(cfg) is None            # cleared


def test_429_on_order_defers_and_saves_pending(tmp_path):
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker(fail_on=["market_entry"], exc=RateLimit)
    out = runner._attempt_entry(cfg, store, broker, T, ANCHOR, CROSS, 63000.0, 64000.0, "2026-08-27", print)
    assert out is None
    pend = runner._load_pending(cfg)
    assert pend is not None and pend["crossover_time"] == str(CROSS)   # will retry next bar
    assert store.s.status != IN_POSITION


def test_ambiguous_error_aborts_no_pending(tmp_path):
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker(fail_on=["market_entry"], exc=Boom)
    out = runner._attempt_entry(cfg, store, broker, T, ANCHOR, CROSS, 63000.0, 64000.0, "2026-08-27", print)
    assert out is None
    assert runner._load_pending(cfg) is None            # NOT saved -> never retried (no double-fill)


def test_429_on_equity_defers(tmp_path):
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker(fail_on=["account_value"], exc=RateLimit)
    out = runner._attempt_entry(cfg, store, broker, T, ANCHOR, CROSS, 63000.0, 64000.0, "2026-08-27", print)
    assert out is None and runner._load_pending(cfg) is not None
    assert broker.calls == []                            # never sent an order


def test_kill_switch_clears_pending_and_skips(tmp_path):
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker()
    runner._save_pending(cfg, ANCHOR, CROSS, 63000.0, T)           # a pending exists
    from ems_live.daylimit import DayLedgerStore, DayLedger
    DayLedgerStore(cfg.state_path + ".day").save(DayLedger(day="2026-08-27", losses=99, trades=99, realized_r=-99))
    out = runner._attempt_entry(cfg, store, broker, T, ANCHOR, CROSS, 63000.0, 64000.0, "2026-08-27", print)
    assert out is None and broker.calls == [] and runner._load_pending(cfg) is None


def test_guard_refuses_broken_setup_clears_pending(tmp_path, monkeypatch):
    # price fell BELOW the stop -> guard refuses (sl not below entry) -> clear pending
    monkeypatch.setattr(runner, "adapt_sl_to_hl", lambda a, c, coin, url: 65000.0)  # sl above mid(64000)
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker()
    runner._save_pending(cfg, ANCHOR, CROSS, 65000.0, T)
    out = runner._attempt_entry(cfg, store, broker, T, ANCHOR, CROSS, 65000.0, 64000.0, "2026-08-27", print)
    assert out is None and broker.calls == [] and runner._load_pending(cfg) is None


class FakeCtx:
    def __init__(self, now):
        self.m30_times = [now]; self.thirty_min = pd.Timedelta(minutes=30); self.m30_closes = [64000.0]


def test_pending_retry_fresh_enters(tmp_path):
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker()
    runner._save_pending(cfg, ANCHOR, CROSS, 63000.0, CROSS)
    ctx = FakeCtx(CROSS + pd.Timedelta(minutes=30))       # 1 bar after crossover
    out = runner._try_pending_entry(cfg, store, broker, ctx, 0, "2026-08-27", print)
    assert out is not None and out.status == IN_POSITION
    assert runner._load_pending(cfg) is None


def test_pending_retry_stale_dropped(tmp_path):
    cfg = _cfg(tmp_path); store = FakeStore(); broker = FakeBroker()
    runner._save_pending(cfg, ANCHOR, CROSS, 63000.0, CROSS)
    ctx = FakeCtx(CROSS + pd.Timedelta(hours=4))          # way past CATCHUP_MAX_BARS
    out = runner._try_pending_entry(cfg, store, broker, ctx, 0, "2026-08-27", print)
    assert out is None and broker.calls == [] and runner._load_pending(cfg) is None


def test_no_pending_returns_none(tmp_path):
    cfg = _cfg(tmp_path); ctx = FakeCtx(T)
    assert runner._try_pending_entry(cfg, FakeStore(), FakeBroker(), ctx, 0, "2026-08-27", print) is None


def test_rate_limit_alert_throttled(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    sent = []
    monkeypatch.setattr(runner.notify, "send_telegram", lambda m: sent.append(m))
    runner._note_rate_limit(cfg, print)      # first today -> card
    runner._note_rate_limit(cfg, print)      # same day -> silent
    runner._note_rate_limit(cfg, print)
    assert len(sent) == 1                     # only ONE card despite 3 hits
