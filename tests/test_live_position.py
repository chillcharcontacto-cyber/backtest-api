import pandas as pd
import pytest

from ems_live.config import LiveConfig
from ems_live.position import (PositionState, PositionStore, reconcile,
                               FLAT, IN_POSITION,
                               EV_OK_FLAT, EV_OK_RESUME,
                               EV_CLOSED_WHILE_DOWN, EV_UNEXPECTED_POS)
from ems_live.runner import (compute_size, guard_order, seconds_until_next_bar)


# --------------------------- reconcile matrix --------------------------- #

def test_reconcile_flat_flat():
    new, ev = reconcile(PositionState(), None)
    assert new.is_flat()
    assert ev[0].kind == EV_OK_FLAT


def test_reconcile_inpos_inpos_resumes():
    local = PositionState(status=IN_POSITION, entry_price=100.0,
                          sl_price=95.0, size=2.0)
    new, ev = reconcile(local, {"size": 2.0, "entry_px": 100.0, "unrealized_pnl": 1.0})
    assert new.status == IN_POSITION
    assert new.sl_price == 95.0          # local meta preserved
    assert ev[0].kind == EV_OK_RESUME


def test_reconcile_inpos_exchflat_closed_while_down():
    local = PositionState(status=IN_POSITION, entry_price=100.0,
                          sl_price=95.0, size=2.0)
    new, ev = reconcile(local, None)
    assert new.is_flat()
    assert ev[0].kind == EV_CLOSED_WHILE_DOWN


def test_reconcile_flat_exchpos_unexpected():
    new, ev = reconcile(PositionState(),
                        {"size": 1.5, "entry_px": 200.0, "unrealized_pnl": 0.0})
    assert new.status == IN_POSITION
    assert new.entry_price == 200.0
    assert ev[0].kind == EV_UNEXPECTED_POS


# --------------------------- store roundtrip --------------------------- #

def test_store_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    store = PositionStore(str(p))
    assert store.load().is_flat()        # missing file -> flat
    s = PositionState(status=IN_POSITION, entry_price=123.4, sl_price=120.0,
                      size=1.2, stop_oid=999)
    store.save(s)
    back = store.load()
    assert back.status == IN_POSITION
    assert back.entry_price == 123.4
    assert back.stop_oid == 999


# --------------------------- sizing --------------------------- #

def test_compute_size_basic():
    # risk 20, stop 5 wide -> 4 units
    assert compute_size(100.0, 95.0, 20.0) == pytest.approx(4.0)


def test_compute_size_rejects_bad_sl():
    with pytest.raises(ValueError):
        compute_size(100.0, 100.0, 20.0)
    with pytest.raises(ValueError):
        compute_size(100.0, 105.0, 20.0)


# --------------------------- guards --------------------------- #

def test_guard_ok():
    cfg = LiveConfig(risk_usd=20.0, max_notional_usd=10000.0,
                     max_risk_band_pct=15.0, min_risk_pct=0.1)
    size = compute_size(100.0, 98.0, 20.0)   # 2% stop, size=10, notional=1000
    ok, reason = guard_order(100.0, 98.0, size, cfg)
    assert ok, reason


def test_guard_sl_above_entry():
    cfg = LiveConfig()
    ok, _ = guard_order(100.0, 101.0, 1.0, cfg)
    assert not ok


def test_guard_stop_too_wide():
    cfg = LiveConfig(max_risk_band_pct=15.0)
    # 20% stop -> too wide
    ok, reason = guard_order(100.0, 80.0, 0.01, cfg)
    assert not ok and "too wide" in reason


def test_guard_notional_ceiling():
    cfg = LiveConfig(max_notional_usd=500.0, max_risk_band_pct=50.0)
    # entry 100, sl 90 -> size for risk 20 = 2 -> notional 200 ok;
    # force big size to breach ceiling
    ok, reason = guard_order(100.0, 90.0, 10.0, cfg)   # notional 1000 > 500
    assert not ok and "notional" in reason


# --------------------------- scheduler --------------------------- #

def test_seconds_until_next_bar_before_30():
    now = pd.Timestamp("2026-01-01 10:10:00", tz="UTC")
    s = seconds_until_next_bar(now, buffer_sec=15)
    # next boundary 10:30 -> 20 min + 15s
    assert s == pytest.approx(20 * 60 + 15)


def test_seconds_until_next_bar_after_30():
    now = pd.Timestamp("2026-01-01 10:45:00", tz="UTC")
    s = seconds_until_next_bar(now, buffer_sec=15)
    # next boundary 11:00 -> 15 min + 15s
    assert s == pytest.approx(15 * 60 + 15)
