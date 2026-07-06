import pytest

from ems_live.config import LiveConfig
from ems_live.runner import plan_trade, guard_order, compute_size


def cfg(**kw):
    base = dict(risk_usd=20.0, margin_buffer_frac=0.90, liq_safety_mult=1.5,
                max_leverage=0, min_risk_pct=0.1, max_risk_band_pct=15.0,
                hl_min_notional_usd=10.0, max_notional_usd=0.0)
    base.update(kw)
    return LiveConfig(**base)


# ------------------------- plan_trade: risk stays exact ------------------------- #

def test_tight_stop_fits_at_low_leverage():
    # 0.58% stop, ample equity -> min leverage to fit, liquidation far beyond stop
    p = plan_trade(100000, 99420, 20.0, 999.0, cfg(), coin_max_lev=40, maint_frac=0.0125)
    assert p.feasible and not p.resized
    # size = 20/580 = 0.03448 -> notional ~3448 ; usable ~899 -> lev_fit = ceil(3.83)=4
    assert p.leverage == 4
    assert p.size == pytest.approx(20 / 580, rel=1e-6)
    # exact risk preserved: size*(entry-sl) == risk_usd
    assert p.size * (100000 - 99420) == pytest.approx(20.0, rel=1e-6)


def test_wide_stop_uses_low_leverage():
    # 15% stop -> tiny notional, liq-safe leverage is small, fits at 1x
    p = plan_trade(100000, 85000, 20.0, 999.0, cfg(), coin_max_lev=40, maint_frac=0.0125)
    assert p.feasible and not p.resized
    assert p.leverage <= 4          # liq-safe cap for a 15% stop
    assert p.size * (100000 - 85000) == pytest.approx(20.0, rel=1e-6)


def test_small_equity_tight_stop_resizes_down():
    # $100 equity, 0.3% stop -> can't fit full risk -> resize down, cap at coin max
    p = plan_trade(100000, 99700, 20.0, 100.0, cfg(), coin_max_lev=40, maint_frac=0.0125)
    assert p.feasible and p.resized
    assert p.leverage == 40                      # pinned to coin max
    full = 20 / 300
    assert p.size < full                          # risk reduced
    # notional == usable * leverage
    assert p.notional == pytest.approx(100 * 0.90 * 40, rel=1e-6)


def test_liq_safety_caps_leverage_below_fit():
    # moderate stop where liq-safe cap (6x) binds before margin fit
    p = plan_trade(100000, 90000, 20.0, 30.0, cfg(), coin_max_lev=40, maint_frac=0.0125)
    # 10% stop -> lev_liqsafe = floor(1/(0.10*1.5+0.0125)) = 6
    assert p.leverage == 6
    assert p.resized                              # equity 30 too small for full size at 6x


def test_sl_not_below_entry_infeasible():
    p = plan_trade(100000, 100000, 20.0, 999.0, cfg(), coin_max_lev=40, maint_frac=0.0125)
    assert not p.feasible


def test_leverage_never_exceeds_coin_max():
    p = plan_trade(100000, 99900, 20.0, 50.0, cfg(), coin_max_lev=20, maint_frac=0.025)
    assert p.leverage <= 20


def test_extremely_wide_stop_is_infeasible_not_silent_1x():
    # 70% stop: even 1x can't keep liquidation the buffered distance beyond the stop.
    # plan_trade must flag infeasible, not ship a full-size 1x that violates its contract.
    p = plan_trade(100000, 30000, 20.0, 999.0, cfg(max_risk_band_pct=99.0),
                   coin_max_lev=40, maint_frac=0.0125)
    assert not p.feasible
    assert "too wide" in p.reason


# ------------------------- guard_order: no fixed ceiling ------------------------- #

def test_guard_allows_large_notional_when_ceiling_off():
    # the exact case that was refused before: ~$3438 notional, ceiling OFF
    ok, reason = guard_order(100000, 99420, 20 / 580, cfg(max_notional_usd=0.0))
    assert ok, reason


def test_guard_min_notional_blocks_tiny():
    ok, reason = guard_order(100000, 85000, 0.00005, cfg())  # notional $5 < $10
    assert not ok and "below HL min" in reason


def test_guard_optional_ceiling_still_works_when_set():
    ok, reason = guard_order(100000, 99420, 20 / 580, cfg(max_notional_usd=500.0))
    assert not ok and "ceiling" in reason


def test_guard_stop_bands():
    assert not guard_order(100, 101, 1.0, cfg())[0]            # sl above entry
    assert not guard_order(100, 80, 1.0, cfg())[0]             # 20% > 15% band
