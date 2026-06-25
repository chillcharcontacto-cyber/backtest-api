import pytest

from ems_live.daylimit import (DayLedger, DayLedgerStore, roll, record, is_halted)


def test_roll_resets_on_new_day():
    led = DayLedger(day="2026-06-09", realized_r=-2.0, trades=3, losses=2)
    rolled = roll(led, "2026-06-10")
    assert rolled.day == "2026-06-10"
    assert rolled.realized_r == 0.0
    assert rolled.trades == 0
    assert rolled.losses == 0


def test_record_accumulates_and_counts_losses():
    led = DayLedger()
    led = record(led, -1.0, "2026-06-10")    # loss
    led = record(led, 2.5, "2026-06-10")     # win
    led = record(led, -0.3, "2026-06-10")    # small loss
    assert led.realized_r == pytest.approx(1.2)
    assert led.trades == 3
    assert led.losses == 2                    # two r<0 trades
    assert led.day == "2026-06-10"


def test_record_rolls_then_adds():
    led = DayLedger(day="2026-06-09", realized_r=-5.0, trades=4, losses=4)
    led = record(led, -1.0, "2026-06-10")
    assert led.day == "2026-06-10"
    assert led.realized_r == pytest.approx(-1.0)
    assert led.trades == 1
    assert led.losses == 1


# --------------------------- count-based kill switch --------------------------- #

def test_halt_on_loss_count():
    led = DayLedger(day="2026-06-10", losses=10, trades=12)
    assert is_halted(led, "2026-06-10", max_daily_losses=10) is True


def test_no_halt_below_loss_count():
    led = DayLedger(day="2026-06-10", losses=9, trades=11)
    assert is_halted(led, "2026-06-10", max_daily_losses=10) is False


def test_loss_count_disabled_when_zero():
    led = DayLedger(day="2026-06-10", losses=99, trades=99)
    assert is_halted(led, "2026-06-10", max_daily_losses=0) is False


# --------------------------- R-based kill switch (still supported) --------------------------- #

def test_halt_on_r_limit():
    led = DayLedger(day="2026-06-10", realized_r=-3.0, trades=3, losses=3)
    assert is_halted(led, "2026-06-10", max_daily_loss_r=3.0) is True


def test_either_limit_trips():
    # R off, count trips
    led = DayLedger(day="2026-06-10", realized_r=-1.0, losses=10, trades=12)
    assert is_halted(led, "2026-06-10", max_daily_loss_r=0.0, max_daily_losses=10) is True
    # count off, R trips
    led2 = DayLedger(day="2026-06-10", realized_r=-8.0, losses=4, trades=5)
    assert is_halted(led2, "2026-06-10", max_daily_loss_r=8.0, max_daily_losses=0) is True


def test_halt_resets_next_day():
    led = DayLedger(day="2026-06-10", losses=10, trades=12)
    assert is_halted(led, "2026-06-11", max_daily_losses=10) is False


def test_store_roundtrip(tmp_path):
    p = tmp_path / "ledger.json"
    s = DayLedgerStore(str(p))
    assert s.load().day == ""
    s.save(DayLedger(day="2026-06-10", realized_r=-1.5, trades=2, losses=1))
    back = s.load()
    assert back.day == "2026-06-10"
    assert back.losses == 1
