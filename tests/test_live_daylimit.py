import pytest

from ems_live.daylimit import (DayLedger, DayLedgerStore, roll, record, is_halted)


def test_roll_resets_on_new_day():
    led = DayLedger(day="2026-06-09", realized_r=-2.0, trades=3)
    rolled = roll(led, "2026-06-10")
    assert rolled.day == "2026-06-10"
    assert rolled.realized_r == 0.0
    assert rolled.trades == 0


def test_roll_keeps_same_day():
    led = DayLedger(day="2026-06-10", realized_r=-2.0, trades=3)
    assert roll(led, "2026-06-10") is led


def test_record_accumulates():
    led = DayLedger()
    led = record(led, -1.0, "2026-06-10")
    led = record(led, 2.5, "2026-06-10")
    assert led.realized_r == pytest.approx(1.5)
    assert led.trades == 2
    assert led.day == "2026-06-10"


def test_record_rolls_then_adds():
    led = DayLedger(day="2026-06-09", realized_r=-5.0, trades=4)
    led = record(led, -1.0, "2026-06-10")   # new day -> reset then add
    assert led.day == "2026-06-10"
    assert led.realized_r == pytest.approx(-1.0)
    assert led.trades == 1


def test_halt_triggers_at_limit():
    led = DayLedger(day="2026-06-10", realized_r=-3.0, trades=3)
    assert is_halted(led, "2026-06-10", 3.0) is True


def test_halt_not_triggered_above_limit():
    led = DayLedger(day="2026-06-10", realized_r=-2.5, trades=2)
    assert is_halted(led, "2026-06-10", 3.0) is False


def test_halt_disabled_when_limit_zero():
    led = DayLedger(day="2026-06-10", realized_r=-99.0, trades=9)
    assert is_halted(led, "2026-06-10", 0.0) is False


def test_halt_resets_next_day():
    led = DayLedger(day="2026-06-10", realized_r=-9.0, trades=5)
    # querying for a new day rolls -> not halted
    assert is_halted(led, "2026-06-11", 3.0) is False


def test_store_roundtrip(tmp_path):
    p = tmp_path / "ledger.json"
    s = DayLedgerStore(str(p))
    assert s.load().day == ""          # missing -> empty
    s.save(DayLedger(day="2026-06-10", realized_r=-1.5, trades=2))
    back = s.load()
    assert back.day == "2026-06-10"
    assert back.realized_r == pytest.approx(-1.5)
    assert back.trades == 2
