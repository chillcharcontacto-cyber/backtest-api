"""
Max-daily-loss kill switch.

Tracks realized R per UTC day. When cumulative realized R for the day drops to or
below -max_daily_loss_r, new entries are halted for the rest of that day. The
ledger rolls over automatically at the UTC date boundary.

Persisted as a small JSON sidecar next to the position state. Pure + testable.
"""
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class DayLedger:
    day:        str   = ""      # 'YYYY-MM-DD' UTC
    realized_r: float = 0.0     # cumulative realized R for `day`
    trades:     int   = 0       # closed trades counted today


class DayLedgerStore:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> DayLedger:
        if not os.path.exists(self.path):
            return DayLedger()
        with open(self.path, "r", encoding="utf-8") as f:
            return DayLedger(**json.load(f))

    def save(self, ledger: DayLedger) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(ledger), f, indent=2)
        os.replace(tmp, self.path)


def roll(ledger: DayLedger, today: str) -> DayLedger:
    """Return a ledger for `today`, reset if the stored day differs."""
    if ledger.day == today:
        return ledger
    return DayLedger(day=today, realized_r=0.0, trades=0)


def record(ledger: DayLedger, r: float, today: str) -> DayLedger:
    """Add a closed trade's R to today's tally (rolling the day first)."""
    led = roll(ledger, today)
    led.realized_r += r
    led.trades += 1
    return led


def is_halted(ledger: DayLedger, today: str, max_daily_loss_r: float) -> bool:
    """
    True if today's cumulative realized R has hit the loss limit.
    max_daily_loss_r <= 0 disables the kill switch.
    """
    if max_daily_loss_r <= 0:
        return False
    led = roll(ledger, today)
    return led.realized_r <= -abs(max_daily_loss_r)
