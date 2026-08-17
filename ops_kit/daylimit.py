"""
Max-daily-loss kill switch.  [PORTS VERBATIM — venue-agnostic]

Tracks realized R per UTC day. When EITHER limit trips, new entries are halted for
the rest of that UTC day; the ledger rolls over automatically at the date boundary:
  - max_daily_losses : number of losing trades today (EMS uses 10)
  - max_daily_loss_r : cumulative realized R today (disabled at <= 0)

Persisted as a small JSON sidecar next to the position state. Pure + testable.
This file has NO exchange/strategy coupling — copy it as-is into the new bot.
"""
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class DayLedger:
    day:        str   = ""      # 'YYYY-MM-DD' UTC
    realized_r: float = 0.0     # cumulative realized R for `day`
    trades:     int   = 0       # closed trades counted today
    losses:     int   = 0       # losing trades (r < 0) counted today


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
    return DayLedger(day=today, realized_r=0.0, trades=0, losses=0)


def record(ledger: DayLedger, r: float, today: str) -> DayLedger:
    """Add a closed trade's R to today's tally (rolling the day first)."""
    led = roll(ledger, today)
    led.realized_r += r
    led.trades += 1
    if r < 0:
        led.losses += 1
    return led


def is_halted(ledger: DayLedger, today: str,
              max_daily_loss_r: float = 0.0,
              max_daily_losses: int = 0) -> bool:
    """
    Kill switch — halt new entries for the rest of the UTC day when EITHER limit
    trips (each disabled when its value is <= 0):
      - max_daily_losses : number of losing trades today
      - max_daily_loss_r : cumulative realized R today
    """
    led = roll(ledger, today)
    if max_daily_losses > 0 and led.losses >= max_daily_losses:
        return True
    if max_daily_loss_r > 0 and led.realized_r <= -abs(max_daily_loss_r):
        return True
    return False
