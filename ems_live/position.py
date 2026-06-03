"""
Position state machine + persistence + boot reconciliation.

The bot keeps a local JSON record of its position (entry, SL, size, signal
context needed for the H1 exit). On every boot it reconciles that record against
the EXCHANGE, which is the ultimate source of truth, and emits events describing
any drift (position closed while the bot was down, unexpected manual position,
etc.). The runner acts on those events.

No orders here — pure state + IO. Fully unit-testable.
"""
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional


FLAT = "FLAT"
IN_POSITION = "IN_POSITION"


@dataclass
class PositionState:
    status:         str = FLAT
    # --- populated when IN_POSITION ---
    entry_time:     Optional[str]   = None   # ISO8601 UTC
    entry_price:    Optional[float] = None   # actual HL fill
    sl_price:       Optional[float] = None   # HL-adapted stop
    size:           Optional[float] = None   # coin units (positive = long)
    anchor_time:    Optional[str]   = None   # ISO8601 — SL anchor bar (Binance)
    crossover_time: Optional[str]   = None   # ISO8601 — crossover bar (Binance)
    stop_oid:       Optional[int]   = None   # resting stop order id on exchange

    def is_flat(self) -> bool:
        return self.status == FLAT


@dataclass
class ReconcileEvent:
    kind:    str      # see constants below
    detail:  str
    payload: dict = field(default_factory=dict)


# event kinds
EV_OK_FLAT        = "OK_FLAT"          # local flat, exchange flat
EV_OK_RESUME      = "OK_RESUME"        # local in-pos, exchange in-pos -> resume managing
EV_CLOSED_WHILE_DOWN = "CLOSED_WHILE_DOWN"  # local in-pos, exchange flat -> SL/TP hit while off
EV_UNEXPECTED_POS = "UNEXPECTED_POSITION"   # local flat, exchange in-pos -> manual/unknown


class PositionStore:
    """Loads/saves PositionState as JSON at `path`."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> PositionState:
        if not os.path.exists(self.path):
            return PositionState()
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PositionState(**data)

    def save(self, state: PositionState) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)
        os.replace(tmp, self.path)   # atomic on same filesystem


def reconcile(local: PositionState,
              exchange_position: Optional[dict]) -> (PositionState, List[ReconcileEvent]):
    """
    Compare local record vs exchange truth; return (new_state, events).

    exchange_position: None if flat, else
        {"size": float, "entry_px": float, "unrealized_pnl": float}

    Resolution matrix:
      local FLAT      + exch FLAT      -> OK_FLAT (no change)
      local IN_POS    + exch IN_POS    -> OK_RESUME (keep local meta: SL/entry/context)
      local IN_POS    + exch FLAT      -> CLOSED_WHILE_DOWN -> reset to FLAT
      local FLAT      + exch IN_POS    -> UNEXPECTED_POSITION -> adopt exch truth,
                                          but SL/context unknown (runner must flatten
                                          or alert; never silently manage)
    """
    events: List[ReconcileEvent] = []
    exch_in_pos = exchange_position is not None

    if local.is_flat() and not exch_in_pos:
        events.append(ReconcileEvent(EV_OK_FLAT, "flat/flat — clean start"))
        return PositionState(), events

    if (not local.is_flat()) and exch_in_pos:
        events.append(ReconcileEvent(
            EV_OK_RESUME,
            "resuming managed position",
            {"exch_size": exchange_position["size"],
             "exch_entry": exchange_position["entry_px"]},
        ))
        return local, events

    if (not local.is_flat()) and not exch_in_pos:
        events.append(ReconcileEvent(
            EV_CLOSED_WHILE_DOWN,
            "position closed while bot was down (SL/TP hit) — recording flat",
            {"last_entry": local.entry_price, "last_sl": local.sl_price},
        ))
        return PositionState(), events

    # local FLAT + exch IN_POS
    events.append(ReconcileEvent(
        EV_UNEXPECTED_POS,
        "exchange has a position the bot did not open — SL context unknown",
        {"exch_size": exchange_position["size"],
         "exch_entry": exchange_position["entry_px"]},
    ))
    adopted = PositionState(
        status=IN_POSITION,
        entry_price=exchange_position["entry_px"],
        size=exchange_position["size"],
    )
    return adopted, events
