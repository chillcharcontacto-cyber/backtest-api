"""
Position state machine + persistence + boot reconciliation.  [PORTS VERBATIM]

The bot keeps a local JSON record of its position. On every boot (and you can call
it per-tick too) it reconciles that record against the BROKER, which is the ultimate
source of truth, and emits events describing any drift (position closed while the bot
was down, unexpected manual position, etc.). The runner acts on those events.

No orders here — pure state + IO. Fully unit-testable. Venue-agnostic: the only thing
the new bot must provide is `broker.open_position()` returning None (flat) or a dict
{"size","entry_px","unrealized_pnl"}. For cTrader that comes from ProtoOAReconcileReq.

Field meanings are venue-neutral:
  entry_price   -> actual fill price
  sl_price      -> protective stop level
  size          -> position size in the venue's unit (coin units on HL; volume/units
                   or lots on cTrader — keep ONE convention across the bot)
  stop_oid      -> id of the resting protective order (HL order id; cTrader
                   position/order id). None until the stop is confirmed placed.
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
    entry_price:    Optional[float] = None   # actual fill
    sl_price:       Optional[float] = None   # protective stop level
    size:           Optional[float] = None   # venue units (positive = long)
    anchor_time:    Optional[str]   = None   # ISO8601 — strategy SL anchor bar
    crossover_time: Optional[str]   = None   # ISO8601 — strategy signal bar
    stop_oid:       Optional[int]   = None   # resting stop order id at the broker

    def is_flat(self) -> bool:
        return self.status == FLAT


@dataclass
class ReconcileEvent:
    kind:    str      # see constants below
    detail:  str
    payload: dict = field(default_factory=dict)


# event kinds
EV_OK_FLAT           = "OK_FLAT"            # local flat, broker flat
EV_OK_RESUME         = "OK_RESUME"          # local in-pos, broker in-pos -> resume managing
EV_CLOSED_WHILE_DOWN = "CLOSED_WHILE_DOWN"  # local in-pos, broker flat -> SL/TP hit while off
EV_UNEXPECTED_POS    = "UNEXPECTED_POSITION"# local flat, broker in-pos -> manual/unknown


class PositionStore:
    """Loads/saves PositionState as JSON at `path` (atomic replace)."""

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
              broker_position: Optional[dict]) -> (PositionState, List[ReconcileEvent]):
    """
    Compare local record vs broker truth; return (new_state, events).

    broker_position: None if flat, else
        {"size": float, "entry_px": float, "unrealized_pnl": float}

    Resolution matrix:
      local FLAT      + broker FLAT    -> OK_FLAT (no change)
      local IN_POS    + broker IN_POS  -> OK_RESUME (keep local meta: SL/entry/context)
      local IN_POS    + broker FLAT    -> CLOSED_WHILE_DOWN -> reset to FLAT
      local FLAT      + broker IN_POS  -> UNEXPECTED_POSITION -> adopt broker truth,
                                          but SL/context unknown (runner must flatten
                                          or alert; never silently manage)
    """
    events: List[ReconcileEvent] = []
    in_pos = broker_position is not None

    if local.is_flat() and not in_pos:
        events.append(ReconcileEvent(EV_OK_FLAT, "flat/flat — clean start"))
        return PositionState(), events

    if (not local.is_flat()) and in_pos:
        events.append(ReconcileEvent(
            EV_OK_RESUME,
            "resuming managed position",
            {"exch_size": broker_position["size"],
             "exch_entry": broker_position["entry_px"]},
        ))
        return local, events

    if (not local.is_flat()) and not in_pos:
        events.append(ReconcileEvent(
            EV_CLOSED_WHILE_DOWN,
            "position closed while bot was down (SL/TP hit) — recording flat",
            {"last_entry": local.entry_price, "last_sl": local.sl_price},
        ))
        return PositionState(), events

    # local FLAT + broker IN_POS
    events.append(ReconcileEvent(
        EV_UNEXPECTED_POS,
        "broker has a position the bot did not open — SL context unknown",
        {"exch_size": broker_position["size"],
         "exch_entry": broker_position["entry_px"]},
    ))
    adopted = PositionState(
        status=IN_POSITION,
        entry_price=broker_position["entry_px"],
        size=broker_position["size"],
    )
    return adopted, events
