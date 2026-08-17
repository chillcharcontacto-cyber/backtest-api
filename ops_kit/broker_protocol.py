"""
Broker contract — the EXACT surface the loop calls on `broker`.  [THE ADAPTATION SEAM]

The generic loop + reconcile never import a venue SDK. They only ever call the methods
below. To port the EMS ops system to a NEW venue you implement ONE class that satisfies
this Protocol; everything else (monitoring, kill switch, state, reconcile, heartbeat)
is reused unchanged.

────────────────────────────────────────────────────────────────────────────────────
Hyperliquid (EMS, reference)          →  cTrader Open API (your new bot)
────────────────────────────────────────────────────────────────────────────────────
open_position()  from clearinghouse   →  ProtoOAReconcileReq (open positions list)
account_value()  perp+spot USDC       →  ProtoOATraderReq .balance / free margin
mid_price()      L2 book mid          →  ProtoOASpotEvent / latest bid-ask mid
market_entry()   IOC market order     →  ProtoOANewOrderReq (MARKET), read ExecutionEvent
place_stop()     reduce-only trigger  →  attach SL via ProtoOAAmendPositionSLTPReq
                                          (or a STOP order); return its id
market_close()   opposite IOC         →  ProtoOAClosePositionReq
cancel_order()   cancel by oid        →  ProtoOACancelOrderReq
update_leverage()/set_margin_mode()   →  NO-OP. Leverage is broker/account-set on
coin_max_leverage / maint_margin_frac    cTrader; there is no per-order leverage or
                                          isolated-liq price to compute. See sizing note.
────────────────────────────────────────────────────────────────────────────────────

SIZING NOTE (important): EMS uses equity-relative AUTO-LEVERAGE because Hyperliquid perps
let you set per-order leverage and expose an isolated liquidation price. cTrader does NOT
work that way — leverage is fixed by the account/symbol and the broker manages margin
call / stop-out. So on cTrader you DROP plan_trade()'s liquidation math and instead:
    volume = risk_ccy / (stop_distance_in_price * pip_value_per_unit)
    round volume to the symbol's volume step; reject if < min volume or > free-margin cap.
Keep the rest (fixed-risk sizing, guard band, min-notional) identical in spirit.
"""
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Broker(Protocol):
    # --- read (used in dry_run too; must work read-only) --------------------- #
    def open_position(self) -> Optional[dict]:
        """None if flat, else {"size": float, "entry_px": float, "unrealized_pnl": float}.
        `size` positive = long. This is the source of truth for reconcile()."""

    def account_value(self) -> float:
        """Total account equity in the risk currency (USD/EUR). Used for sizing and the
        exit card's % line. If this can't be read on the LIVE path, the loop ABORTS the
        entry rather than sizing blind — keep that behavior."""

    def mid_price(self) -> float:
        """Current mid/last price of the traded symbol, for the entry-price estimate."""

    # --- write (only called when NOT dry_run) ------------------------------- #
    def market_entry(self, size: float) -> dict:
        """Send a market order for `size`. Return {"avg_px": float, "filled": float}
        with the ACTUAL filled size (a partial fill must report what really filled —
        the loop sizes P&L and the stop off `filled`, never the requested size)."""

    def place_stop(self, trigger_px: float, size: float) -> Optional[int]:
        """Place the protective stop for the open position. Return the order id.
        Return -1 if price is already through the trigger (treated as an immediate
        stop-out = -1R). Return None on failure (loop then flattens + guards)."""

    def market_close(self) -> None:
        """Flatten the open position at market (opposite side, reduce-only)."""

    def cancel_order(self, oid: int) -> None:
        """Cancel a resting order (the stop) by id."""

    def update_leverage(self, lev: int) -> None:
        """HL: set per-order leverage before entry. cTrader: no-op."""

    def set_margin_mode(self) -> None:
        """HL: set isolated/cross once at boot. cTrader: no-op."""

    # --- venue constants (HL perps). cTrader: return safe dummies ----------- #
    @property
    def coin_max_leverage(self) -> int:
        """HL: coin's max leverage from meta. cTrader: unused — return 1."""

    @property
    def maint_margin_frac(self) -> float:
        """HL: maintenance-margin fraction for the liq calc. cTrader: unused — 0.0."""
