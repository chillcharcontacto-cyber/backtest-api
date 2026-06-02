"""
Hyperliquid broker wrapper.

SAFETY (this session): order-placing methods are hard-stubbed and raise
NotImplementedError. Only read-only methods are wired. No code path can place,
modify, or cancel an order until Phase 3 explicitly implements them on TESTNET.

Read-only methods need only a public wallet ADDRESS.
Order methods (Phase 3) will additionally need an API wallet SECRET key.
"""
from typing import Optional

from .config import LiveConfig


class LiveBroker:
    def __init__(self, cfg: LiveConfig,
                 address: Optional[str] = None,
                 secret_key: Optional[str] = None):
        self.cfg = cfg
        self.address = address
        self._secret = secret_key

        # Lazy import so the rest of ems_live works without the SDK installed
        from hyperliquid.info import Info
        self._info = Info(cfg.hl_api_url, skip_ws=True)

        self._exchange = None
        if secret_key:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            wallet = Account.from_key(secret_key)
            self._exchange = Exchange(wallet, cfg.hl_api_url,
                                      account_address=address)

    # ----------------------------------------------------------------- #
    #  READ-ONLY (safe)                                                   #
    # ----------------------------------------------------------------- #

    def mid_price(self) -> float:
        """Current mid price for the configured coin."""
        mids = self._info.all_mids()
        return float(mids[self.cfg.coin])

    def account_value(self) -> float:
        """Total account value (USD) from the margin summary."""
        if not self.address:
            raise ValueError("address required for account_value()")
        state = self._info.user_state(self.address)
        return float(state["marginSummary"]["accountValue"])

    def open_position(self) -> Optional[dict]:
        """
        Open position for the configured coin, or None if flat.
        Returns {"size": float, "entry_px": float, "unrealized_pnl": float}.
        """
        if not self.address:
            raise ValueError("address required for open_position()")
        state = self._info.user_state(self.address)
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            if pos.get("coin") == self.cfg.coin and float(pos.get("szi", 0)) != 0:
                return {
                    "size":           float(pos["szi"]),
                    "entry_px":       float(pos["entryPx"]),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                }
        return None

    # ----------------------------------------------------------------- #
    #  ORDER PLACEMENT — STUBBED until Phase 3 (testnet)                   #
    # ----------------------------------------------------------------- #

    def market_entry(self, size: float):
        raise NotImplementedError(
            "Order placement is disabled. Phase 3 (testnet) implements market_entry."
        )

    def place_stop(self, trigger_price: float, size: float):
        raise NotImplementedError(
            "Order placement is disabled. Phase 3 (testnet) implements place_stop."
        )

    def market_close(self):
        raise NotImplementedError(
            "Order placement is disabled. Phase 3 (testnet) implements market_close."
        )
