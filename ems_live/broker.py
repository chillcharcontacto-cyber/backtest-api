"""
Hyperliquid broker wrapper.

Read-only methods need only a public wallet ADDRESS.
Order methods need the API/agent SECRET key (trade-only; cannot withdraw).

Long-only. Stop is a reduce-only trigger-market order resting on the exchange,
so the position stays protected even if the bot process dies.

Rounding: BTC perp szDecimals=5 -> size to 5 dp; perp price decimals = 6 - szDecimals.
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

        from hyperliquid.info import Info
        self._info = Info(cfg.hl_api_url, skip_ws=True)

        # asset metadata for rounding + leverage/margin limits
        meta = self._info.meta()
        self._asset = next(a for a in meta["universe"] if a["name"] == cfg.coin)
        self._sz_decimals = int(self._asset["szDecimals"])
        self._px_decimals = max(0, 6 - self._sz_decimals)   # perps: MAX_DECIMALS=6
        self._max_leverage = int(self._asset.get("maxLeverage", 20))
        # HL maintenance margin ≈ half the initial margin at max leverage
        self._maint_margin_frac = 0.5 / self._max_leverage

        self._exchange = None
        if secret_key:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            wallet = Account.from_key(secret_key)
            self._exchange = Exchange(wallet, cfg.hl_api_url,
                                      account_address=address)

    # ----------------------------------------------------------------- #
    #  Rounding helpers                                                   #
    # ----------------------------------------------------------------- #

    def round_size(self, sz: float) -> float:
        return round(sz, self._sz_decimals)

    def _px_decimals_for(self, px: float) -> int:
        import math
        sig_decimals = 5 - 1 - int(math.floor(math.log10(abs(px))))  # decimals for 5 sig figs
        return min(sig_decimals, self._px_decimals)                  # may be negative

    def round_px(self, px: float) -> float:
        """
        Hyperliquid perp price: at most 5 significant figures AND at most
        (6 - szDecimals) decimal places. Integer prices always allowed.
        e.g. BTC 63522.9 -> 63523 (5 sig figs forces integer at ~65k).
        """
        if px <= 0:
            return 0.0
        return round(px, self._px_decimals_for(px))

    def round_stop_px(self, px: float) -> float:
        """
        Round a LONG's stop trigger DOWN to a valid tick, so rounding never nudges
        the stop up toward (or through) the market — keeps it protective.
        """
        import math
        if px <= 0:
            return 0.0
        nd = self._px_decimals_for(px)
        factor = 10.0 ** nd
        return math.floor(px * factor) / factor

    @property
    def coin_max_leverage(self) -> int:
        return self._max_leverage

    @property
    def maint_margin_frac(self) -> float:
        return self._maint_margin_frac

    def _require_exchange(self):
        if self._exchange is None:
            raise RuntimeError("secret_key required for order actions")

    # ----------------------------------------------------------------- #
    #  READ-ONLY                                                          #
    # ----------------------------------------------------------------- #

    def mid_price(self) -> float:
        return float(self._info.all_mids()[self.cfg.coin])

    STABLES = ("USDC", "USDT", "USDT0", "USDE", "USDH", "USD")

    def account_value(self) -> float:
        """
        Tradable equity = perp accountValue + spot stablecoins. Under a Unified
        Account the perp marginSummary can read 0 while collateral sits in spot.
        Raises on a hard read failure so callers can treat it as retryable rather
        than silently sizing against a wrong (zeroed) equity.
        """
        if not self.address:
            raise ValueError("address required")
        st = self._info.user_state(self.address)
        try:
            perp = float(st["marginSummary"]["accountValue"])
        except (KeyError, TypeError, ValueError):
            perp = float(st.get("withdrawable", 0) or 0)
        spot = 0.0
        try:
            sp = self._info.spot_user_state(self.address)
            for b in sp.get("balances", []):
                if b.get("coin") in self.STABLES:
                    spot += float(b.get("total", 0) or 0)
        except Exception:
            pass   # spot leg optional; perp already captured
        return perp + spot

    def open_position(self) -> Optional[dict]:
        if not self.address:
            raise ValueError("address required")
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

    def open_orders(self) -> list:
        if not self.address:
            raise ValueError("address required")
        return self._info.open_orders(self.address)

    # ----------------------------------------------------------------- #
    #  SETUP                                                              #
    # ----------------------------------------------------------------- #

    def set_margin_mode(self):
        """Set the fallback/boot leverage + isolated/cross on the coin. Idempotent."""
        return self.update_leverage(self.cfg.leverage)

    def update_leverage(self, leverage: int):
        """Set leverage (isolated/cross per cfg) on the coin. Idempotent; per-trade."""
        self._require_exchange()
        lev = max(1, min(int(leverage), self._max_leverage))
        is_cross = not self.cfg.isolated_margin
        return self._exchange.update_leverage(lev, self.cfg.coin, is_cross)

    # ----------------------------------------------------------------- #
    #  ORDERS (long-only)                                                 #
    # ----------------------------------------------------------------- #

    def market_entry(self, size: float) -> dict:
        """
        Market BUY (open long) with an explicit slippage bound (cfg.slippage).
        Returns {"avg_px": float, "filled": float (actual), "raw": ...}.
        Raises RuntimeError if the size rounds to 0 or nothing filled.
        """
        self._require_exchange()
        size = self.round_size(size)
        if size <= 0:
            raise RuntimeError("market_entry: size rounds to 0")
        res = self._exchange.market_open(self.cfg.coin, True, size, None, self.cfg.slippage)
        if res.get("status") != "ok":
            raise RuntimeError(f"market_entry failed: {res}")
        statuses = res["response"]["data"]["statuses"]
        filled = next((s["filled"] for s in statuses if "filled" in s), None)
        if filled is None:
            raise RuntimeError(f"market_entry not filled: {statuses}")
        return {"avg_px": float(filled["avgPx"]),
                "filled": float(filled["totalSz"]),   # ACTUAL filled size (may be partial)
                "raw": res}

    def place_stop(self, trigger_price: float, size: float) -> int:
        """
        Reduce-only stop-market SELL at trigger_price. Returns the resting order id.
        Trigger is floored to a valid tick so rounding never nudges the stop up.
        """
        self._require_exchange()
        size = self.round_size(size)
        trig = self.round_stop_px(trigger_price)
        order_type = {"trigger": {"triggerPx": trig, "isMarket": True, "tpsl": "sl"}}
        res = self._exchange.order(
            self.cfg.coin, False, size, trig, order_type, reduce_only=True
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"place_stop failed: {res}")
        statuses = res["response"]["data"]["statuses"]
        st = statuses[0]
        if "resting" in st:
            return int(st["resting"]["oid"])
        if "filled" in st:               # triggered immediately (stop already breached)
            return -1
        raise RuntimeError(f"place_stop unexpected status: {st}")

    def cancel_order(self, oid: int) -> dict:
        self._require_exchange()
        return self._exchange.cancel(self.cfg.coin, oid)

    def market_close(self) -> dict:
        """Market-close the whole position for the coin."""
        self._require_exchange()
        return self._exchange.market_close(self.cfg.coin)
