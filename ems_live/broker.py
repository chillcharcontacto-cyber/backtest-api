"""
Hyperliquid broker wrapper.

Read-only methods need only a public wallet ADDRESS.
Order methods need the API/agent SECRET key (trade-only; cannot withdraw).

Long-only. Stop is a reduce-only trigger-market order resting on the exchange,
so the position stays protected even if the bot process dies.

Rounding: BTC perp szDecimals=5 -> size to 5 dp; perp price decimals = 6 - szDecimals.

Resilience: every Hyperliquid API call goes through _retry(), which backs off and
retries on a transient HTTP status (429 rate-limit, 5xx gateway blip) from HL's edge
(CloudFront/nginx). A one-off 429 self-heals in-tick instead of aborting the whole bar.
"""
import time
from typing import Optional

from .config import LiveConfig


# Transient HTTP statuses from Hyperliquid's edge (CloudFront/nginx), surfaced by the
# SDK's ClientError. A 429/5xx is rejected BEFORE the matching engine, so retrying a
# READ is always safe. For WRITES we retry only on 429 (unambiguously "rate-limited,
# never processed") — a 5xx could in theory hide a lost success response, so an order
# is not blindly resent on 5xx (avoids any chance of a double fill).
_TRANSIENT_READ = frozenset({429, 500, 502, 503, 504})
_TRANSIENT_WRITE = frozenset({429})


def _http_status(exc) -> Optional[int]:
    """Best-effort HTTP status from a hyperliquid ClientError or a requests error."""
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code
    return None


def _retry(fn, statuses, tries: int = 4, base_delay: float = 0.5):
    """
    Call fn(); on a transient HTTP status (in `statuses`) retry with exponential
    backoff (0.5s, 1s, 2s). Any non-transient error — or the final attempt — raises,
    so a genuine outage still surfaces to the tick handler (which retries next bar).
    Turns a one-off 429/gateway blip into a silent self-heal instead of a skipped bar.
    """
    delay = base_delay
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries or _http_status(e) not in statuses:
                raise
            time.sleep(delay)
            delay *= 2


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
        meta = _retry(lambda: self._info.meta(), _TRANSIENT_READ)
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
    #  READ-ONLY  (retry on 429 + 5xx — idempotent)                       #
    # ----------------------------------------------------------------- #

    def mid_price(self) -> float:
        mids = _retry(lambda: self._info.all_mids(), _TRANSIENT_READ)
        return float(mids[self.cfg.coin])

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
        st = _retry(lambda: self._info.user_state(self.address), _TRANSIENT_READ)
        try:
            perp = float(st["marginSummary"]["accountValue"])
        except (KeyError, TypeError, ValueError):
            perp = float(st.get("withdrawable", 0) or 0)
        spot = 0.0
        try:
            sp = _retry(lambda: self._info.spot_user_state(self.address), _TRANSIENT_READ)
            for b in sp.get("balances", []):
                if b.get("coin") in self.STABLES:
                    spot += float(b.get("total", 0) or 0)
        except Exception:
            pass   # spot leg optional; perp already captured
        return perp + spot

    def open_position(self) -> Optional[dict]:
        if not self.address:
            raise ValueError("address required")
        state = _retry(lambda: self._info.user_state(self.address), _TRANSIENT_READ)
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
        return _retry(lambda: self._info.open_orders(self.address), _TRANSIENT_READ)

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
        return _retry(lambda: self._exchange.update_leverage(lev, self.cfg.coin, is_cross),
                      _TRANSIENT_WRITE)

    # ----------------------------------------------------------------- #
    #  ORDERS (long-only) — retry on 429 only (never risk a double fill)  #
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
        res = _retry(
            lambda: self._exchange.market_open(self.cfg.coin, True, size, None, self.cfg.slippage),
            _TRANSIENT_WRITE)
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
        res = _retry(
            lambda: self._exchange.order(
                self.cfg.coin, False, size, trig, order_type, reduce_only=True),
            _TRANSIENT_WRITE)
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
        return _retry(lambda: self._exchange.cancel(self.cfg.coin, oid), _TRANSIENT_WRITE)

    def market_close(self) -> dict:
        """Market-close the whole position for the coin."""
        self._require_exchange()
        return _retry(lambda: self._exchange.market_close(self.cfg.coin), _TRANSIENT_WRITE)
