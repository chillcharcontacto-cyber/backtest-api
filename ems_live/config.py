from dataclasses import dataclass


@dataclass
class LiveConfig:
    """
    Live bot configuration.

    Sizing is fixed-$ risk per trade (hot-adjustable): on each entry,
        position_size = risk_usd / (entry_price - sl_price)
    so a wider stop -> smaller size, keeping dollar risk constant at risk_usd.
    """
    # --- instrument ---
    coin:           str   = "BTC"        # Hyperliquid perp coin
    binance_symbol: str   = "BTCUSDT"    # Binance signal symbol

    # --- indicators (must match EMS V2) ---
    ema_fast:       int   = 20
    ema_slow:       int   = 50
    h1_trend_ema:   int   = 50
    h1_exit_ema:    int   = 100
    min_risk_pct:   float = 0.1          # 0.1 = 0.1% min stop distance

    # Live has no warmup gate (EMAs seeded by lookback fetch); kept for decider parity
    warmup_bars:    int   = 0

    # --- sizing / risk ---
    risk_usd:       float = 20.0         # fixed $ risk per trade (change anytime)

    # --- safety guards (orders refused if violated) ---
    max_notional_usd:   float = 500.0    # absolute ceiling on position notional
    max_risk_band_pct:  float = 15.0     # reject if stop distance > this % of entry
    leverage:           int   = 3        # leverage cap on the coin
    isolated_margin:    bool  = True     # isolated so one trade can't bleed the account
    dry_run:            bool  = True     # compute+log orders, DO NOT send (flip in Phase 3)

    # --- scheduler ---
    poll_buffer_sec:    int   = 15       # wait this long after bar close before fetching

    # --- venue ---
    testnet:        bool  = True         # ALWAYS start true; flip only when proven

    # --- data fetch windows (bars pulled each poll for EMA convergence) ---
    lookback_m30:   int   = 1200         # >= 500 needed for EMA50 to converge
    lookback_h1:    int   = 800          # >= 200 needed for EMA100 to converge

    # --- persistence / labels ---
    state_path:     str   = "ems_live_state.json"
    strategy_name:  str   = "EMA-Cross"

    @property
    def hl_api_url(self) -> str:
        return ("https://api.hyperliquid-testnet.xyz"
                if self.testnet else
                "https://api.hyperliquid.xyz")
