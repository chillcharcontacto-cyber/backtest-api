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

    # --- indicators (must match EMS V3) ---
    ema_fast:       int   = 20
    ema_slow:       int   = 50
    h1_trend_ema:   int   = 50
    h1_exit_ema:    int   = 100
    min_risk_pct:   float = 0.1          # 0.1 = 0.1% min stop distance

    # --- H4 confluence filter (V3 — the locked final model) ---
    h4_filter:      bool  = True         # enter only when H4 EMA20 > H4 EMA100
    h4_ema_fast:    int   = 20
    h4_ema_slow:    int   = 100          # robustness-locked period

    # Live has no warmup gate (EMAs seeded by lookback fetch); kept for decider parity
    warmup_bars:    int   = 0

    # --- sizing / risk ---
    risk_usd:       float = 20.0         # fixed $ risk per trade (change anytime)
    taker_fee:      float = 0.00045      # HL perp base taker (~0.045%/side) for fee estimate

    # --- sizing / leverage (equity-relative; per-trade auto-leverage) ---
    # size is ALWAYS risk_usd/(entry-sl) — risk is fixed. Leverage adapts so the
    # position fits margin while keeping liquidation beyond the structural stop.
    leverage:           int   = 3        # fallback/boot leverage if equity read fails
    max_leverage:       int   = 0        # 0 = use the coin's exchange max (from meta)
    margin_buffer_frac: float = 0.90     # use at most this fraction of equity as margin
    liq_safety_mult:    float = 1.5      # liquidation must be >= stop_dist * this beyond entry
    hl_min_notional_usd: float = 10.0    # Hyperliquid minimum order value
    slippage:           float = 0.01     # explicit market-order slippage bound (1%)
    isolated_margin:    bool  = True     # isolated so one trade can't bleed the account

    # --- safety guards ---
    max_notional_usd:   float = 0.0      # optional absolute notional ceiling (0 = OFF; margin-fit governs)
    max_risk_band_pct:  float = 15.0     # reject if stop distance > this % of entry
    max_daily_losses:   int   = 10       # halt new entries after this many losing trades/day (0 disables)
    max_daily_loss_r:   float = 0.0      # also halt if today's realized R <= -this (0 disables)
    dry_run:            bool  = True     # compute+log orders, DO NOT send (flip in Phase 3)

    # --- scheduler ---
    poll_buffer_sec:    int   = 15       # wait this long after bar close before fetching
    heartbeat_sec:      int   = 60       # liveness ping cadence between bars (hang detection)

    # --- venue ---
    testnet:        bool  = True         # ALWAYS start true; flip only when proven

    # --- data fetch windows (bars pulled each poll for EMA convergence) ---
    lookback_m30:   int   = 1200         # >= 500 needed for EMA50 to converge
    lookback_h1:    int   = 800          # >= 200 needed for EMA100 to converge

    # --- monitoring ---
    status_mode:    str   = "steps"      # "steps" (H4→H1→M30 ladder + daily heartbeat) | "change" | "always" | "off"

    # --- persistence / labels ---
    state_path:     str   = "ems_live_state.json"
    strategy_name:  str   = "EMA-Cross-H4F"

    @property
    def hl_api_url(self) -> str:
        return ("https://api.hyperliquid-testnet.xyz"
                if self.testnet else
                "https://api.hyperliquid.xyz")
