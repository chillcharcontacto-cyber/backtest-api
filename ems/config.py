from dataclasses import dataclass


@dataclass
class Config:
    symbol:        str   = "BTCUSDT"
    ema_fast:      int   = 20
    ema_slow:      int   = 50
    h1_trend_ema:  int   = 50
    h1_exit_ema:   int   = 100
    h4_ema_fast:   int   = 20        # H4 confluence filter (V3)
    h4_ema_slow:   int   = 50        # H4 confluence filter (V3)
    min_risk_pct:  float = 0.1       # 0.1 = 0.1% of entry price
    warmup_bars:   int   = 500       # M30 bars to skip (EMA seeding protection)
    h4_filter:     bool  = False     # Enable H4 EMA confluence filter (V3)
    strategy_name: str   = "EMA-Cross"   # CSV strategy column
    start:         str   = "2017-08-17"
    end:           str   = "2026-05-12"
    output_csv:    str   = "trades.csv"
    data_dir:      str   = "data"
