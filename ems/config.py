from dataclasses import dataclass


@dataclass
class Config:
    symbol:       str   = "BTCUSDT"
    ema_fast:     int   = 20
    ema_slow:     int   = 50
    h1_trend_ema: int   = 50
    h1_exit_ema:  int   = 100
    min_risk_pct: float = 0.1    # 0.1 means 0.1%
    sl_lookback:  int   = 50
    start:        str   = "2017-08-17"
    end:          str   = "2026-05-12"
    output_csv:   str   = "trades.csv"
    data_dir:     str   = "data"
