import argparse

from .config import Config
from .data import fetch_ohlcv, fetch_ohlcv_bitstamp
from .indicators import add_emas, add_h1_emas, mark_crossovers, build_h4, add_h4_emas
from .engine import simulate
from .output import trades_to_csv

# Default symbols per exchange
EXCHANGE_DEFAULTS = {
    "binance":  "BTCUSDT",
    "bitstamp": "btcusd",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="EMS System BTC M30 Backtest")
    parser.add_argument("--start",     default="2017-08-17",  help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",       default="2026-05-12",  help="End date (YYYY-MM-DD)")
    parser.add_argument("--output",    default="trades.csv",  help="Output CSV path")
    parser.add_argument("--data-dir",  default="data",        help="Parquet cache directory")
    parser.add_argument("--exchange",  default="binance",
                        choices=["binance", "bitstamp"],      help="Data source exchange")
    parser.add_argument("--symbol",    default=None,
                        help="Override symbol (default: BTCUSDT for binance, btcusd for bitstamp)")
    parser.add_argument("--h4-filter", action="store_true",   help="Enable H4 EMA confluence filter (V3)")
    args = parser.parse_args()

    exchange   = args.exchange
    symbol     = args.symbol or EXCHANGE_DEFAULTS[exchange]
    use_h4     = args.h4_filter
    strat_name = "EMA-Cross-H4F" if use_h4 else "EMA-Cross"

    cfg = Config(
        symbol        = symbol,
        start         = args.start,
        end           = args.end,
        output_csv    = args.output,
        data_dir      = args.data_dir,
        h4_filter     = use_h4,
        strategy_name = strat_name,
    )

    v_label = "V3" if use_h4 else "V2"
    print(f"\n=== EMS Backtest {v_label}: {symbol} M30  {cfg.start} -> {cfg.end}  [{exchange}] ===\n")

    # --- fetch data ---
    def _fetch(interval: str):
        if exchange == "bitstamp":
            return fetch_ohlcv_bitstamp(symbol, interval, cfg.start, cfg.end, cfg.data_dir)
        return fetch_ohlcv(symbol, interval, cfg.start, cfg.end, cfg.data_dir)

    print("[1/4] Fetching M30 data ...")
    m30_raw = _fetch("30m")

    print("[2/4] Fetching H1 data ...")
    h1_raw  = _fetch("1h")

    print(f"\nM30 bars : {len(m30_raw):,}")
    print(f"H1 bars  : {len(h1_raw):,}\n")

    # --- indicators ---
    print("[3/4] Computing indicators ...")
    m30 = add_emas(m30_raw, cfg.ema_fast, cfg.ema_slow)
    m30 = mark_crossovers(m30)
    h1  = add_h1_emas(h1_raw, cfg.h1_trend_ema, cfg.h1_exit_ema)

    # H4 (V3 only)
    h4 = None
    if use_h4:
        print("        Building H4 from H1 ...")
        h4_raw = build_h4(h1_raw)
        h4     = add_h4_emas(h4_raw, cfg.h4_ema_fast, cfg.h4_ema_slow)
        print(f"        H4 bars : {len(h4):,}")

    # --- simulate ---
    print("[4/4] Running simulation ...\n")
    trades = simulate(m30, h1, cfg, h4=h4)

    # --- summary ---
    print(f"{'='*50}")
    print(f"  Total trades : {len(trades)}")

    if trades:
        rs        = [t.r_multiple for t in trades]
        winners   = [r for r in rs if r > 0]
        losers    = [r for r in rs if r <= 0]
        wr        = len(winners) / len(rs) * 100
        avg_r     = sum(rs) / len(rs)
        total_r   = sum(rs)
        gp        = sum(winners) if winners else 0.0
        gl        = abs(sum(losers)) if losers else 0.0
        pf        = gp / gl if gl > 0 else float("inf")
        by_reason = {}
        for t in trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

        print(f"  Win rate     : {wr:.1f}%")
        print(f"  Avg R        : {avg_r:.3f}")
        print(f"  Total R      : {total_r:.2f}")
        print(f"  Profit factor: {pf:.2f}")
        print(f"  Exit reasons : {by_reason}")
        print(f"{'='*50}\n")

    trades_to_csv(trades, cfg.output_csv)
    print(f"Trades written -> {cfg.output_csv}")


if __name__ == "__main__":
    main()
