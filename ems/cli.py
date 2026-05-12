import argparse

from .config import Config
from .data import fetch_ohlcv
from .indicators import add_emas, add_h1_emas, mark_crossovers
from .engine import simulate
from .output import trades_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="EMS System BTCUSDT M30 Backtest")
    parser.add_argument("--start",    default="2017-08-17", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",      default="2026-05-12", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output",   default="trades.csv", help="Output CSV path")
    parser.add_argument("--data-dir", default="data",       help="Parquet cache directory")
    parser.add_argument("--symbol",   default="BTCUSDT",    help="Binance symbol")
    args = parser.parse_args()

    cfg = Config(
        symbol    = args.symbol,
        start     = args.start,
        end       = args.end,
        output_csv = args.output,
        data_dir  = args.data_dir,
    )

    print(f"\n=== EMS Backtest: {cfg.symbol} M30  {cfg.start} -> {cfg.end} ===\n")

    # --- fetch data ---
    print("[1/4] Fetching M30 data ...")
    m30_raw = fetch_ohlcv(cfg.symbol, "30m", cfg.start, cfg.end, cfg.data_dir)

    print("[2/4] Fetching H1 data ...")
    h1_raw  = fetch_ohlcv(cfg.symbol, "1h",  cfg.start, cfg.end, cfg.data_dir)

    print(f"\nM30 bars : {len(m30_raw):,}")
    print(f"H1 bars  : {len(h1_raw):,}\n")

    # --- indicators ---
    print("[3/4] Computing indicators ...")
    m30 = add_emas(m30_raw, cfg.ema_fast, cfg.ema_slow)
    m30 = mark_crossovers(m30)
    h1  = add_h1_emas(h1_raw, cfg.h1_trend_ema, cfg.h1_exit_ema)

    # --- simulate ---
    print("[4/4] Running simulation ...\n")
    trades = simulate(m30, h1, cfg)

    # --- summary ---
    print(f"{'='*45}")
    print(f"  Total trades : {len(trades)}")

    if trades:
        rs       = [t.r_multiple for t in trades]
        winners  = [r for r in rs if r > 0]
        losers   = [r for r in rs if r <= 0]
        wr       = len(winners) / len(rs) * 100
        avg_r    = sum(rs) / len(rs)
        total_r  = sum(rs)
        gp       = sum(winners) if winners else 0.0
        gl       = abs(sum(losers)) if losers else 0.0
        pf       = gp / gl if gl > 0 else float("inf")
        by_reason = {}
        for t in trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

        print(f"  Win rate     : {wr:.1f}%")
        print(f"  Avg R        : {avg_r:.3f}")
        print(f"  Total R      : {total_r:.2f}")
        print(f"  Profit factor: {pf:.2f}")
        print(f"  Exit reasons : {by_reason}")
        print(f"{'='*45}\n")

    trades_to_csv(trades, cfg.output_csv)
    print(f"Trades written -> {cfg.output_csv}")


if __name__ == "__main__":
    main()
