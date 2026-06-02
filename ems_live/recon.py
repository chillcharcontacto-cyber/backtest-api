"""
Phase 1 recon — read-only connectivity + logic proof. NO ORDERS.

Run: python -m ems_live.recon
Optional account read: set HL_ADDRESS env to a Hyperliquid wallet address.

Proves:
  1. Binance recent candle fetch + indicators (signal source)
  2. Hyperliquid recent candle fetch + basis vs Binance (execution venue)
  3. Live entry-predicate evaluation on the latest closed bar
  4. SL adaptation: most recent crossover -> Binance SL vs Hyperliquid-adapted SL
  5. (optional) account value + open position via read-only broker
"""
import os

import numpy as np

from ems.indicators import add_emas, add_h1_emas, mark_crossovers
from .config import LiveConfig
from .decider import build_ctx, check_entry
from .feed import fetch_binance_recent, fetch_hl_recent
from .sl_adapter import adapt_sl_to_hl


def main():
    cfg = LiveConfig()
    print(f"\n=== EMS LIVE RECON (read-only) — testnet={cfg.testnet} ===\n")
    print(f"HL API: {cfg.hl_api_url}\n")

    # 1. Binance signal candles -------------------------------------------------
    print("[1] Binance signal candles")
    m30 = fetch_binance_recent(cfg.binance_symbol, "30m", cfg.lookback_m30)
    h1  = fetch_binance_recent(cfg.binance_symbol, "1h",  cfg.lookback_h1)
    m30 = mark_crossovers(add_emas(m30, cfg.ema_fast, cfg.ema_slow))
    h1  = add_h1_emas(h1, cfg.h1_trend_ema, cfg.h1_exit_ema)
    print(f"    M30 bars: {len(m30)}  last: {m30.index[-1]}")
    print(f"    H1  bars: {len(h1)}  last: {h1.index[-1]}")
    last = m30.iloc[-1]
    print(f"    last M30 close={last['close']:.2f}  ema20={last['ema_fast']:.2f}  "
          f"ema50={last['ema_slow']:.2f}  cross_up={bool(last['cross_up'])}")

    # 2. Hyperliquid candles + basis -------------------------------------------
    print("\n[2] Hyperliquid candles + basis vs Binance")
    hl = fetch_hl_recent(cfg.coin, "30m", 50, cfg.hl_api_url)
    print(f"    HL M30 bars: {len(hl)}  last: {hl.index[-1]}")
    common = m30.index.intersection(hl.index)
    if len(common) >= 3:
        diff = (m30.loc[common, "low"] - hl.loc[common, "low"]).tail(5)
        print("    Binance.low - HL.low (last 5 common bars):")
        for ts, d in diff.items():
            print(f"      {ts}  basis={d:+.2f}")

    # 3. Live entry predicate on latest closed bar -----------------------------
    print("\n[3] Entry predicate on latest closed bar")
    ctx = build_ctx(m30, h1)
    i = len(m30) - 1
    sig = check_entry(ctx, i, cfg)
    if sig is None:
        print("    No entry signal on the latest bar (expected most of the time).")
    else:
        print(f"    ENTRY SIGNAL: entry={sig.entry_price:.2f}  binance_sl={sig.sl_price:.2f}")

    # 4. SL adaptation on the most recent crossover in the window ---------------
    print("\n[4] SL adaptation demo (most recent crossover in window)")
    cross_idx = np.where(m30["cross_up"].to_numpy())[0]
    cross_idx = cross_idx[cross_idx >= 1]
    demo = None
    for ci in reversed(cross_idx):
        entry_i = ci + 1
        if entry_i >= len(m30):
            continue
        demo = check_entry(ctx, entry_i, cfg)
        if demo is not None:
            break
    if demo is None:
        print("    No qualifying crossover->entry in the current window.")
    else:
        print(f"    crossover bar: {demo.crossover_time}")
        print(f"    SL anchor bar: {demo.anchor_time}")
        print(f"    Binance SL   : {demo.sl_price:.2f}")
        try:
            hl_sl = adapt_sl_to_hl(demo.anchor_time, demo.crossover_time,
                                   cfg.coin, cfg.hl_api_url)
            print(f"    HL-adapted SL: {hl_sl:.2f}  (basis {demo.sl_price - hl_sl:+.2f})")
        except RuntimeError as e:
            print(f"    HL SL adaptation failed: {e}")

    # 5. Optional account read -------------------------------------------------
    addr = os.environ.get("HL_ADDRESS")
    print("\n[5] Account read (optional)")
    if not addr:
        print("    HL_ADDRESS not set — skipping. (read-only; safe to set anytime)")
    else:
        from .broker import LiveBroker
        broker = LiveBroker(cfg, address=addr)
        print(f"    mid price     : {broker.mid_price():.2f}")
        print(f"    account value : {broker.account_value():.2f}")
        pos = broker.open_position()
        print(f"    open position : {pos if pos else 'flat'}")

    print("\n=== recon complete — no orders placed ===\n")


if __name__ == "__main__":
    main()
