"""
Entry-timing parity: the LIVE entry (crossover on the just-closed bar i, enter at
market ~ next open) must select EXACTLY the trades the backtest takes at open[i+1].

check_entry_live(ctx, i) should agree with check_entry(ctx, i+1) on:
  - whether a signal fires,
  - the structural SL price, the crossover time, and the anchor time.
Only the entry_price differs (close[i] estimate vs backtest open[i+1]).

This proves the fix changes WHEN/at-what-price the live bot acts (on time, not a bar
late) WITHOUT changing WHICH trades it takes.
"""
import os

import pandas as pd
import pytest

from ems.config import Config
from ems.indicators import (add_emas, add_h1_emas, mark_crossovers,
                            build_h4, add_h4_emas)
from ems_live.decider import build_ctx, check_entry, check_entry_live

DATA_M30 = "data/BTCUSDT_30m.parquet"
DATA_H1 = "data/BTCUSDT_1h.parquet"
THIRTY = pd.Timedelta(minutes=30)


def _ctx_v3():
    m30 = mark_crossovers(add_emas(pd.read_parquet(DATA_M30).iloc[:30000], 20, 50))
    h1r = pd.read_parquet(DATA_H1).iloc[:15000]
    h1 = add_h1_emas(h1r, 50, 100)
    h4 = add_h4_emas(build_h4(h1r), 20, 100)
    return build_ctx(m30, h1, h4)


@pytest.mark.skipif(
    not (os.path.exists(DATA_M30) and os.path.exists(DATA_H1)),
    reason="Binance parquet cache not present",
)
def test_live_entry_selects_same_trades_as_backtest_next_bar():
    ctx = _ctx_v3()
    cfg = Config(h4_filter=True, h4_ema_fast=20, h4_ema_slow=100,
                 warmup_bars=500, strategy_name="EMA-Cross-H4F")
    n = len(ctx.m30_times)
    live_hits = bt_hits = agree = 0
    for i in range(1, n - 1):
        live = check_entry_live(ctx, i, cfg, THIRTY)
        bt = check_entry(ctx, i + 1, cfg)      # backtest: crossover at (i+1)-1 = i
        assert (live is None) == (bt is None), f"presence mismatch at i={i}"
        if live is not None:
            live_hits += 1
            assert live.sl_price == bt.sl_price
            assert live.crossover_time == bt.crossover_time == ctx.m30_times[i]
            assert live.anchor_time == bt.anchor_time
            agree += 1
        if bt is not None:
            bt_hits += 1
    assert live_hits > 20, "slice should produce entries to compare"
    assert live_hits == bt_hits == agree
