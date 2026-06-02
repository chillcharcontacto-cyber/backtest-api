"""
EMS Live — turns the EMS V2 backtest engine into a live Hyperliquid trading bot.

Signal source : Binance spot candles (crossover, H1 trend filter, H1 EMA100 exit)
SL placement  : adapted to Hyperliquid candles (stop price = HL low over the same
                anchor->crossover timestamp range the Binance signal identified)
Execution     : Hyperliquid BTC perp (testnet first)
Sizing        : fixed $ risk per trade (hot-adjustable), size = risk_usd / (entry - sl)

Parity guarantee: ems_live.decider.replay() reproduces ems.engine.simulate() trade
for trade on identical data (see tests/test_live_parity.py). The live runner and the
backtest share one set of entry/exit predicates -> no logic drift.
"""
