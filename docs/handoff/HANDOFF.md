# Handoff

## Last Session Summary

**EMS V2 → live Hyperliquid bot: Phase 0+1 foundation shipped (read-only, no orders)**

New `ems_live/` package scaffolds the live bot built on EMS V2. Order placement is
hard-stubbed (`NotImplementedError`) — no code path can trade until Phase 3.

| File | Role |
|---|---|
| `config.py` | `LiveConfig` — fixed-$ risk/trade (hot-adjustable), testnet flag, fetch windows |
| `feed.py` | recent-candle fetch: Binance (signals) + Hyperliquid (SL); drops forming bar |
| `sl_adapter.py` | Binance anchor→crossover range → Hyperliquid low = venue-correct stop |
| `decider.py` | single-bar predicates + `replay()` driver — ONE source of truth, live = backtest |
| `broker.py` | Hyperliquid wrapper; read-only methods wired, orders stubbed |
| `recon.py` | runnable read-only proof: `python -m ems_live.recon` |

**Hybrid data design (user decision):** entry/exit signals from Binance candles;
SL price adapted to Hyperliquid candles over the same anchor timestamp range.

**Parity guarantee** (`tests/test_live_parity.py`):
- `replay() == simulate()` on 8 synthetic seeds (logic equality)
- `replay() == simulate()` on real cached Binance slice: **171 trades, byte-identical**

**Live recon proven (read-only):**
- Binance + HL candle fetch, timestamps align
- Entry predicates evaluate on live bars
- SL adaptation: Binance SL 73565.73 → **HL mainnet 73528.00, basis +37.73** (tight, correct)
- Testnet HL data thin → big basis there; mainnet basis ~15–45 USD as expected

**Minimal non-breaking refactor:** `sl_finder.py` gains `find_sl_with_anchor()`
(returns SL price + anchor idx); `find_sl()` delegates — identical behavior.

Tests: **35 green** (26 prior + 9 parity). `hyperliquid-python-sdk` added.
`.gitignore` created; committed `__pycache__` untracked.
Committed + pushed (`7efa190`).

---

## Currently Working On

**Funding the Hyperliquid testnet account (blocked on user action).**

- Sizing decision: **fixed $ risk per trade**, hot-adjustable
- Venue: **testnet first**, then mainnet
- Hosting: **Render worker** (chosen, not yet set up)
- User testnet/master address: `0x18ce2b5c85827c343c35de25fc477a62c5bd6964`
  (verified read-only: mainnet value 0, testnet value 0, flat)

**Blocker:** testnet faucet (app.hyperliquid-testnet.xyz/drip) is anti-bot gated —
requires a prior **mainnet deposit on the same address**. User's address has no
mainnet history, so faucet rejects it. User is moving ~10 USDC + ~$2 ETH gas from
KuCoin → Arbitrum One → their wallet → deposit on app.hyperliquid.xyz to unlock the
faucet, then claim 1000 mock USDC, then generate the API agent key.

---

## Parked / Unfinished

**EMS live bot (next phases):**
- Phase 2: `position.py` (state + persistence + boot reconcile vs exchange),
  `runner.py` (bar-close scheduler), missed-bar catch-up on restart
- Phase 3: implement `market_entry` / `place_stop` / `market_close` on **testnet**;
  full entry→SL→exit lifecycle with fake money
- Safety guards to build before any mainnet order: `assert sl < entry`, sane
  risk-band check, absolute `max_notional` ceiling, leverage cap + isolated margin,
  stop-confirmed-or-flatten, one-position-only, max-daily-loss kill switch, dry-run mode
- Phase 4: mainnet dry-run (orders logged, not sent) → Phase 5: live tiny size

**EMS engine (pre-existing):**
- Parity check vs TradingView Pine strategy report
- Short-side extension — Samuel may have short rules, not discussed

**MCT engine (carried):**
- No-divergence test result outstanding
- `max_sweep_age_bars` not implemented
- `div_found` counter cosmetic bug
- RSI divergence 4-model logic unaudited

---

## Next Steps

1. **User funds testnet** — KuCoin → Arbitrum → wallet → HL mainnet deposit →
   claim faucet → generate API agent key → hand over testnet address + private key
2. **Verify funding** — point `recon.py` at the address, confirm testnet balance + flat
3. **Phase 2** — build `position.py` + `runner.py` (scheduler, reconcile, catch-up)
4. **Phase 3** — implement order methods on testnet, run full lifecycle with safety guards
