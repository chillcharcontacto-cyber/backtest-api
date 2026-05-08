# TradingEdgeLabs — Backtest API

## Project

Backtesting SaaS for the **MCT (Multi Candle Trap)** strategy. Live at tradingedgelabs.com.
Deployed on Render (auto-deploy from `main`). Data: yfinance (stocks/crypto) + Twelve Data (forex, env var `TWELVEDATA_API_KEY`).

**Current goal:** verify the coded engine matches manual trades from the journal → via Pine Script export → load in TradingView → compare.

---

## Architecture

### Two engines

| Endpoint | File | Engine |
|---|---|---|
| `POST /backtest` | `backtest_rsi_ema.py` | Legacy RSI + EMA — simple, standalone |
| `POST /strategy` | `engine.py` | MCT sequential protocol — the real system |
| `POST /export/pinescript` | `api.py` + `engine.py` | Runs MCT + returns Pine Script v6 |

### MCT sequential protocol (`engine.py::simulate()`)

The entry logic is **hardcoded** — entry_confirmations don't control what fires, they only supply parameters:

```
Liquidity Sweep (SS→SE)
  → RSI Divergence in sweep context (4 models: M1 Regular, M2 Sweep, M3 Multiple, M4 Extended)
    → BOS (Break of Structure) — direction must match sweep
      → Session filter (if session in entry_confirmations)
        → SL validation (if sl_filter in entry_confirmations)
          → RR >= min_rr
            → ENTRY at BOS candle close
```

Exit: SL hit / MCT TP (OL, or Fib 0.71 if RR ≥ rr_threshold) / Cut Early

### Supporting modules

- `market_structure_bos.py` — BOS detector (LL/LH/HH/HL with circular validation)
- `cut_early.py` — Cut Early Protocol (MSB Against → pullback → Cut Level state machine: IDLE → MSB_PENDING → CUT_ACTIVE)
- `liquidity_channels.py` — pivot-based liquidity sweep detection (SS, SE, OL)
- `rsi_divergence.py` — 4 RSI divergence models

---

## Key design facts

- **entry_confirmations from the UI are largely ignored** in `simulate()`. Only these indicators affect behavior: `session`, `sl_filter`, `rr_mct`, `bos`/`liquidity` (for `liq_strength`), `mct_exit`, `cut_early`.
- `RSI`, `EMA`, `SMA`, `MACD`, etc. chosen in the builder don't affect MCT entry logic — `precompute_indicators()` exists but is not called from `simulate()`.
- The SL filter (`sl_filter`) is forex-specific (pip-based). It only runs when `sl_filter` is explicitly in entry_confirmations.

---

## Full MCT JSON for `/strategy`

```json
{
  "market": {
    "ticker": "EURUSD",
    "timeframe": "5m",
    "start": "2026-04-01",
    "end": "2026-04-30"
  },
  "risk": {
    "capital": 10000,
    "fees": 0.0,
    "slippage": 0.0005,
    "size": 0.99
  },
  "strategy": {
    "entry_confirmations": [
      {"indicator": "bos",       "params": {"liq_strength": 25}, "condition": "is_true"},
      {"indicator": "session",   "params": {"hour_from": 7, "hour_to": 11}, "condition": "is_true"},
      {"indicator": "sl_filter", "params": {"min_pips": 4.5, "max_pips": 35.0, "pip_size": 0.0001}, "condition": "is_true"},
      {"indicator": "rr_mct",    "params": {"min_rr": 1.2, "pip_size": 0.0001}, "condition": "is_true"}
    ],
    "exit_confirmations": [
      {"indicator": "mct_exit",  "params": {"rr_threshold": 3.0, "pip_size": 0.0001}, "condition": "is_true"},
      {"indicator": "cut_early", "params": {}, "condition": "is_true"}
    ]
  }
}
```

---

## Bug fixed (2026-05-08)

**`engine.py::simulate()` — SL filter was always active with wrong pip_size**

- `pip_size` default was `0.00001` (should be `0.0001`, matching INDICATOR_CATALOG)
- The SL pip-range check ran unconditionally — killed all trades on stocks/crypto and most forex setups
- **Fix:** corrected `pip_size` default to `0.0001` in all three places; SL pip filter now only runs when `sl_filter` is explicitly in `entry_confirmations`

---

## Deploy workflow

- Render auto-deploys from `main` branch on GitHub
- `VALID_API_KEYS` env var on Render controls auth (comma-separated keys, format `TEL-xxxx`)
- Generate a new key: `python -c "import secrets; print('TEL-' + secrets.token_hex(16))"`
