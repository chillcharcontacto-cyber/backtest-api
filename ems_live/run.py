"""
EMS live bot entrypoint.

Loads Hyperliquid credentials from .env (local) or real environment variables
(Render dashboard), builds the broker + state store, and runs the bar-close loop.

Usage:
  python -m ems_live.run          # run forever (Render worker)
  python -m ems_live.run once     # single boot+tick then exit (local smoke test)

Environment:
  HL_MASTER_ADDRESS   (required)  master wallet address
  HL_AGENT_KEY        (required to place orders; omit for read-only dry runs)
  HL_TESTNET          true/false  default true
  EMS_DRY_RUN         true/false  default TRUE (safe — computes+logs, no orders)
  EMS_RISK_USD        float       default 20

Real env vars take precedence over .env (Render has no .env file).
"""
import os
import sys

from .config import LiveConfig
from .position import PositionStore
from .broker import LiveBroker
from .runner import run_forever, boot_reconcile, tick


def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())   # env wins over .env


def _bool(val, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def build():
    load_env()
    master = os.environ.get("HL_MASTER_ADDRESS")
    if not master:
        raise SystemExit("HL_MASTER_ADDRESS is required")
    agent    = os.environ.get("HL_AGENT_KEY")            # None -> read-only
    testnet  = _bool(os.environ.get("HL_TESTNET"), True)
    dry_run  = _bool(os.environ.get("EMS_DRY_RUN"), True)  # SAFE default
    risk     = float(os.environ.get("EMS_RISK_USD", "20"))
    max_dd_r = float(os.environ.get("EMS_MAX_DAILY_LOSS_R", "0"))
    max_dd_n = int(os.environ.get("EMS_MAX_DAILY_LOSSES", "10"))
    status_mode = os.environ.get("EMS_STATUS_MODE", "steps")
    state_path = os.environ.get("EMS_STATE_PATH", "ems_live_state.json")

    # --- sizing / leverage / guards (equity-relative; tunable per venue) ---
    max_lev      = int(os.environ.get("EMS_MAX_LEVERAGE", "0"))          # 0 = coin max
    lev_fallback = int(os.environ.get("EMS_LEVERAGE", "3"))
    buffer_frac  = float(os.environ.get("EMS_MARGIN_BUFFER_FRAC", "0.90"))
    liq_mult     = float(os.environ.get("EMS_LIQ_SAFETY_MULT", "1.5"))
    max_band     = float(os.environ.get("EMS_MAX_RISK_BAND_PCT", "15"))
    min_risk     = float(os.environ.get("EMS_MIN_RISK_PCT", "0.1"))
    slippage     = float(os.environ.get("EMS_SLIPPAGE", "0.01"))
    min_notional = float(os.environ.get("EMS_HL_MIN_NOTIONAL", "10"))
    max_notional = float(os.environ.get("EMS_MAX_NOTIONAL_USD", "0"))    # 0 = off
    isolated     = _bool(os.environ.get("EMS_ISOLATED_MARGIN"), True)

    cfg = LiveConfig(testnet=testnet, dry_run=dry_run, risk_usd=risk,
                     max_daily_loss_r=max_dd_r, max_daily_losses=max_dd_n,
                     status_mode=status_mode, state_path=state_path,
                     leverage=lev_fallback, max_leverage=max_lev,
                     margin_buffer_frac=buffer_frac, liq_safety_mult=liq_mult,
                     max_risk_band_pct=max_band, min_risk_pct=min_risk,
                     slippage=slippage, hl_min_notional_usd=min_notional,
                     max_notional_usd=max_notional, isolated_margin=isolated)
    store = PositionStore(cfg.state_path)
    broker = LiveBroker(cfg, address=master, secret_key=agent)

    print(f"[run] testnet={cfg.testnet} dry_run={cfg.dry_run} risk_usd={cfg.risk_usd} "
          f"strategy={cfg.strategy_name} h4={cfg.h4_ema_fast}/{cfg.h4_ema_slow} "
          f"auto-lev(max={cfg.max_leverage or 'coin'},buf={cfg.margin_buffer_frac},liqx={cfg.liq_safety_mult}) "
          f"kill={cfg.max_daily_losses}losses/{cfg.max_daily_loss_r}R state={cfg.state_path} "
          f"agent={'yes' if agent else 'no (read-only)'}")
    return cfg, store, broker


def main():
    cfg, store, broker = build()
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        boot_reconcile(cfg, store, broker)
        tick(cfg, store, broker)
        print("[run] single tick complete (once mode)")
        return
    run_forever(cfg, store, broker)


if __name__ == "__main__":
    main()
