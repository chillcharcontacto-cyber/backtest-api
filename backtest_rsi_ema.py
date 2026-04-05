"""
=============================================================
  RSI + EMA Backtesting Engine  —  sin vectorbt
  Motor puro pandas/numpy — bajo consumo de memoria
=============================================================
INPUT  → JSON  (archivo o argumento --json)
OUTPUT → JSON  (stdout o archivo --output)

Uso:
  python backtest_rsi_ema.py --json-file config.json
  python backtest_rsi_ema.py --ticker BTC-USD --timeframe 1d --start 2024-01-01 --end 2025-01-01
  python backtest_rsi_ema.py --json-file config.json --output resultado.json
  python backtest_rsi_ema.py --json-file config.json --json-only
=============================================================
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

# ── Colores terminal ──────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ─────────────────────────────────────────────────────────────
# 1.  SCHEMA
# ─────────────────────────────────────────────────────────────
INPUT_SCHEMA = {
    "ticker":     {"type": str,   "default": "BTC-USD"},
    "timeframe":  {"type": str,   "default": "1d"},
    "start":      {"type": str,   "default": "2021-01-01"},
    "end":        {"type": str,   "default": datetime.today().strftime("%Y-%m-%d")},
    "rsi_period": {"type": int,   "default": 14,     "min": 2,    "max": 100},
    "ema_period": {"type": int,   "default": 50,     "min": 2,    "max": 500},
    "rsi_entry":  {"type": float, "default": 30.0,   "min": 1,    "max": 49},
    "rsi_exit":   {"type": float, "default": 70.0,   "min": 51,   "max": 99},
    "init_cash":  {"type": float, "default": 10000,  "min": 1},
    "fees":       {"type": float, "default": 0.001,  "min": 0,    "max": 0.1},
    "size":       {"type": float, "default": 0.99,   "min": 0.01, "max": 1.0},
    "slippage":   {"type": float, "default": 0.0005, "min": 0,    "max": 0.05},
}

TIMEFRAME_FREQ = {
    "1m":  "1min",  "5m":  "5min",  "15m": "15min", "30m": "30min",
    "1h":  "1h",    "2h":  "2h",    "4h":  "4h",    "90m": "90min",
    "1d":  "1d",    "5d":  "5d",    "1wk": "1W",    "1mo": "1M",   "3mo": "3M",
}

INTRADAY_TF = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "90m"}


# ─────────────────────────────────────────────────────────────
# 2.  VALIDACION
# ─────────────────────────────────────────────────────────────
def validate_and_fill(raw: dict) -> tuple:
    params = {}
    errors = []

    for field, rules in INPUT_SCHEMA.items():
        val = raw.get(field, rules["default"])
        try:
            val = rules["type"](val)
        except (ValueError, TypeError):
            errors.append(f"'{field}': no se puede convertir '{val}' a {rules['type'].__name__}")
            val = rules["default"]
        if "min" in rules and val < rules["min"]:
            errors.append(f"'{field}': {val} menor que minimo ({rules['min']})")
        if "max" in rules and val > rules["max"]:
            errors.append(f"'{field}': {val} mayor que maximo ({rules['max']})")
        params[field] = val

    if params["timeframe"] not in TIMEFRAME_FREQ:
        errors.append(f"'timeframe': '{params['timeframe']}' no valido. Opciones: {list(TIMEFRAME_FREQ.keys())}")

    try:
        d_start = datetime.strptime(params["start"], "%Y-%m-%d")
        d_end   = datetime.strptime(params["end"],   "%Y-%m-%d")
        if d_start >= d_end:
            errors.append("'start' debe ser anterior a 'end'")
        days = (d_end - d_start).days
        if params["timeframe"] in INTRADAY_TF and days > 60:
            errors.append(
                f"Timeframe '{params['timeframe']}' intradía: yfinance permite ~60 dias max. "
                f"Tu rango es {days} dias."
            )
    except ValueError as e:
        errors.append(f"Formato de fecha invalido: {e}")

    if params.get("rsi_entry", 0) >= params.get("rsi_exit", 100):
        errors.append("'rsi_entry' debe ser menor que 'rsi_exit'")

    return params, errors


# ─────────────────────────────────────────────────────────────
# 3.  INDICADORES
# ─────────────────────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────
# 4.  MOTOR DE BACKTEST PURO (sin vectorbt)
# ─────────────────────────────────────────────────────────────
def simulate(close: pd.Series, params: dict) -> dict:
    """
    Simulacion de portfolio barra a barra.
    Devuelve trades, equity_curve y metricas.
    """
    rsi = compute_rsi(close, params["rsi_period"])
    ema = compute_ema(close, params["ema_period"])

    fees     = params["fees"]
    slippage = params["slippage"]
    size     = params["size"]
    cash     = params["init_cash"]

    in_trade     = False
    entry_price  = 0.0
    entry_date   = None
    shares       = 0.0
    equity       = cash

    trades       = []
    equity_curve = []

    dates  = close.index
    prices = close.values
    rsi_v  = rsi.values
    ema_v  = ema.values

    entry_signals = 0
    exit_signals  = 0

    for i in range(1, len(prices)):
        price = prices[i]
        date  = str(dates[i])[:10]

        # Señal de entrada
        rsi_cross_up = (rsi_v[i-1] < params["rsi_entry"]) and (rsi_v[i] >= params["rsi_entry"])
        above_ema    = price > ema_v[i]
        entry_signal = rsi_cross_up and above_ema and not in_trade

        # Señal de salida
        rsi_cross_dn = (rsi_v[i-1] > params["rsi_exit"]) and (rsi_v[i] <= params["rsi_exit"])
        exit_signal  = rsi_cross_dn and in_trade

        if entry_signal:
            entry_signals += 1
            exec_price  = price * (1 + slippage)
            shares      = (cash * size) / exec_price
            cost        = shares * exec_price * (1 + fees)
            cash       -= cost
            entry_price = exec_price
            entry_date  = date
            in_trade    = True

        elif exit_signal:
            exit_signals += 1
            exec_price  = price * (1 - slippage)
            proceeds    = shares * exec_price * (1 - fees)
            cash       += proceeds
            ret_pct     = (exec_price - entry_price) / entry_price * 100
            pnl         = proceeds - (shares * entry_price * (1 + fees))
            trades.append({
                "trade_id":   len(trades) + 1,
                "entry_date": entry_date,
                "exit_date":  date,
                "return_pct": round(ret_pct, 4),
                "pnl":        round(pnl, 2),
                "status":     "win" if ret_pct > 0 else "loss",
            })
            in_trade = False
            shares   = 0.0

        # Equity barra a barra
        if in_trade:
            equity = cash + shares * price
        else:
            equity = cash

        equity_curve.append({"date": date, "value": round(equity, 2)})

    # Cerrar trade abierto al final
    if in_trade:
        price      = prices[-1]
        date       = str(dates[-1])[:10]
        exec_price = price * (1 - slippage)
        proceeds   = shares * exec_price * (1 - fees)
        cash      += proceeds
        ret_pct    = (exec_price - entry_price) / entry_price * 100
        pnl        = proceeds - (shares * entry_price * (1 + fees))
        trades.append({
            "trade_id":   len(trades) + 1,
            "entry_date": entry_date,
            "exit_date":  date,
            "return_pct": round(ret_pct, 4),
            "pnl":        round(pnl, 2),
            "status":     "win" if ret_pct > 0 else "loss",
        })
        equity_curve[-1]["value"] = round(cash, 2)

    return {
        "trades":        trades,
        "equity_curve":  equity_curve,
        "final_value":   round(cash, 2),
        "entry_signals": entry_signals,
        "exit_signals":  exit_signals,
    }


# ─────────────────────────────────────────────────────────────
# 5.  METRICAS
# ─────────────────────────────────────────────────────────────
def compute_metrics(sim: dict, params: dict, close: pd.Series) -> dict:
    trades      = sim["trades"]
    eq_curve    = sim["equity_curve"]
    init_cash   = params["init_cash"]
    final_value = sim["final_value"]

    # Returns
    strategy_return = round((final_value - init_cash) / init_cash * 100, 4)
    bh_return       = round((float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) * 100, 4)

    returns = [t["return_pct"] for t in trades]
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r <= 0]
    n       = len(trades)

    win_rate      = round(len(wins) / n * 100, 4) if n > 0 else 0.0
    avg_win       = round(float(np.mean(wins)),   4) if wins   else 0.0
    avg_loss      = round(float(np.mean(losses)), 4) if losses else 0.0
    profit_factor = round(abs(sum(wins) / sum(losses)), 4) if losses and sum(losses) != 0 else 0.0

    win_rate_dec  = win_rate / 100
    expectancy    = round((win_rate_dec * avg_win) + ((1 - win_rate_dec) * avg_loss), 4)
    avg_rr        = round(abs(avg_win / avg_loss), 4) if avg_loss != 0 else 0.0

    # Max drawdown
    eq_values = [p["value"] for p in eq_curve]
    peak      = init_cash
    max_dd    = 0.0
    for v in eq_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    max_dd = round(max_dd, 4)

    # Sharpe (diario simplificado)
    if len(eq_values) > 1:
        eq_arr  = np.array(eq_values, dtype=float)
        ret_arr = np.diff(eq_arr) / eq_arr[:-1]
        sharpe  = round(float(np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(252)), 4) if np.std(ret_arr) > 0 else 0.0
    else:
        sharpe = 0.0

    # Max consecutive losses
    max_consec = 0
    current    = 0
    for r in returns:
        if r <= 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    # Equity smoothness (R²)
    smoothness = 0.0
    if len(eq_values) > 2:
        try:
            x      = np.arange(len(eq_values), dtype=float)
            y      = np.array(eq_values, dtype=float)
            coeffs = np.polyfit(x, y, 1)
            y_hat  = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            smoothness = round(max(0.0, min(1.0, 1 - ss_res / ss_tot)), 4) if ss_tot > 0 else 0.0
        except Exception:
            smoothness = 0.0

    summary = {
        "strategy_return_pct": strategy_return,
        "buyhold_return_pct":  bh_return,
        "final_value":         final_value,
        "sharpe_ratio":        sharpe,
        "max_drawdown_pct":    max_dd,
        "win_rate_pct":        win_rate,
        "profit_factor":       profit_factor,
        "total_trades":        n,
        "bars_loaded":         len(close),
        "entry_signals":       sim["entry_signals"],
        "exit_signals":        sim["exit_signals"],
    }

    analysis = {
        "expectancy":        expectancy,
        "avg_win_pct":       avg_win,
        "avg_loss_pct":      avg_loss,
        "avg_rr":            avg_rr,
        "max_consec_losses": max_consec,
        "equity_smoothness": smoothness,
    }

    return summary, analysis


# ─────────────────────────────────────────────────────────────
# 6.  RUN BACKTEST
# ─────────────────────────────────────────────────────────────
def run_backtest(params: dict) -> dict:
    output = {
        "status":       "ok",
        "input":        {k: v for k, v in params.items() if not k.startswith("_")},
        "summary":      {},
        "trades":       [],
        "equity_curve": [],
        "analysis":     {},
        "error":        None,
    }

    try:
        import yfinance as yf

        raw = yf.download(
            params["ticker"],
            start=params["start"],
            end=params["end"],
            interval=params["timeframe"],
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"yfinance no devolvio datos para '{params['ticker']}' "
                f"en {params['timeframe']}. Verifica el ticker y el rango de fechas."
            )

        # Compatibilidad MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        close = raw["Close"].dropna()

        if len(close) < params["rsi_period"] + 10:
            raise ValueError(
                f"Muy pocas barras ({len(close)}) para RSI({params['rsi_period']}). "
                "Amplia el rango de fechas."
            )

        # Simulacion
        sim = simulate(close, params)

        if sim["entry_signals"] == 0:
            raise ValueError(
                "No se generaron senales de entrada. "
                "Prueba rsi_entry mas alto (ej. 40) o rsi_exit mas bajo (ej. 60), "
                "o amplia el rango de fechas."
            )

        # Metricas
        summary, analysis = compute_metrics(sim, params, close)

        output["summary"]      = summary
        output["trades"]       = sim["trades"]
        output["equity_curve"] = sim["equity_curve"]
        output["analysis"]     = analysis
        output["_close"]       = close

    except Exception as e:
        output["status"] = "error"
        output["error"]  = str(e)

    return output


# ─────────────────────────────────────────────────────────────
# 7.  PRINT TERMINAL
# ─────────────────────────────────────────────────────────────
def print_output(output: dict):
    params  = output.get("input", {})
    summary = output.get("summary", {})

    print(f"\n{CYAN}{BOLD}{'─'*55}")
    print(f"  RSI + EMA Backtesting Engine")
    print(f"{'─'*55}{RESET}")
    print(f"  Ticker     : {BOLD}{params.get('ticker')}{RESET}")
    print(f"  Timeframe  : {BOLD}{params.get('timeframe')}{RESET}")
    print(f"  Periodo    : {params.get('start')}  →  {params.get('end')}")
    print(f"  RSI        : period={params.get('rsi_period')}  entry<{params.get('rsi_entry')}  exit>{params.get('rsi_exit')}")
    print(f"  EMA        : period={params.get('ema_period')}")
    print(f"  Capital    : ${params.get('init_cash', 0):,.0f}")
    print(f"  Fees       : {params.get('fees', 0)*100:.2f}%   Slippage: {params.get('slippage', 0)*100:.3f}%")
    print(f"{CYAN}{'─'*55}{RESET}\n")

    if output.get("status") == "error":
        print(f"{RED}{BOLD}  ERROR: {output['error']}{RESET}\n")
        return

    print(f"  Barras cargadas  : {summary.get('bars_loaded')}")
    print(f"  Entry signals    : {summary.get('entry_signals')}")
    print(f"  Exit  signals    : {summary.get('exit_signals')}\n")

    ret   = summary.get("strategy_return_pct", 0)
    color = GREEN if ret > 0 else RED

    print(f"{BOLD}{'═'*55}")
    print(f"  RESULTADOS — {params.get('ticker')}  [{params.get('timeframe')}]")
    print(f"{'═'*55}{RESET}")
    print(f"  {'Strategy Return':<28} {color}{BOLD}{ret:+.2f}%{RESET}")
    print(f"  {'Buy-and-Hold Return':<28} {summary.get('buyhold_return_pct', 0):+.2f}%")
    print(f"  {'Final Portfolio Value':<28} ${summary.get('final_value', 0):,.2f}")
    print(f"  {'Sharpe Ratio':<28} {summary.get('sharpe_ratio', 0):.2f}")
    print(f"  {'Max Drawdown':<28} {RED}{summary.get('max_drawdown_pct', 0):.2f}%{RESET}")
    print(f"  {'Win Rate':<28} {summary.get('win_rate_pct', 0):.2f}%")
    print(f"  {'Profit Factor':<28} {summary.get('profit_factor', 0):.2f}")
    print(f"  {'Total Trades':<28} {summary.get('total_trades', 0)}")
    print(f"{BOLD}{'═'*55}{RESET}\n")

    analysis = output.get("analysis", {})
    if analysis:
        ev   = analysis.get("expectancy", 0)
        sm   = analysis.get("equity_smoothness", 0)
        ev_c = GREEN if ev > 0 else RED
        sm_c = GREEN if sm > .7 else (YELLOW if sm > .4 else RED)
        print(f"{BOLD}  ANALISIS AVANZADO{RESET}")
        print(f"  {'─'*45}")
        print(f"  {'Expectancy (EV)':<28} {ev_c}{BOLD}{ev:+.2f}%{RESET} por trade")
        print(f"  {'Avg Win':<28} {GREEN}+{analysis.get('avg_win_pct', 0):.2f}%{RESET}")
        print(f"  {'Avg Loss':<28} {RED}{analysis.get('avg_loss_pct', 0):.2f}%{RESET}")
        print(f"  {'Avg R:R':<28} {analysis.get('avg_rr', 0):.2f}")
        print(f"  {'Max Racha Perdedora':<28} {RED}{analysis.get('max_consec_losses', 0)} trades{RESET}")
        print(f"  {'Equity Smoothness':<28} {sm_c}{sm:.2f}{RESET}  (0=caos, 1=perfecto)")
        print()

    for t in output.get("trades", []):
        c = GREEN if t["status"] == "win" else RED
        print(f"  #{t['trade_id']:<3}  {t['entry_date']} → {t['exit_date']}"
              f"  {c}{t['return_pct']:>8.2f}%{RESET}  ${t['pnl']:>10.2f}  [{t['status']}]")
    print()


# ─────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RSI + EMA backtest — motor puro pandas")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--json",      type=str, metavar="JSON_STRING")
    g.add_argument("--json-file", type=str, metavar="FILE")

    p.add_argument("--ticker",     type=str,   default=None)
    p.add_argument("--timeframe",  type=str,   default=None, choices=list(TIMEFRAME_FREQ.keys()))
    p.add_argument("--start",      type=str,   default=None)
    p.add_argument("--end",        type=str,   default=None)
    p.add_argument("--rsi-period", type=int,   default=None)
    p.add_argument("--ema-period", type=int,   default=None)
    p.add_argument("--rsi-entry",  type=float, default=None)
    p.add_argument("--rsi-exit",   type=float, default=None)
    p.add_argument("--cash",       type=float, default=None)
    p.add_argument("--fees",       type=float, default=None)
    p.add_argument("--output",     type=str,   default=None)
    p.add_argument("--json-only",  action="store_true")
    p.add_argument("--no-plot",    action="store_true")
    return p.parse_args()


def build_raw_input(args) -> dict:
    if args.json:
        return json.loads(args.json)
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    cli_map = {
        "ticker": args.ticker, "timeframe": args.timeframe,
        "start": args.start, "end": args.end,
        "rsi_period": args.rsi_period, "ema_period": args.ema_period,
        "rsi_entry": args.rsi_entry, "rsi_exit": args.rsi_exit,
        "init_cash": args.cash, "fees": args.fees,
    }
    return {k: v for k, v in cli_map.items() if v is not None}


def clean_output(output: dict) -> dict:
    return {k: v for k, v in output.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    raw    = build_raw_input(args)
    params, errors = validate_and_fill(raw)

    if errors:
        err = {"status": "error", "input": raw, "summary": {}, "trades": [], "error": " | ".join(errors)}
        if args.json_only:
            print(json.dumps(err, indent=2, ensure_ascii=False))
        else:
            print(f"\n{RED}{BOLD}Errores:{RESET}")
            for e in errors:
                print(f"  • {e}")
        sys.exit(1)

    output = run_backtest(params)

    if args.json_only:
        print(json.dumps(clean_output(output), indent=2, ensure_ascii=False))
    else:
        print_output(output)

    if args.output and output["status"] == "ok":
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(clean_output(output), f, indent=2, ensure_ascii=False)
        print(f"  Resultado guardado → {args.output}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────
# 10.  FUNCION PUBLICA — importable desde API / web
# ─────────────────────────────────────────────────────────────
def backtest(config: dict) -> dict:
    """
    Motor reutilizable. Importar desde FastAPI o cualquier script.

    from backtest_rsi_ema import backtest
    resultado = backtest({"ticker": "BTC-USD", "timeframe": "1d", ...})
    """
    params, errors = validate_and_fill(config)
    if errors:
        return {"status": "error", "input": config, "summary": {}, "trades": [], "error": " | ".join(errors)}
    return clean_output(run_backtest(params))
