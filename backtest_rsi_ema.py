"""
=============================================================
  RSI + EMA Backtesting Engine  —  powered by vectorbt
=============================================================
INPUT  → JSON  (archivo, stdin, o argumento --json)
OUTPUT → JSON  (stdout o archivo --output)

Tres modos de uso:
  1. JSON directo:
       python backtest_rsi_ema.py --json '{"ticker":"EURUSD=X","timeframe":"1h",...}'

  2. Archivo JSON:
       python backtest_rsi_ema.py --json-file config.json

  3. CLI clásico (compatibilidad):
       python backtest_rsi_ema.py --ticker EURUSD=X --timeframe 1h ...

Guardar resultado JSON:
       python backtest_rsi_ema.py --json-file config.json --output resultado.json

Modo API (solo JSON puro, sin decoración):
       python backtest_rsi_ema.py --json-file config.json --json-only

=============================================================
SCHEMA DE INPUT (todos los campos con sus defaults):
{
  "ticker":      "BTC-USD",       # Activo. Forex: añadir =X (ej. EURUSD=X)
  "timeframe":   "1d",            # 1m 5m 15m 30m 1h 2h 4h 90m 1d 5d 1wk 1mo
  "start":       "2021-01-01",    # Fecha inicio YYYY-MM-DD
  "end":         "2025-03-30",    # Fecha fin   YYYY-MM-DD
  "rsi_period":  14,              # Periodo RSI
  "ema_period":  50,              # Periodo EMA
  "rsi_entry":   30,              # Nivel sobreventa (entry)
  "rsi_exit":    70,              # Nivel sobrecompra (exit)
  "init_cash":   10000,           # Capital inicial USD
  "fees":        0.001,           # Comision (0.001 = 0.1%)
  "size":        0.99,            # Fraccion del equity por trade
  "slippage":    0.0005           # Slippage (0.0005 = 0.05%)
}

SCHEMA DE OUTPUT:
{
  "status": "ok" | "error",
  "input":  { ...params usados },
  "summary": {
    "strategy_return_pct":   float,
    "buyhold_return_pct":    float,
    "final_value":           float,
    "sharpe_ratio":          float,
    "max_drawdown_pct":      float,
    "win_rate_pct":          float,
    "profit_factor":         float,
    "total_trades":          int,
    "bars_loaded":           int,
    "entry_signals":         int,
    "exit_signals":          int
  },
  "trades": [
    {
      "trade_id":    int,
      "entry_date":  "YYYY-MM-DD",
      "exit_date":   "YYYY-MM-DD",
      "return_pct":  float,
      "pnl":         float,
      "status":      "win" | "loss"
    }
  ],
  "error": null | "mensaje de error",
  "equity_curve": [
    {"date": "YYYY-MM-DD", "value": float}   // un punto por barra
  ],
  "analysis": {
    "expectancy":        float,   # EV medio por trade (% ponderado por win/loss rate)
    "avg_win_pct":       float,   # retorno medio trades ganadores
    "avg_loss_pct":      float,   # retorno medio trades perdedores
    "avg_rr":            float,   # ratio reward/risk medio
    "max_consec_losses": int,     # maxima racha de perdidas consecutivas
    "equity_smoothness": float    # suavidad curva equity 0-1 (1=perfecta)
  }
}
=============================================================
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from datetime import datetime

# ── Colores terminal ──────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ─────────────────────────────────────────────────────────────
# 1.  SCHEMA — defaults y validacion
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
# 2.  VALIDACION DE INPUT
# ─────────────────────────────────────────────────────────────
def validate_and_fill(raw: dict) -> tuple:
    """
    Recibe dict raw (puede tener campos faltantes o incorrectos).
    Devuelve (params_completos, lista_de_errores).
    """
    params = {}
    errors = []

    for field, rules in INPUT_SCHEMA.items():
        val = raw.get(field, rules["default"])
        try:
            val = rules["type"](val)
        except (ValueError, TypeError):
            errors.append(
                f"'{field}': no se puede convertir '{val}' a {rules['type'].__name__}"
            )
            val = rules["default"]

        if "min" in rules and val < rules["min"]:
            errors.append(f"'{field}': {val} es menor que el minimo ({rules['min']})")
        if "max" in rules and val > rules["max"]:
            errors.append(f"'{field}': {val} es mayor que el maximo ({rules['max']})")

        params[field] = val

    # Timeframe valido
    if params["timeframe"] not in TIMEFRAME_FREQ:
        errors.append(
            f"'timeframe': '{params['timeframe']}' no valido. "
            f"Opciones: {list(TIMEFRAME_FREQ.keys())}"
        )

    # Fechas
    try:
        d_start = datetime.strptime(params["start"], "%Y-%m-%d")
        d_end   = datetime.strptime(params["end"],   "%Y-%m-%d")
        if d_start >= d_end:
            errors.append("'start' debe ser anterior a 'end'")
        days = (d_end - d_start).days
        if params["timeframe"] in INTRADAY_TF and days > 60:
            errors.append(
                f"Timeframe '{params['timeframe']}' intradía: "
                f"yfinance solo permite ~60 dias historicos. "
                f"Tu rango es {days} dias — reduce el rango a menos de 60 dias."
            )
    except ValueError as e:
        errors.append(f"Formato de fecha invalido: {e}")

    # rsi_entry < rsi_exit
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
# 4.  SENALES
# ─────────────────────────────────────────────────────────────
def generate_signals(close, rsi_period, ema_period, rsi_entry, rsi_exit):
    rsi = compute_rsi(close, rsi_period)
    ema = compute_ema(close, ema_period)

    # Entry: RSI cruza hacia arriba sobreventa + precio > EMA
    entries = (rsi.shift(1) < rsi_entry) & (rsi >= rsi_entry) & (close > ema)

    # Exit: RSI cruza hacia abajo sobrecompra
    exits = (rsi.shift(1) > rsi_exit) & (rsi <= rsi_exit)

    return entries, exits



# ─────────────────────────────────────────────────────────────
# 5.  MOTOR DE ANALISIS — metricas avanzadas
# ─────────────────────────────────────────────────────────────
def compute_analysis(trades_list: list, equity_curve: "pd.Series") -> dict:
    """
    Calcula metricas avanzadas a partir de la lista de trades y la curva de equity.

    Metricas:
      expectancy        → ganancia media esperada por trade (EV)
      profit_factor     → suma ganancias / suma perdidas
      avg_win_pct       → retorno medio de trades ganadores
      avg_loss_pct      → retorno medio de trades perdedores
      avg_rr            → ratio medio reward/risk (avg_win / |avg_loss|)
      max_consec_losses → maxima racha de perdidas consecutivas
      equity_smoothness → R² de la curva de equity vs linea ideal (0-1, 1=perfecta)
    """
    if not trades_list:
        return {
            "expectancy":         0.0,
            "avg_win_pct":        0.0,
            "avg_loss_pct":       0.0,
            "avg_rr":             0.0,
            "max_consec_losses":  0,
            "equity_smoothness":  0.0,
        }

    returns = [t["return_pct"] for t in trades_list]
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r <= 0]

    avg_win  = round(float(np.mean(wins)),   4) if wins   else 0.0
    avg_loss = round(float(np.mean(losses)), 4) if losses else 0.0

    win_rate  = len(wins) / len(returns) if returns else 0
    loss_rate = 1 - win_rate

    # Expectancy (EV): % medio ganado por trade teniendo en cuenta probabilidades
    expectancy = round((win_rate * avg_win) + (loss_rate * avg_loss), 4)

    # R:R medio (reward/risk)
    avg_rr = round(abs(avg_win / avg_loss), 4) if avg_loss != 0 else 0.0

    # Maxima racha de perdidas consecutivas
    max_consec = 0
    current    = 0
    for r in returns:
        if r <= 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    # Equity smoothness: R² entre equity real y linea recta ideal
    # 1.0 = curva perfectamente suave y creciente, 0.0 = caos total
    smoothness = 0.0
    try:
        eq_values = equity_curve.values.astype(float)
        x         = np.arange(len(eq_values))
        x_mean    = x.mean()
        y_mean    = eq_values.mean()
        ss_tot    = ((eq_values - y_mean) ** 2).sum()
        ss_res    = ((eq_values - (y_mean + (np.polyfit(x, eq_values, 1)[0] * (x - x_mean)))) ** 2).sum()
        r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        smoothness = round(max(0.0, min(1.0, float(r2))), 4)
    except Exception:
        smoothness = 0.0

    return {
        "expectancy":        expectancy,
        "avg_win_pct":       avg_win,
        "avg_loss_pct":      avg_loss,
        "avg_rr":            avg_rr,
        "max_consec_losses": max_consec,
        "equity_smoothness": smoothness,
    }


# ─────────────────────────────────────────────────────────────
# 5.  MOTOR DE BACKTEST  →  devuelve dict JSON-serializable
# ─────────────────────────────────────────────────────────────
def run_backtest(params: dict) -> dict:
    """
    Recibe params ya validados.
    Devuelve siempre un dict con estructura OUTPUT SCHEMA.
    Nunca lanza excepciones — los errores van en output["error"].
    """
    output = {
        "status":  "ok",
        "input":   {k: v for k, v in params.items() if not k.startswith("_")},
        "summary": {},
        "trades":  [],
        "error":   None,
    }

    try:
        freq = TIMEFRAME_FREQ.get(params["timeframe"], params["timeframe"])

        # Descarga — headers para evitar bloqueo de yfinance en servidores cloud
        import yfinance as yf
        yf.set_tz_cache_location("/tmp")
        session = None
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            })
        except Exception:
            pass

        raw = yf.download(
            params["ticker"],
            start=params["start"],
            end=params["end"],
            interval=params["timeframe"],
            auto_adjust=True,
            progress=False,
            session=session,
        )
        if raw.empty:
            raise ValueError(
                f"yfinance no devolvio datos para '{params['ticker']}' en {params['timeframe']}. "
                "Verifica el ticker y el rango de fechas."
            )
        # Compatibilidad con MultiIndex de yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        data_close = raw["Close"].dropna()
        close = data_close
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        if len(close) < params["rsi_period"] + 10:
            raise ValueError(
                f"Muy pocas barras ({len(close)}) para RSI({params['rsi_period']}). "
                "Amplia el rango de fechas."
            )

        # Senales
        entries, exits = generate_signals(
            close,
            params["rsi_period"],
            params["ema_period"],
            params["rsi_entry"],
            params["rsi_exit"],
        )

        entry_signals = int(entries.sum())
        exit_signals  = int(exits.sum())

        if entry_signals == 0:
            raise ValueError(
                "No se generaron senales de entrada. "
                "Prueba rsi_entry mas alto (ej. 40) o rsi_exit mas bajo (ej. 60)."
            )

        # Portfolio
        pf = vbt.Portfolio.from_signals(
            close, entries, exits,
            init_cash  = params["init_cash"],
            size       = params["size"],
            size_type  = "percent",
            fees       = params["fees"],
            slippage   = params["slippage"],
            freq       = freq,
        )
        bh = vbt.Portfolio.from_holding(
            close,
            init_cash = params["init_cash"],
            fees      = params["fees"],
            freq      = freq,
        )

        stats  = pf.stats()
        trades = pf.trades.records_readable

        def safe(key, default=0.0):
            v = stats.get(key, default)
            return round(float(v), 4) if pd.notna(v) else default

        output["summary"] = {
            "strategy_return_pct": safe("Total Return [%]"),
            "buyhold_return_pct":  round(float(bh.stats().get("Total Return [%]", 0)), 4),
            "final_value":         round(float(pf.final_value()), 2),
            "sharpe_ratio":        safe("Sharpe Ratio"),
            "max_drawdown_pct":    safe("Max Drawdown [%]"),
            "win_rate_pct":        safe("Win Rate [%]"),
            "profit_factor":       safe("Profit Factor"),
            "total_trades":        int(safe("Total Trades")),
            "bars_loaded":         len(close),
            "entry_signals":       entry_signals,
            "exit_signals":        exit_signals,
        }

        trades_list = []
        if not trades.empty:
            for i, row in trades.iterrows():
                ret = float(row.get("Return", 0)) * 100
                pnl = float(row.get("PnL", 0))
                trades_list.append({
                    "trade_id":   int(i) + 1,
                    "entry_date": str(row.get("Entry Timestamp", ""))[:10],
                    "exit_date":  str(row.get("Exit Timestamp",  ""))[:10],
                    "return_pct": round(ret, 4),
                    "pnl":        round(pnl, 2),
                    "status":     "win" if ret > 0 else "loss",
                })
        output["trades"]   = trades_list
        output["analysis"]    = compute_analysis(trades_list, pf.value())

        # Equity curve como lista de puntos {date, value} para la web
        eq = pf.value()
        output["equity_curve"] = [
            {"date": str(d)[:10], "value": round(float(v), 2)}
            for d, v in zip(eq.index, eq.values)
        ]

        # Objetos internos para grafica (no van al JSON)
        output["_close"] = close
        output["_pf"]    = pf
        output["_bh"]    = bh

    except Exception as e:
        output["status"] = "error"
        output["error"]  = str(e)

    return output


# ─────────────────────────────────────────────────────────────
# 6.  PRINT FORMATEADO (terminal humano)
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
    print(f"  RSI        : period={params.get('rsi_period')}  "
          f"entry<{params.get('rsi_entry')}  exit>{params.get('rsi_exit')}")
    print(f"  EMA        : period={params.get('ema_period')}")
    print(f"  Capital    : ${params.get('init_cash', 0):,.0f}")
    print(f"  Fees       : {params.get('fees', 0)*100:.2f}%   "
          f"Slippage: {params.get('slippage', 0)*100:.3f}%")
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
    print(f"  {'Strategy Return':<28} {color}{BOLD}{ret:.2f}%{RESET}")
    print(f"  {'Buy-and-Hold Return':<28} {summary.get('buyhold_return_pct', 0):.2f}%")
    print(f"  {'Final Portfolio Value':<28} ${summary.get('final_value', 0):,.2f}")
    print(f"  {'Sharpe Ratio':<28} {summary.get('sharpe_ratio', 0):.2f}")
    print(f"  {'Max Drawdown':<28} {RED}{summary.get('max_drawdown_pct', 0):.2f}%{RESET}")
    print(f"  {'Win Rate':<28} {summary.get('win_rate_pct', 0):.2f}%")
    print(f"  {'Profit Factor':<28} {summary.get('profit_factor', 0):.2f}")
    print(f"  {'Total Trades':<28} {summary.get('total_trades', 0)}")
    print(f"{BOLD}{'═'*55}{RESET}\n")

    analysis = output.get("analysis", {})
    if analysis:
        ev    = analysis.get("expectancy", 0)
        ev_c  = GREEN if ev > 0 else RED
        sm    = analysis.get("equity_smoothness", 0)
        sm_c  = GREEN if sm > 0.7 else (YELLOW if sm > 0.4 else RED)
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
# 7.  GRAFICA
# ─────────────────────────────────────────────────────────────
def plot_results(output: dict):
    if output.get("status") == "error":
        return

    close  = output["_close"]
    pf     = output["_pf"]
    bh     = output["_bh"]
    params = output["input"]

    rsi = compute_rsi(close, params["rsi_period"])
    ema = compute_ema(close, params["ema_period"])

    fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
    gs  = plt.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1.4], hspace=0.08)
    ax_price  = fig.add_subplot(gs[0])
    ax_rsi    = fig.add_subplot(gs[1], sharex=ax_price)
    ax_equity = fig.add_subplot(gs[2], sharex=ax_price)

    for ax in (ax_price, ax_rsi, ax_equity):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.spines[:].set_color("#30363d")
        ax.yaxis.label.set_color("#8b949e")

    ax_price.plot(close.index, close.values, color="#58a6ff", lw=1.2, label="Price")
    ax_price.plot(ema.index,   ema.values,   color="#f0883e", lw=1.2,
                  linestyle="--", label=f"EMA({params['ema_period']})")

    try:
        td = pf.trades.records_readable
        e_idx = pd.to_datetime(td["Entry Timestamp"])
        x_idx = pd.to_datetime(td["Exit Timestamp"])
        ax_price.scatter(e_idx, close.reindex(e_idx),
                         marker="^", color="#3fb950", s=90, zorder=5, label="Entry")
        ax_price.scatter(x_idx, close.reindex(x_idx),
                         marker="v", color="#f85149", s=90, zorder=5, label="Exit")
    except Exception:
        pass

    ax_price.set_title(
        f"{params['ticker']}  ·  RSI({params['rsi_period']}) + EMA({params['ema_period']})"
        f"  ·  {params['timeframe']}  ·  {params['start']} → {params['end']}",
        color="#e6edf3", fontsize=12, pad=12, fontweight="bold")
    ax_price.legend(framealpha=0.15, labelcolor="#c9d1d9", fontsize=9)
    ax_price.set_ylabel("Price")

    ax_rsi.plot(rsi.index, rsi.values, color="#bc8cff", lw=1.1)
    ax_rsi.axhline(params["rsi_entry"], color="#3fb950", linestyle="--", lw=0.8, alpha=0.7)
    ax_rsi.axhline(params["rsi_exit"],  color="#f85149", linestyle="--", lw=0.8, alpha=0.7)
    ax_rsi.fill_between(rsi.index, rsi, params["rsi_entry"],
                        where=(rsi < params["rsi_entry"]), color="#3fb950", alpha=0.12)
    ax_rsi.fill_between(rsi.index, rsi, params["rsi_exit"],
                        where=(rsi > params["rsi_exit"]),  color="#f85149", alpha=0.12)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel(f"RSI({params['rsi_period']})")

    eq_s = pf.value()
    eq_b = bh.value()
    ax_equity.plot(eq_s.index, eq_s.values, color="#58a6ff", lw=1.2, label="Strategy")
    ax_equity.plot(eq_b.index, eq_b.values, color="#8b949e", lw=1,
                   linestyle=":", label="Buy & Hold")
    ax_equity.fill_between(eq_s.index, eq_s.values, params["init_cash"],
                           where=(eq_s >= params["init_cash"]), color="#3fb950", alpha=0.12)
    ax_equity.fill_between(eq_s.index, eq_s.values, params["init_cash"],
                           where=(eq_s <  params["init_cash"]), color="#f85149", alpha=0.12)
    ax_equity.set_ylabel("Portfolio Value ($)")
    ax_equity.legend(framealpha=0.15, labelcolor="#c9d1d9", fontsize=9)

    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_rsi.get_xticklabels(),   visible=False)
    plt.tight_layout()

    ticker_safe = params["ticker"].replace("-", "_").replace("=", "")
    fname = f"backtest_{ticker_safe}_{params['timeframe']}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Grafica guardada → {fname}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 8.  OPTIMIZADOR
# ─────────────────────────────────────────────────────────────
def optimise(close: pd.Series, params: dict):
    print(f"\n{YELLOW}{BOLD}  Optimizando parametros RSI...{RESET}")
    results = []
    freq = TIMEFRAME_FREQ.get(params["timeframe"], params["timeframe"])

    for rsi_e in range(20, 50, 5):
        for rsi_x in range(55, 85, 5):
            if rsi_e >= rsi_x:
                continue
            entries, exits = generate_signals(
                close, params["rsi_period"], params["ema_period"], rsi_e, rsi_x)
            if entries.sum() < 3:
                continue
            pf = vbt.Portfolio.from_signals(
                close, entries, exits,
                init_cash=params["init_cash"], size=params["size"],
                size_type="percent", fees=params["fees"],
                slippage=params["slippage"], freq=freq,
            )
            s = pf.stats()
            results.append({
                "rsi_entry":    rsi_e,
                "rsi_exit":     rsi_x,
                "total_return": round(float(s.get("Total Return [%]", 0)), 2),
                "sharpe":       round(float(s.get("Sharpe Ratio", 0)), 2),
                "max_dd":       round(float(s.get("Max Drawdown [%]", 0)), 2),
                "num_trades":   int(s.get("Total Trades", 0)),
            })

    if not results:
        print(f"{RED}  Sin combinaciones validas.{RESET}")
        return

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(f"\n{BOLD}  Top 5 combinaciones (por Sharpe):{RESET}")
    print(f"  {'Entry':>7} {'Exit':>6} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7}")
    print(f"  {'─'*50}")
    for _, row in df.head(5).iterrows():
        print(f"  {int(row['rsi_entry']):>7}  {int(row['rsi_exit']):>6}  "
              f"{row['total_return']:>8.2f}  {row['sharpe']:>7.2f}  "
              f"{row['max_dd']:>7.2f}  {int(row['num_trades']):>6}")
    print()


# ─────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="RSI + EMA backtest — input/output JSON estandarizado",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--json",      type=str, metavar="JSON_STRING",
                   help="Input como string JSON")
    g.add_argument("--json-file", type=str, metavar="FILE",
                   help="Input desde archivo JSON")

    # CLI clasico
    p.add_argument("--ticker",     type=str,   default=None)
    p.add_argument("--timeframe",  type=str,   default=None,
                   choices=list(TIMEFRAME_FREQ.keys()))
    p.add_argument("--start",      type=str,   default=None)
    p.add_argument("--end",        type=str,   default=None)
    p.add_argument("--rsi-period", type=int,   default=None)
    p.add_argument("--ema-period", type=int,   default=None)
    p.add_argument("--rsi-entry",  type=float, default=None)
    p.add_argument("--rsi-exit",   type=float, default=None)
    p.add_argument("--cash",       type=float, default=None)
    p.add_argument("--fees",       type=float, default=None)

    # Output
    p.add_argument("--output",    type=str,  default=None,
                   help="Guardar resultado JSON en archivo")
    p.add_argument("--json-only", action="store_true",
                   help="Imprimir solo JSON puro (para API/web)")
    p.add_argument("--no-plot",   action="store_true")
    p.add_argument("--optimise",  action="store_true")
    return p.parse_args()


def build_raw_input(args) -> dict:
    if args.json:
        return json.loads(args.json)
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # CLI clasico
    cli_map = {
        "ticker":     args.ticker,
        "timeframe":  args.timeframe,
        "start":      args.start,
        "end":        args.end,
        "rsi_period": args.rsi_period,
        "ema_period": args.ema_period,
        "rsi_entry":  args.rsi_entry,
        "rsi_exit":   args.rsi_exit,
        "init_cash":  args.cash,
        "fees":       args.fees,
    }
    return {k: v for k, v in cli_map.items() if v is not None}


def clean_output(output: dict) -> dict:
    """Elimina claves internas (_close, _pf, _bh) para serializar a JSON."""
    return {k: v for k, v in output.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Construir raw input
    raw = build_raw_input(args)

    # 2. Validar y rellenar con defaults
    params, errors = validate_and_fill(raw)

    if errors:
        error_output = {
            "status": "error",
            "input":  raw,
            "summary": {},
            "trades": [],
            "error":  " | ".join(errors),
        }
        if args.json_only:
            print(json.dumps(error_output, indent=2, ensure_ascii=False))
        else:
            print(f"\n{RED}{BOLD}Errores de validacion:{RESET}")
            for e in errors:
                print(f"  • {e}")
        sys.exit(1)

    # 3. Ejecutar backtest
    output = run_backtest(params)

    # 4. Mostrar resultado
    if args.json_only:
        print(json.dumps(clean_output(output), indent=2, ensure_ascii=False))
    else:
        print_output(output)

    # 5. Guardar JSON si se pidio
    if args.output and output["status"] == "ok":
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(clean_output(output), f, indent=2, ensure_ascii=False)
        print(f"  Resultado guardado → {args.output}")

    # 6. Optimizar
    if args.optimise and output["status"] == "ok":
        optimise(output["_close"], params)

    # 7. Grafica
    if not args.no_plot and output["status"] == "ok":
        plot_results(output)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────
# 11.  FUNCION PUBLICA — importable desde la API / web
# ─────────────────────────────────────────────────────────────
def backtest(config: dict) -> dict:
    """
    Motor reutilizable. Llamar directamente desde FastAPI, Flask, o cualquier script.

    Uso:
        from backtest_rsi_ema import backtest

        resultado = backtest({
            "ticker":     "EURUSD=X",
            "timeframe":  "1h",
            "start":      "2025-01-01",
            "end":        "2025-03-01",
            "rsi_period": 14,
            "ema_period": 50,
            "rsi_entry":  40,
            "rsi_exit":   60,
            "init_cash":  10000,
            "fees":       0.001,
        })

        print(resultado["status"])           # "ok" o "error"
        print(resultado["summary"])          # metricas
        print(resultado["trades"])           # lista de trades
        print(resultado["error"])            # None o mensaje de error

    Returns:
        dict con keys: status, input, summary, trades, error
        (nunca lanza excepciones — los errores van dentro del dict)
    """
    params, errors = validate_and_fill(config)

    if errors:
        return {
            "status":  "error",
            "input":   config,
            "summary": {},
            "trades":  [],
            "error":   " | ".join(errors),
        }

    output = run_backtest(params)
    return clean_output(output)
