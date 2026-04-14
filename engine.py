"""
=============================================================
  TradingEdgeLabs — Motor Universal de Estrategias
=============================================================
Acepta estrategias con N confirmaciones en orden secuencial.
Cada confirmación es un indicador + condición evaluada AND.

INPUT JSON:
{
  "market": {
    "ticker":    "AAPL",
    "timeframe": "1d",
    "start":     "2024-01-01",
    "end":       "2024-06-01"
  },
  "risk": {
    "capital":  10000,
    "fees":     0.001,
    "slippage": 0.0005,
    "size":     0.99
  },
  "strategy": {
    "entry_confirmations": [
      {"indicator": "rsi",  "params": {"period": 14}, "condition": "crosses_above", "value": 30},
      {"indicator": "ema",  "params": {"period": 50}, "condition": "price_above"},
      {"indicator": "sma",  "params": {"period": 200},"condition": "price_above"}
    ],
    "exit_confirmations": [
      {"indicator": "rsi",  "params": {"period": 14}, "condition": "crosses_below", "value": 70}
    ]
  }
}

INDICADORES DISPONIBLES:
  rsi    — RSI de Wilder
  ema    — Media móvil exponencial
  sma    — Media móvil simple
  macd   — MACD (line, signal, histogram)
  bb     — Bandas de Bollinger
  atr    — Average True Range
  stoch  — Estocástico

CONDICIONES DISPONIBLES:
  crosses_above  — el indicador cruza hacia arriba un valor
  crosses_below  — el indicador cruza hacia abajo un valor
  above          — el indicador está por encima de un valor
  below          — el indicador está por debajo de un valor
  price_above    — el precio está por encima del indicador
  price_below    — el precio está por debajo del indicador
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1.  INDICADORES
# ─────────────────────────────────────────────────────────────

def ind_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ind_ema(close: pd.Series, period: int = 20) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def ind_sma(close: pd.Series, period: int = 20) -> pd.Series:
    return close.rolling(window=period).mean()


def ind_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Devuelve dict con line, signal_line, histogram."""
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return {"line": macd_line, "signal": signal_line, "histogram": histogram}


def ind_bb(close: pd.Series, period: int = 20, std: float = 2.0) -> dict:
    """Bandas de Bollinger. Devuelve upper, middle, lower."""
    middle = close.rolling(window=period).mean()
    sigma  = close.rolling(window=period).std()
    return {"upper": middle + std * sigma, "middle": middle, "lower": middle - std * sigma}


def ind_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def ind_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
              k_period: int = 14, d_period: int = 3) -> dict:
    """Estocástico. Devuelve k y d."""
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return {"k": k, "d": d}


# ─────────────────────────────────────────────────────────────
# 2.  FUNCIÓN UNIVERSAL DE INDICADORES
# ─────────────────────────────────────────────────────────────

def evaluate_indicator(
    data:           pd.DataFrame,
    indicator_name: str,
    params:         dict,
) -> pd.Series:
    """
    Calcula el indicador y devuelve la serie principal.
    data debe tener columnas: Close, High, Low (si aplica).
    """
    close = data["Close"]
    name  = indicator_name.lower()

    if name == "rsi":
        return ind_rsi(close, period=params.get("period", 14))

    elif name == "ema":
        return ind_ema(close, period=params.get("period", 20))

    elif name == "sma":
        return ind_sma(close, period=params.get("period", 20))

    elif name == "macd":
        result = ind_macd(
            close,
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            signal=params.get("signal", 9),
        )
        # Por defecto devuelve la línea MACD; se puede especificar "output"
        output = params.get("output", "line")
        return result[output]

    elif name == "bb":
        result = ind_bb(close, period=params.get("period", 20), std=params.get("std", 2.0))
        output = params.get("output", "middle")
        return result[output]

    elif name == "atr":
        if "High" in data.columns and "Low" in data.columns:
            return ind_atr(data["High"], data["Low"], close, period=params.get("period", 14))
        return ind_atr(close, close, close, period=params.get("period", 14))

    elif name == "stoch":
        if "High" in data.columns and "Low" in data.columns:
            result = ind_stoch(data["High"], data["Low"], close,
                               k_period=params.get("k_period", 14),
                               d_period=params.get("d_period", 3))
        else:
            result = ind_stoch(close, close, close,
                               k_period=params.get("k_period", 14),
                               d_period=params.get("d_period", 3))
        output = params.get("output", "k")
        return result[output]

    else:
        raise ValueError(f"Indicador '{indicator_name}' no reconocido. "
                         f"Disponibles: rsi, ema, sma, macd, bb, atr, stoch")


# ─────────────────────────────────────────────────────────────
# 3.  EVALUADOR DE CONDICIONES
# ─────────────────────────────────────────────────────────────

def check_condition(
    ind_values: pd.Series,
    condition:  str,
    value:      Optional[float],
    close:      pd.Series,
    i:          int,
) -> bool:
    """
    Evalúa una condición en la barra i.
    Retorna True si la condición se cumple.
    """
    if i < 1:
        return False

    iv      = ind_values.iloc
    cv      = close.iloc
    cond    = condition.lower()

    curr_ind  = float(iv[i])   if not np.isnan(iv[i])   else None
    prev_ind  = float(iv[i-1]) if not np.isnan(iv[i-1]) else None
    curr_price = float(cv[i])

    if curr_ind is None or prev_ind is None:
        return False

    if cond == "crosses_above":
        # indicador cruza hacia arriba el valor
        return prev_ind < value and curr_ind >= value

    elif cond == "crosses_below":
        # indicador cruza hacia abajo el valor
        return prev_ind > value and curr_ind <= value

    elif cond == "above":
        # indicador está por encima del valor
        return curr_ind > value

    elif cond == "below":
        # indicador está por debajo del valor
        return curr_ind < value

    elif cond == "price_above":
        # precio está por encima del indicador
        return curr_price > curr_ind

    elif cond == "price_below":
        # precio está por debajo del indicador
        return curr_price < curr_ind

    elif cond == "macd_crosses_above_signal":
        # MACD cruza su signal hacia arriba
        return prev_ind < 0 and curr_ind >= 0

    elif cond == "macd_crosses_below_signal":
        return prev_ind > 0 and curr_ind <= 0

    else:
        raise ValueError(f"Condición '{condition}' no reconocida. "
                         f"Disponibles: crosses_above, crosses_below, above, below, "
                         f"price_above, price_below")


# ─────────────────────────────────────────────────────────────
# 4.  PRECOMPUTAR INDICADORES
# ─────────────────────────────────────────────────────────────

def precompute_indicators(data: pd.DataFrame, confirmations: list) -> list:
    """
    Pre-calcula todos los indicadores de una lista de confirmaciones.
    Devuelve lista de (ind_series, condition, value) lista para evaluar.
    """
    computed = []
    for conf in confirmations:
        ind_name  = conf["indicator"]
        ind_params = conf.get("params", {})
        condition = conf["condition"]
        value     = conf.get("value", None)

        ind_series = evaluate_indicator(data, ind_name, ind_params)
        computed.append({
            "name":      ind_name,
            "series":    ind_series,
            "condition": condition,
            "value":     value,
        })
    return computed


# ─────────────────────────────────────────────────────────────
# 5.  SIMULACIÓN
# ─────────────────────────────────────────────────────────────

def simulate(data: pd.DataFrame, risk: dict, entry_confs: list, exit_confs: list) -> dict:
    """
    Simulación barra a barra con N confirmaciones de entrada y salida.
    Todas las confirmaciones se evalúan en AND.
    """
    close    = data["Close"]
    fees     = risk.get("fees",     0.001)
    slippage = risk.get("slippage", 0.0005)
    size     = risk.get("size",     0.99)
    cash     = risk.get("capital",  10000)

    prices = close.values
    dates  = close.index

    # Precomputar indicadores
    entry_computed = precompute_indicators(data, entry_confs)
    exit_computed  = precompute_indicators(data, exit_confs)

    in_trade      = False
    entry_price   = 0.0
    entry_date    = None
    shares        = 0.0
    trades        = []
    equity_curve  = []
    entry_signals = 0
    exit_signals  = 0

    for i in range(1, len(prices)):
        price = float(prices[i])
        date  = str(dates[i])[:10]

        # Evaluar TODAS las confirmaciones de entrada (AND)
        if not in_trade:
            entry_signal = all(
                check_condition(c["series"], c["condition"], c["value"], close, i)
                for c in entry_computed
            )
            if entry_signal:
                entry_signals += 1
                exec_price  = price * (1 + slippage)
                shares      = (cash * size) / exec_price
                cost        = shares * exec_price * (1 + fees)
                cash       -= cost
                entry_price = exec_price
                entry_date  = date
                in_trade    = True

        # Evaluar TODAS las confirmaciones de salida (AND)
        elif in_trade:
            exit_signal = all(
                check_condition(c["series"], c["condition"], c["value"], close, i)
                for c in exit_computed
            )
            if exit_signal:
                exit_signals += 1
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
                in_trade = False
                shares   = 0.0

        equity = cash + shares * price if in_trade else cash
        equity_curve.append({"date": date, "value": round(equity, 2)})

    # Cerrar trade abierto al final
    if in_trade:
        price      = float(prices[-1])
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
        if equity_curve:
            equity_curve[-1]["value"] = round(cash, 2)

    return {
        "trades":        trades,
        "equity_curve":  equity_curve,
        "final_value":   round(cash, 2),
        "entry_signals": entry_signals,
        "exit_signals":  exit_signals,
    }


# ─────────────────────────────────────────────────────────────
# 6.  MÉTRICAS (igual que antes)
# ─────────────────────────────────────────────────────────────

def compute_metrics(sim: dict, risk: dict, close: pd.Series) -> tuple:
    trades      = sim["trades"]
    eq_curve    = sim["equity_curve"]
    init_cash   = risk.get("capital", 10000)
    final_value = sim["final_value"]

    eq_values = np.array([p["value"] for p in eq_curve], dtype=float)
    returns   = [t["return_pct"] for t in trades]
    wins      = [r for r in returns if r > 0]
    losses    = [r for r in returns if r <= 0]
    n         = len(trades)

    strategy_return = round((final_value - init_cash) / init_cash * 100, 4)
    bh_return       = round((float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) * 100, 4)

    sum_wins   = sum(wins)         if wins   else 0.0
    sum_losses = abs(sum(losses))  if losses else 0.0
    profit_factor = round(sum_wins / sum_losses, 4) if sum_losses > 0 else 0.0

    win_rate  = round(len(wins) / n * 100, 4) if n > 0 else 0.0
    avg_win   = round(float(np.mean(wins)),   4) if wins   else 0.0
    avg_loss  = round(float(np.mean(losses)), 4) if losses else 0.0
    avg_rr    = round(abs(avg_win / avg_loss), 4) if avg_loss != 0 else 0.0
    wr_dec    = win_rate / 100
    expectancy             = round((wr_dec * avg_win) + ((1 - wr_dec) * avg_loss), 4)
    expectancy_normalized  = round(expectancy / abs(avg_loss), 4) if avg_loss != 0 else 0.0

    # Max Drawdown
    peak = init_cash
    max_dd_pct = max_dd_abs = 0.0
    for v in eq_values:
        if v > peak: peak = v
        dd_pct = (peak - v) / peak * 100
        dd_abs = peak - v
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_abs = dd_abs
    max_dd_pct = round(max_dd_pct, 4)
    max_dd_abs = round(max_dd_abs, 2)

    # Sharpe
    sharpe = 0.0
    if len(eq_values) > 1:
        ret_arr = np.diff(eq_values) / eq_values[:-1]
        sharpe  = round(float(np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(252)), 4) if np.std(ret_arr) > 0 else 0.0

    # Ulcer Index
    ulcer_index = 0.0
    if len(eq_values) > 1:
        peak_arr    = np.maximum.accumulate(eq_values)
        dd_arr      = (peak_arr - eq_values) / peak_arr * 100
        ulcer_index = round(float(np.sqrt(np.mean(dd_arr ** 2))), 4)

    recovery_factor   = round(abs(strategy_return) / max_dd_pct, 4) if max_dd_pct > 0 else 0.0
    breakeven_winrate = round(1 / (1 + avg_rr) * 100, 4) if avg_rr > 0 else 50.0
    winrate_edge      = round(win_rate - breakeven_winrate, 4)

    # Trade durations
    durations = []
    for t in trades:
        try:
            d1 = datetime.strptime(t["entry_date"], "%Y-%m-%d")
            d2 = datetime.strptime(t["exit_date"],  "%Y-%m-%d")
            durations.append((d2 - d1).days)
        except Exception:
            pass
    avg_trade_duration = round(float(np.mean(durations)), 2) if durations else 0.0
    max_trade_duration = int(max(durations)) if durations else 0
    min_trade_duration = int(min(durations)) if durations else 0

    # Max consecutive losses
    max_consec = current = 0
    for r in returns:
        if r <= 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    # Distribution / Skewness / Kurtosis
    ret_arr_trades = np.array(returns, dtype=float)
    pct_big_wins   = 0.0
    if n > 0 and avg_win > 0:
        pct_big_wins = round(len([r for r in wins if r > avg_win * 2]) / n * 100, 4)

    skewness = kurtosis = 0.0
    if n >= 4:
        try:
            mean_r = float(np.mean(ret_arr_trades))
            std_r  = float(np.std(ret_arr_trades))
            if std_r > 0:
                skewness = round(float(np.mean(((ret_arr_trades - mean_r) / std_r) ** 3)), 4)
                kurtosis = round(float(np.mean(((ret_arr_trades - mean_r) / std_r) ** 4)) - 3, 4)
        except Exception:
            pass

    # Equity slope / R²
    eq_slope = eq_r_squared = 0.0
    if len(eq_values) > 2:
        try:
            x      = np.arange(len(eq_values), dtype=float)
            coeffs = np.polyfit(x, eq_values, 1)
            y_hat  = np.polyval(coeffs, x)
            ss_res = np.sum((eq_values - y_hat) ** 2)
            ss_tot = np.sum((eq_values - eq_values.mean()) ** 2)
            eq_slope     = round(float(coeffs[0]), 6)
            eq_r_squared = round(max(0.0, min(1.0, 1 - ss_res / ss_tot)), 4) if ss_tot > 0 else 0.0
        except Exception:
            pass

    # Consistency score
    consistency_score = 0.0
    if eq_curve:
        try:
            df_eq = pd.DataFrame(eq_curve)
            df_eq["date"]  = pd.to_datetime(df_eq["date"])
            df_eq["month"] = df_eq["date"].dt.to_period("M")
            monthly    = df_eq.groupby("month")["value"].last()
            monthly_ret = monthly.pct_change().dropna()
            if len(monthly_ret) > 0:
                consistency_score = round(float((monthly_ret > 0).mean() * 100), 4)
        except Exception:
            pass

    # Edge Score
    edge_score = 0.0
    edge_label = "SIN EDGE"
    try:
        pf_norm   = min(profit_factor / 3.0, 1.0)
        ev_norm   = min(max(expectancy_normalized, 0) / 2.0, 1.0)
        cons_norm = consistency_score / 100.0
        dd_pen    = max(max_dd_pct, 1.0)
        raw_score = (pf_norm * ev_norm * cons_norm * 100) / (dd_pen / 10)
        edge_score = round(min(max(raw_score, 0.0), 100.0), 2)
        edge_label = (
            "EDGE FUERTE"   if edge_score >= 75 else
            "EDGE MODERADO" if edge_score >= 50 else
            "EDGE DEBIL"    if edge_score >= 25 else
            "SIN EDGE"
        )
    except Exception:
        pass

    summary = {
        "strategy_return_pct": strategy_return,
        "buyhold_return_pct":  bh_return,
        "final_value":         final_value,
        "sharpe_ratio":        sharpe,
        "max_drawdown_pct":    max_dd_pct,
        "max_drawdown_abs":    max_dd_abs,
        "win_rate_pct":        win_rate,
        "profit_factor":       profit_factor,
        "total_trades":        n,
        "bars_loaded":         len(close),
        "entry_signals":       sim["entry_signals"],
        "exit_signals":        sim["exit_signals"],
    }

    analysis = {
        "profit_factor":           profit_factor,
        "expectancy_pct":          expectancy,
        "expectancy_normalized":   expectancy_normalized,
        "max_drawdown_pct":        max_dd_pct,
        "max_drawdown_abs":        max_dd_abs,
        "sharpe_ratio":            sharpe,
        "ulcer_index":             ulcer_index,
        "recovery_factor":         recovery_factor,
        "winrate_edge":            winrate_edge,
        "breakeven_winrate":       breakeven_winrate,
        "avg_trade_duration_days": avg_trade_duration,
        "max_trade_duration_days": max_trade_duration,
        "min_trade_duration_days": min_trade_duration,
        "pct_big_wins":            pct_big_wins,
        "skewness":                skewness,
        "kurtosis":                kurtosis,
        "equity_slope":            eq_slope,
        "equity_r_squared":        eq_r_squared,
        "consistency_score":       consistency_score,
        "avg_win_pct":             avg_win,
        "avg_loss_pct":            avg_loss,
        "avg_rr":                  avg_rr,
        "max_consec_losses":       max_consec,
        "edge_score":              edge_score,
        "edge_label":              edge_label,
    }

    return summary, analysis


# ─────────────────────────────────────────────────────────────
# 7.  DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def download_data(market: dict) -> pd.DataFrame:
    import yfinance as yf
    ticker    = market["ticker"]
    timeframe = market.get("timeframe", "1d")
    start     = market["start"]
    end       = market["end"]

    raw = yf.download(ticker, start=start, end=end, interval=timeframe,
                      auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            f"yfinance no devolvio datos para '{ticker}' en {timeframe}. "
            "Verifica el ticker y el rango de fechas."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return raw.dropna()


# ─────────────────────────────────────────────────────────────
# 8.  FUNCIÓN PRINCIPAL — importable desde API
# ─────────────────────────────────────────────────────────────

def run_strategy(config: dict) -> dict:
    """
    Punto de entrada principal. Acepta el config JSON completo.
    Siempre devuelve un dict — nunca lanza excepciones.

    config = {
        "market":   { ticker, timeframe, start, end },
        "risk":     { capital, fees, slippage, size },
        "strategy": {
            "entry_confirmations": [...],
            "exit_confirmations":  [...]
        }
    }
    """
    output = {
        "status":       "ok",
        "input":        config,
        "summary":      {},
        "trades":       [],
        "equity_curve": [],
        "analysis":     {},
        "error":        None,
    }

    try:
        market   = config.get("market",   {})
        risk     = config.get("risk",     {})
        strategy = config.get("strategy", {})

        entry_confs = strategy.get("entry_confirmations", [])
        exit_confs  = strategy.get("exit_confirmations",  [])

        if not entry_confs:
            raise ValueError("Debes añadir al menos una confirmación de entrada.")
        if not exit_confs:
            raise ValueError("Debes añadir al menos una confirmación de salida.")

        # Descarga
        data  = download_data(market)
        close = data["Close"]

        if len(close) < 30:
            raise ValueError(f"Muy pocas barras ({len(close)}). Amplía el rango de fechas.")

        # Simulación
        sim = simulate(data, risk, entry_confs, exit_confs)

        if sim["entry_signals"] == 0:
            raise ValueError(
                "No se generaron señales de entrada con estas confirmaciones. "
                "Prueba a ajustar los parámetros o ampliar el rango de fechas."
            )

        # Métricas
        summary, analysis = compute_metrics(sim, risk, close)

        output["summary"]      = summary
        output["trades"]       = sim["trades"]
        output["equity_curve"] = sim["equity_curve"]
        output["analysis"]     = analysis

    except Exception as e:
        output["status"] = "error"
        output["error"]  = str(e)

    return output


# ─────────────────────────────────────────────────────────────
# 9.  CATÁLOGO DE INDICADORES (para el frontend)
# ─────────────────────────────────────────────────────────────

INDICATOR_CATALOG = {
    "rsi": {
        "name":        "RSI — Relative Strength Index",
        "params":      [{"key": "period", "label": "Periodo", "default": 14, "min": 2, "max": 100}],
        "conditions":  ["crosses_above", "crosses_below", "above", "below"],
        "needs_value": True,
        "value_label": "Nivel RSI (0-100)",
        "value_range": [0, 100],
    },
    "ema": {
        "name":        "EMA — Media Móvil Exponencial",
        "params":      [{"key": "period", "label": "Periodo", "default": 20, "min": 2, "max": 500}],
        "conditions":  ["price_above", "price_below"],
        "needs_value": False,
    },
    "sma": {
        "name":        "SMA — Media Móvil Simple",
        "params":      [{"key": "period", "label": "Periodo", "default": 50, "min": 2, "max": 500}],
        "conditions":  ["price_above", "price_below"],
        "needs_value": False,
    },
    "macd": {
        "name":   "MACD",
        "params": [
            {"key": "fast",   "label": "Rápida",  "default": 12, "min": 2,  "max": 100},
            {"key": "slow",   "label": "Lenta",   "default": 26, "min": 2,  "max": 200},
            {"key": "signal", "label": "Signal",  "default": 9,  "min": 2,  "max": 50},
        ],
        "conditions":  ["crosses_above", "crosses_below", "above", "below"],
        "needs_value": True,
        "value_label": "Nivel (0 = cruce de señal)",
        "value_range": [-10, 10],
    },
    "stoch": {
        "name":   "Estocástico",
        "params": [
            {"key": "k_period", "label": "%K Periodo", "default": 14, "min": 2, "max": 100},
            {"key": "d_period", "label": "%D Periodo", "default": 3,  "min": 1, "max": 20},
        ],
        "conditions":  ["crosses_above", "crosses_below", "above", "below"],
        "needs_value": True,
        "value_label": "Nivel (0-100)",
        "value_range": [0, 100],
    },
    "bb": {
        "name":   "Bandas de Bollinger",
        "params": [
            {"key": "period", "label": "Periodo",     "default": 20,  "min": 2, "max": 200},
            {"key": "std",    "label": "Desviaciones", "default": 2.0, "min": 0.5, "max": 4.0},
        ],
        "conditions":  ["price_above", "price_below"],
        "needs_value": False,
        "outputs":     ["upper", "middle", "lower"],
    },
    "atr": {
        "name":        "ATR — Average True Range",
        "params":      [{"key": "period", "label": "Periodo", "default": 14, "min": 2, "max": 100}],
        "conditions":  ["above", "below"],
        "needs_value": True,
        "value_label": "Valor ATR",
    },
}

CONDITION_LABELS = {
    "crosses_above": "Cruza hacia arriba",
    "crosses_below": "Cruza hacia abajo",
    "above":         "Está por encima de",
    "below":         "Está por debajo de",
    "price_above":   "Precio sobre el indicador",
    "price_below":   "Precio bajo el indicador",
}
