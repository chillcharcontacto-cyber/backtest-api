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
from dataclasses import dataclass

# BOS module — separate chain: engine → market_structure_bos → liquidity_channels
try:
    from market_structure_bos import evaluate_bos_indicator
    _BOS_AVAILABLE = True
except ImportError:
    _BOS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# LIQUIDITY CHANNELS — v2 (updated logic with Sweep dataclass)
# ─────────────────────────────────────────────────────────────

@dataclass
class _LiqLevel:
    price: float
    bar_index: int
    taken: bool = False
    taken_at: int = -1

@dataclass
class _Sweep:
    direction:   str    # 'bullish' o 'bearish'
    ss_bar:      int
    se_bar:      int
    sweep_level: float
    ol_price:    float
    ol_bar:      int


def _pivot_high(series: pd.Series, strength: int) -> pd.Series:
    n = strength
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(n, len(values) - n):
        window = values[i - n: i + n + 1]
        if values[i] == window.max():
            result.iloc[i] = values[i]
    return result


def _pivot_low(series: pd.Series, strength: int) -> pd.Series:
    n = strength
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(n, len(values) - n):
        window = values[i - n: i + n + 1]
        if values[i] == window.min():
            result.iloc[i] = values[i]
    return result


def compute_liquidity(data: pd.DataFrame, strength: int = 25,
                      del_after: int = 288) -> dict:
    """
    Detecta niveles de liquidez activos (ph_taken / pl_taken).
    ph_taken: pivot high cruzado (high > pivot high) — señal bajista
    pl_taken: pivot low  cruzado (low  < pivot low)  — señal alcista
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]

    high = df["High"].values
    low  = df["Low"].values
    ph_v = _pivot_high(df["High"], strength).values
    pl_v = _pivot_low(df["Low"],  strength).values
    n    = len(df)

    ph_taken   = np.zeros(n, dtype=float)
    pl_taken   = np.zeros(n, dtype=float)
    ph_bar_out = np.full(n, np.nan)
    pl_bar_out = np.full(n, np.nan)
    ph_price_out = np.full(n, np.nan)
    pl_price_out = np.full(n, np.nan)

    active_ph: list = []
    active_pl: list = []

    for i in range(n):
        if not np.isnan(ph_v[i]):
            active_ph.append(_LiqLevel(price=ph_v[i], bar_index=i))
        if not np.isnan(pl_v[i]):
            active_pl.append(_LiqLevel(price=pl_v[i], bar_index=i))

        surviving_ph = []
        for lvl in active_ph:
            if not lvl.taken:
                if high[i] > lvl.price:
                    lvl.taken = True
                    lvl.taken_at = i
                    ph_taken[i] = 1.0
                    ph_bar_out[i]   = lvl.bar_index
                    ph_price_out[i] = lvl.price
                elif (i - lvl.bar_index) >= del_after:
                    continue
            surviving_ph.append(lvl)
        active_ph = surviving_ph

        surviving_pl = []
        for lvl in active_pl:
            if not lvl.taken:
                if low[i] < lvl.price:
                    lvl.taken = True
                    lvl.taken_at = i
                    pl_taken[i] = 1.0
                    pl_bar_out[i]   = lvl.bar_index
                    pl_price_out[i] = lvl.price
                elif (i - lvl.bar_index) >= del_after:
                    continue
            surviving_pl.append(lvl)
        active_pl = surviving_pl

    return {
        "ph_taken":   pd.Series(ph_taken,   index=df.index),
        "pl_taken":   pd.Series(pl_taken,   index=df.index),
        "ph_bar":     pd.Series(ph_bar_out, index=df.index),
        "pl_bar":     pd.Series(pl_bar_out, index=df.index),
        "ph_price":   pd.Series(ph_price_out, index=df.index),
        "pl_price":   pd.Series(pl_price_out, index=df.index),
    }


def detect_sweeps_internal(data: pd.DataFrame, liq: dict) -> list:
    """
    Construye lista de _Sweep a partir de los niveles de liquidez.
    SS  = barra del pivot
    SE  = barra donde el precio cruza el sweep level
    OL  = extremo del precio en rango SS→SE
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]
    high = df["High"].values
    low  = df["Low"].values

    ph_taken   = liq["ph_taken"].values
    pl_taken   = liq["pl_taken"].values
    ph_bar     = liq["ph_bar"].values
    pl_bar     = liq["pl_bar"].values
    ph_price   = liq["ph_price"].values
    pl_price   = liq["pl_price"].values

    sweeps = []
    for se in range(len(df)):
        # Bearish sweep: pivot high cruzado
        if ph_taken[se] == 1.0 and not np.isnan(ph_bar[se]):
            ss          = int(ph_bar[se])
            sweep_level = ph_price[se]
            rng_low     = low[ss: se + 1]
            ol_bar      = ss + int(np.argmin(rng_low))
            ol_price    = rng_low.min()
            sweeps.append(_Sweep('bearish', ss, se, sweep_level, ol_price, ol_bar))

        # Bullish sweep: pivot low cruzado
        if pl_taken[se] == 1.0 and not np.isnan(pl_bar[se]):
            ss          = int(pl_bar[se])
            sweep_level = pl_price[se]
            rng_high    = high[ss: se + 1]
            ol_bar      = ss + int(np.argmax(rng_high))
            ol_price    = rng_high.max()
            sweeps.append(_Sweep('bullish', ss, se, sweep_level, ol_price, ol_bar))

    return sorted(sweeps, key=lambda s: s.se_bar)


# ─────────────────────────────────────────────────────────────
# RSI DIVERGENCE DETECTOR — v2 (uses detect_sweeps_internal)
# ─────────────────────────────────────────────────────────────

def _pivot_indices(series: pd.Series, strength: int, mode: str) -> list:
    fn = _pivot_high if mode == 'high' else _pivot_low
    result = fn(series, strength)
    return [series.index.get_loc(i) for i in result.dropna().index]


def _has_div_pair(close, rsi, start_i, end_i, direction):
    for a in range(start_i, end_i):
        for b in range(a + 1, end_i + 1):
            if np.isnan(rsi[a]) or np.isnan(rsi[b]):
                continue
            if direction == 'bull' and close[b] < close[a] and rsi[b] > rsi[a]:
                return True
            if direction == 'bear' and close[b] > close[a] and rsi[b] < rsi[a]:
                return True
    return False


def compute_rsi_divergence(data: pd.DataFrame, rsi_period: int = 14,
                            pivot_strength: int = 5, liq_strength: int = 25) -> dict:
    """
    Detecta los 4 modelos de divergencia RSI usando la lógica actualizada.
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]
    n     = len(df)
    close = df['Close'].values
    rsi_s = ind_rsi(df['Close'], rsi_period)
    rsi   = rsi_s.values

    liq     = compute_liquidity(df, strength=liq_strength)
    sweeps  = detect_sweeps_internal(df, liq)

    # Contextos para modelos 2 y 4
    contexts = []
    for idx, sw in enumerate(sweeps):
        next_se = sweeps[idx + 1].se_bar if idx + 1 < len(sweeps) else n
        contexts.append({
            'direction': 'bull' if sw.direction == 'bullish' else 'bear',
            'ss_i':      sw.ss_bar,
            'se_i':      sw.se_bar,
            'ol_i':      sw.ol_bar,
            'next_se_i': next_se,
        })

    # Modelo 1
    m1_bull = np.zeros(n, dtype=bool)
    m1_bear = np.zeros(n, dtype=bool)
    pl_pos  = _pivot_indices(df['Close'], pivot_strength, 'low')
    ph_pos  = _pivot_indices(df['Close'], pivot_strength, 'high')
    for k in range(1, len(pl_pos)):
        a, b = pl_pos[k-1], pl_pos[k]
        if not (np.isnan(rsi[a]) or np.isnan(rsi[b])):
            if close[b] < close[a] and rsi[b] > rsi[a]:
                m1_bull[b] = True
    for k in range(1, len(ph_pos)):
        a, b = ph_pos[k-1], ph_pos[k]
        if not (np.isnan(rsi[a]) or np.isnan(rsi[b])):
            if close[b] > close[a] and rsi[b] < rsi[a]:
                m1_bear[b] = True

    # Modelos 2 y 4
    m2_bull = np.zeros(n, dtype=bool)
    m2_bear = np.zeros(n, dtype=bool)
    m4_bull = np.zeros(n, dtype=bool)
    m4_bear = np.zeros(n, dtype=bool)

    for ctx in contexts:
        se_i      = ctx['se_i']
        ol_i      = ctx['ol_i']
        next_se_i = ctx['next_se_i']
        direction = ctx['direction']
        if next_se_i <= se_i + 1:
            continue
        segment = close[se_i: next_se_i]

        if direction == 'bull':
            bb_i = se_i + int(np.argmin(segment))
            if bb_i != se_i and not (np.isnan(rsi[se_i]) or np.isnan(rsi[bb_i])):
                if close[bb_i] < close[se_i] and rsi[bb_i] > rsi[se_i]:
                    m2_bull[bb_i] = True
            if bb_i > ol_i and _has_div_pair(close, rsi, ol_i, bb_i, 'bull'):
                m4_bull[bb_i] = True
        else:
            aa_i = se_i + int(np.argmax(segment))
            if aa_i != se_i and not (np.isnan(rsi[se_i]) or np.isnan(rsi[aa_i])):
                if close[aa_i] > close[se_i] and rsi[aa_i] < rsi[se_i]:
                    m2_bear[aa_i] = True
            if aa_i > ol_i and _has_div_pair(close, rsi, ol_i, aa_i, 'bear'):
                m4_bear[aa_i] = True

    m3_bull = m1_bull & m4_bull
    m3_bear = m1_bear & m4_bear

    def tf(a): return pd.Series(a.astype(float), index=df.index)
    return {
        "m1_bull": tf(m1_bull), "m1_bear": tf(m1_bear),
        "m2_bull": tf(m2_bull), "m2_bear": tf(m2_bear),
        "m3_bull": tf(m3_bull), "m3_bear": tf(m3_bear),
        "m4_bull": tf(m4_bull), "m4_bear": tf(m4_bear),
    }


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

    elif name == "liquidity":
        liq = compute_liquidity(data, strength=int(params.get("strength", 25)),
                                del_after=int(params.get("del_after", 288)))
        # Activo si hay cualquier sweep (ph_taken O pl_taken)
        combined = ((liq["ph_taken"] == 1.0) | (liq["pl_taken"] == 1.0)).astype(float)
        return combined

    elif name == "rsi_divergence":
        div = compute_rsi_divergence(
            data,
            rsi_period=int(params.get("rsi_period", 14)),
            pivot_strength=int(params.get("pivot_strength", 5)),
            liq_strength=int(params.get("liq_strength", 25)),
        )
        combined = (
            (div["m1_bull"] == 1.0) | (div["m1_bear"] == 1.0) |
            (div["m2_bull"] == 1.0) | (div["m2_bear"] == 1.0) |
            (div["m3_bull"] == 1.0) | (div["m3_bear"] == 1.0) |
            (div["m4_bull"] == 1.0) | (div["m4_bear"] == 1.0)
        ).astype(float)
        return combined

    elif name == "session":
        return compute_session_filter(
            data,
            hour_from=int(params.get("hour_from", 7)),
            hour_to=int(params.get("hour_to", 11)),
            tz=params.get("tz", "Europe/Madrid"),
        )

    elif name == "sl_filter":
        return compute_sl_filter(
            data,
            min_pips=float(params.get("min_pips", 4.5)),
            max_pips=float(params.get("max_pips", 35.0)),
            pip_size=float(params.get("pip_size", 0.0001)),
        )

    elif name == "rr_mct":
        return compute_rr_mct_filter(
            data,
            min_rr=float(params.get("min_rr", 1.2)),
            liq_strength=int(params.get("liq_strength", 25)),
            pip_size=float(params.get("pip_size", 0.0001)),
        )

    elif name == "mct_exit":
        return compute_mct_exit(
            data,
            liq_strength=int(params.get("liq_strength", 25)),
            rr_threshold=float(params.get("rr_threshold", 3.0)),
            pip_size=float(params.get("pip_size", 0.0001)),
        )

    elif name == "bos":
        if not _BOS_AVAILABLE:
            raise ValueError("market_structure_bos.py no encontrado en el directorio.")
        sig = evaluate_bos_indicator(data, params)
        # Convertir: 1.0=bull, -1.0=bear, 0.0=sin señal → is_true activa en bull y bear
        return (sig != 0.0).astype(float)

    else:
        raise ValueError(f"Indicador '{indicator_name}' no reconocido. "
                         f"Disponibles: rsi, ema, sma, macd, bb, atr, stoch, liquidity")


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

    elif cond == "is_true":
        # Para indicadores binarios como liquidity sweep
        return curr_ind == 1.0

    else:
        raise ValueError(f"Condición '{condition}' no reconocida. "
                         f"Disponibles: crosses_above, crosses_below, above, below, "
                         f"price_above, price_below, is_true")


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
# FILTROS ESPECIALES
# ─────────────────────────────────────────────────────────────

def compute_session_filter(data: pd.DataFrame, hour_from: int, hour_to: int,
                            tz: str = "Europe/Madrid") -> pd.Series:
    """
    Devuelve 1.0 en barras dentro del rango horario, 0.0 fuera.
    Convierte el índice a la zona horaria especificada.
    """
    try:
        idx = data.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx_local = idx.tz_convert(tz)
        hours = idx_local.hour
        if hour_from <= hour_to:
            mask = (hours >= hour_from) & (hours < hour_to)
        else:
            # rango nocturno (ej. 22-02)
            mask = (hours >= hour_from) | (hours < hour_to)
        return pd.Series(mask.astype(float), index=data.index)
    except Exception:
        # Si falla la conversión de zona horaria, usar UTC directamente
        hours = pd.Series(data.index).dt.hour.values
        if hour_from <= hour_to:
            mask = (hours >= hour_from) & (hours < hour_to)
        else:
            mask = (hours >= hour_from) | (hours < hour_to)
        return pd.Series(mask.astype(float), index=data.index)


def _get_bos_levels(data: pd.DataFrame, liq_strength: int = 25) -> dict:
    """
    Extrae SL (BB/AA) y OL directamente del BOS detector.
    Requiere market_structure_bos.py disponible.
    Fallback a high/low de barra anterior si no está disponible.
    """
    n      = len(data)
    sl_bull = np.full(n, np.nan)  # BB — SL para compras
    sl_bear = np.full(n, np.nan)  # AA — SL para ventas
    ol_bull = np.full(n, np.nan)  # OL para compras
    ol_bear = np.full(n, np.nan)  # OL para ventas

    if _BOS_AVAILABLE:
        try:
            from market_structure_bos import detect_bos
            df = data.copy()
            df.columns = [c.capitalize() for c in df.columns]
            if 'Open' not in df.columns:
                df['Open'] = df['Close']
            r = detect_bos(df, liq_strength=liq_strength)
            sl_bull = r['sl_bull'].values  # BB del sweep actual
            sl_bear = r['sl_bear'].values  # AA del sweep actual
            tp_bull = r['tp_bull'].values  # OL para compras
            tp_bear = r['tp_bear'].values  # OL para ventas
            # Rellenar OL desde tp_bull/tp_bear (que ya vienen del sweep)
            ol_bull = tp_bull
            ol_bear = tp_bear
        except Exception:
            pass

    # Si BOS no disponible o falló, usar OL de sweeps internos
    if np.all(np.isnan(ol_bull)) and np.all(np.isnan(ol_bear)):
        ol = compute_ol_levels(data, liq_strength)
        ol_bull = ol["ol_bull"].values
        ol_bear = ol["ol_bear"].values
        # SL fallback: high/low de barra anterior
        high_v = data["High"].values if "High" in data.columns else data["high"].values
        low_v  = data["Low"].values  if "Low"  in data.columns else data["low"].values
        for i in range(1, n):
            sl_bull[i] = low_v[i-1]
            sl_bear[i] = high_v[i-1]

    return {
        "sl_bull": sl_bull,
        "sl_bear": sl_bear,
        "ol_bull": ol_bull,
        "ol_bear": ol_bear,
    }


def compute_ol_levels(data: pd.DataFrame, liq_strength: int = 25) -> dict:
    """
    Calcula el Opposite Liquidity (OL) usando detect_sweeps_internal.
    - OL compras (bullish sweep) = max(High[SS:SE])
    - OL ventas  (bearish sweep) = min(Low[SS:SE])
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]
    liq    = compute_liquidity(df, strength=liq_strength)
    sweeps = detect_sweeps_internal(df, liq)
    n      = len(df)

    ol_bull = np.full(n, np.nan)
    ol_bear = np.full(n, np.nan)

    for idx, sw in enumerate(sweeps):
        next_se = sweeps[idx + 1].se_bar if idx + 1 < len(sweeps) else n
        if sw.direction == 'bullish':
            for k in range(sw.se_bar, next_se):
                ol_bull[k] = sw.ol_price
        else:
            for k in range(sw.se_bar, next_se):
                ol_bear[k] = sw.ol_price

    return {
        "ol_bull": pd.Series(ol_bull, index=data.index),
        "ol_bear": pd.Series(ol_bear, index=data.index),
    }


def compute_sl_filter(data: pd.DataFrame, min_pips: float, max_pips: float,
                      pip_size: float = 0.0001, liq_strength: int = 25) -> pd.Series:
    """
    Filtro SL — usa BB/AA del BOS real (MCT Protocol).
    SL compras = BB (min Low post-SE del sweep).
    SL ventas  = AA (max High post-SE del sweep).
    """
    close_v = data["Close"].values
    n       = len(data)
    result  = np.zeros(n, dtype=float)

    levels  = _get_bos_levels(data, liq_strength)
    sl_bull = levels["sl_bull"]
    sl_bear = levels["sl_bear"]

    for i in range(1, n):
        price = close_v[i]

        # Compras: distancia entrada → BB
        if not np.isnan(sl_bull[i]):
            sl_pips = abs(price - sl_bull[i]) / pip_size
            if min_pips <= sl_pips <= max_pips:
                result[i] = 1.0

        # Ventas: distancia entrada → AA
        if not np.isnan(sl_bear[i]) and result[i] == 0:
            sl_pips = abs(price - sl_bear[i]) / pip_size
            if min_pips <= sl_pips <= max_pips:
                result[i] = 1.0

    return pd.Series(result, index=data.index)


def compute_rr_mct_filter(data: pd.DataFrame, min_rr: float,
                           liq_strength: int = 25,
                           pip_size: float = 0.0001) -> pd.Series:
    """
    Filtro RR Mínimo MCT.
    RR = distancia(entrada → OL) / distancia(entrada → SL)
    SL = BB para compras / AA para ventas (del BOS real).
    Solo pasa si RR >= min_rr.
    """
    close_v = data["Close"].values
    n       = len(data)
    result  = np.zeros(n, dtype=float)

    levels  = _get_bos_levels(data, liq_strength)
    sl_bull = levels["sl_bull"]
    sl_bear = levels["sl_bear"]
    ol_bull = levels["ol_bull"]
    ol_bear = levels["ol_bear"]

    for i in range(1, n):
        price = close_v[i]

        # Compras
        if not np.isnan(ol_bull[i]) and not np.isnan(sl_bull[i]):
            sl_dist = abs(price - sl_bull[i])
            tp_dist = abs(ol_bull[i] - price)
            if sl_dist > 0 and tp_dist / sl_dist >= min_rr:
                result[i] = 1.0

        # Ventas
        if not np.isnan(ol_bear[i]) and not np.isnan(sl_bear[i]) and result[i] == 0:
            sl_dist = abs(price - sl_bear[i])
            tp_dist = abs(price - ol_bear[i])
            if sl_dist > 0 and tp_dist / sl_dist >= min_rr:
                result[i] = 1.0

    return pd.Series(result, index=data.index)


def compute_mct_exit(data: pd.DataFrame, liq_strength: int = 25,
                     rr_threshold: float = 3.0,
                     pip_size: float = 0.0001) -> pd.Series:
    """
    MCT Protocol — condición de salida.
    TP normal: precio alcanza el OL.
    Si RR entrada→OL / entrada→SL >= rr_threshold (3):
      - Compras: TP en Fib 0.71 desde OL hasta BB  → TP = OL - 0.71*(OL-BB)
      - Ventas:  TP en Fib 0.71 desde OL hasta AA  → TP = OL + 0.71*(AA-OL)
    SL = BB/AA del BOS real.
    """
    close_v = data["Close"].values
    high_v  = data["High"].values if "High" in data.columns else data["high"].values
    low_v   = data["Low"].values  if "Low"  in data.columns else data["low"].values
    n       = len(data)
    result  = np.zeros(n, dtype=float)

    levels  = _get_bos_levels(data, liq_strength)
    sl_bull = levels["sl_bull"]  # BB
    sl_bear = levels["sl_bear"]  # AA
    ol_bull = levels["ol_bull"]
    ol_bear = levels["ol_bear"]

    for i in range(1, n):
        price = close_v[i]

        # ── Salida compras ──
        if not np.isnan(ol_bull[i]) and not np.isnan(sl_bull[i]):
            sl_dist = abs(price - sl_bull[i])
            tp_dist = abs(ol_bull[i] - price)

            if sl_dist > 0 and tp_dist / sl_dist >= rr_threshold:
                # Fib 0.71: desde OL hasta BB
                bb = sl_bull[i]
                tp = ol_bull[i] - 0.71 * (ol_bull[i] - bb)
            else:
                tp = ol_bull[i]

            if price >= tp:
                result[i] = 1.0

        # ── Salida ventas ──
        elif not np.isnan(ol_bear[i]) and not np.isnan(sl_bear[i]):
            sl_dist = abs(price - sl_bear[i])
            tp_dist = abs(price - ol_bear[i])

            if sl_dist > 0 and tp_dist / sl_dist >= rr_threshold:
                # Fib 0.71: desde OL hasta AA
                aa = sl_bear[i]
                tp = ol_bear[i] + 0.71 * (aa - ol_bear[i])
            else:
                tp = ol_bear[i]

            if price <= tp:
                result[i] = 1.0

    return pd.Series(result, index=data.index)

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

    # EV = retorno total acumulado / numero de trades
    total_return_sum = sum(returns)
    expectancy = round(total_return_sum / n, 4) if n > 0 else 0.0
    # EV Normalizado = EV / |avg_loss| = cuanto R ganas por trade de media
    # Ejemplo: EV=0.84%, avg_loss=-4%, EV_norm = 0.84/4 = 0.21R por trade
    expectancy_normalized = round(expectancy / abs(avg_loss), 4) if avg_loss != 0 else 0.0

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
        # Edge Score recalibrado:
        # PF:          1.0=neutral, 2.0=bueno, 3.0+=excelente
        # EV%:         positivo = edge, negativo = sin edge
        # Consistency: % meses positivos
        # Recovery:    retorno/drawdown
        pf_score   = min(max((profit_factor - 1.0) / 2.0, 0.0), 1.0) * 30   # 0-30 pts
        ev_score   = min(max(expectancy / 2.0, 0.0), 1.0) * 25               # 0-25 pts
        cons_score = (consistency_score / 100.0) * 25                         # 0-25 pts
        rec_score  = min(max(recovery_factor / 3.0, 0.0), 1.0) * 20          # 0-20 pts
        edge_score = round(pf_score + ev_score + cons_score + rec_score, 2)
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
        "expectancy_pct":          expectancy,          # EV = profit_total / n_trades
        "expectancy_normalized":   expectancy_normalized, # EV en R = EV / |avg_loss|
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

    # Ensure Open is present (needed for BOS)
    if 'Open' not in raw.columns:
        raw['Open'] = raw['Close']

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
    "liquidity": {
        "name":        "Liquidity Sweep (TFO)",
        "params":      [
            {"key": "strength",  "label": "Fuerza pivot", "default": 25, "min": 5,  "max": 100},
            {"key": "del_after", "label": "Expirar tras", "default": 1000, "min": 50, "max": 5000},
        ],
        "conditions":  ["is_true"],
        "needs_value": False,
        "outputs":     ["ph_taken", "pl_taken"],
        "output_labels": {
            "pl_taken": "PL Barrido (señal alcista — pivot low cruzado)",
            "ph_taken": "PH Barrido (señal bajista — pivot high cruzado)",
        },
    },
    "rsi_divergence": {
        "name":        "RSI Divergence",
        "params":      [
            {"key": "rsi_period",     "label": "RSI Period",     "default": 14, "min": 2,  "max": 50},
            {"key": "pivot_strength", "label": "Pivot Strength", "default": 5,  "min": 2,  "max": 20},
            {"key": "liq_strength",   "label": "Liq Strength",   "default": 25, "min": 5,  "max": 100},
        ],
        "conditions":  ["is_true"],
        "needs_value": False,
    },
    "session": {
        "name":        "Filtro Horario",
        "params":      [
            {"key": "hour_from", "label": "Hora inicio (0-23)", "default": 7,  "min": 0, "max": 23},
            {"key": "hour_to",   "label": "Hora fin   (0-23)", "default": 11, "min": 0, "max": 23},
        ],
        "conditions":  ["is_true"],
        "needs_value": False,
        "note":        "Zona horaria: Europe/Madrid. Añade dos veces para dos rangos.",
    },
    "sl_filter": {
        "name":        "Filtro SL (pips)",
        "params":      [
            {"key": "min_pips", "label": "SL mínimo (pips)", "default": 4.5,  "min": 0.1, "max": 500},
            {"key": "max_pips", "label": "SL máximo (pips)", "default": 35.0, "min": 0.1, "max": 500},
            {"key": "pip_size", "label": "Pip size",         "default": 0.0001, "min": 0.00001, "max": 0.01},
        ],
        "conditions":  ["is_true"],
        "needs_value": False,
        "note":        "SL = BB/AA del BOS real (MCT Protocol).",
    },
    "rr_mct": {
        "name":        "Filtro RR Mínimo MCT",
        "params":      [
            {"key": "min_rr",      "label": "RR mínimo",   "default": 1.2, "min": 0.1, "max": 20},
            {"key": "pip_size",    "label": "Pip size",    "default": 0.0001, "min": 0.00001, "max": 0.01},
            {"key": "liq_strength","label": "Liq Strength","default": 25,  "min": 5,   "max": 100},
        ],
        "conditions":  ["is_true"],
        "needs_value": False,
        "note":        "RR calculado con SL = BB/AA del BOS real.",
    },
    "mct_exit": {
        "name":        "MCT Protocol (salida)",
        "params":      [
            {"key": "rr_threshold", "label": "RR para Fib 0.71", "default": 3.0, "min": 1.0, "max": 10},
            {"key": "liq_strength", "label": "Liq Strength",     "default": 25,  "min": 5,   "max": 100},
            {"key": "pip_size",     "label": "Pip size",          "default": 0.0001, "min": 0.00001, "max": 0.01},
        ],
        "conditions":  ["is_true"],
        "needs_value": False,
        "note":        "TP = OL normal. Si RR >= threshold: TP en Fib 0.71.",
    },
    "bos": {
        "name":        "BOS — Break of Structure (MCT)",
        "params":      [
            {"key": "liq_strength", "label": "Liq Strength", "default": 25, "min": 5, "max": 100},
        ],
        "conditions":  ["is_true"],
        "needs_value": False,
        "note":        "Detecta BOS + retest segun MCT Protocol. Requiere Open en los datos.",
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
