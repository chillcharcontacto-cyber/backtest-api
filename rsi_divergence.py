"""
RSI Divergence Detector
Timeframe: M5 | Compatible con EURUSD, GBPUSD

Detecta 4 modelos de divergencia RSI sobre gráfico de líneas (Close).
Requiere liquidity_channels.py en el mismo directorio.

─── Glosario ────────────────────────────────────────────────────────────────
SS          Barra donde se forma el pivot high/low (origen del sweep).
SE          Barra donde el precio cruza el sweep level (fin del sweep).
Sweep Level Precio exacto del pivot en SS.
OL          Extremo del precio crudo en rango SS→SE.
            · Compras : max(High[SS:SE])
            · Ventas  : min(Low[SS:SE])
BB          min(Close[SE:next_SE]) — punto más bajo post-SE en compras.
AA          max(Close[SE:next_SE]) — punto más alto post-SE en ventas.
Divergencia Siempre sobre Close (gráfico de líneas) vs RSI(Close).

─── Modelos ─────────────────────────────────────────────────────────────────
#1 MSB Build Up
   Últimos 2 pivot lows/highs del Close.
   Bull: precio LL + RSI HL → señal en 2º pivot low.
   Bear: precio HH + RSI LH → señal en 2º pivot high.

#2 Sweep Divergence
   Bull: Close[SE] vs Close[BB], RSI HL → señal en BB.
   Bear: Close[SE] vs Close[AA], RSI LH → señal en AA.

#3 Multiple Div Build Up
   #1 AND #4 activos simultáneamente en la misma barra.

#4 MSB Build Up Extended
   Bull: cualquier par (A,B) con ol_bar <= A < B <= BB donde
         Close[B] < Close[A] (LL) y RSI[B] > RSI[A] (HL) -> señal en BB.
   Bear: cualquier par (A,B) con ol_bar <= A < B <= AA donde
         Close[B] > Close[A] (HH) y RSI[B] < RSI[A] (LH) -> señal en AA.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from liquidity_channels import (
    detect_liquidity_levels,
    detect_sweeps,
    pivot_high,
    pivot_low,
)


# ─── RSI ──────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pivot_indices(series: pd.Series, strength: int, mode: str) -> list:
    """
    Retorna posiciones enteras de los pivots confirmados sobre 'series'.
    mode: 'high' o 'low'
    """
    fn     = pivot_high if mode == 'high' else pivot_low
    result = fn(series, strength)
    return [series.index.get_loc(i) for i in result.dropna().index]


def _has_divergence_pair(
    close: np.ndarray,
    rsi: np.ndarray,
    start_i: int,
    end_i: int,
    direction: str,
) -> bool:
    """
    Busca cualquier par (A, B) con start_i <= A < B <= end_i donde:
      bull: Close[B] < Close[A] (LL) y RSI[B] > RSI[A] (HL)
      bear: Close[B] > Close[A] (HH) y RSI[B] < RSI[A] (LH)
    """
    for a in range(start_i, end_i):
        for b in range(a + 1, end_i + 1):
            if np.isnan(rsi[a]) or np.isnan(rsi[b]):
                continue
            if direction == 'bull':
                if close[b] < close[a] and rsi[b] > rsi[a]:
                    return True
            else:
                if close[b] > close[a] and rsi[b] < rsi[a]:
                    return True
    return False


# ─── Modelo #1: MSB Build Up ──────────────────────────────────────────────────

def _detect_model1(
    df: pd.DataFrame,
    close: np.ndarray,
    rsi: np.ndarray,
    pivot_strength: int,
) -> tuple:
    """
    Divergencia entre los ultimos 2 pivot lows/highs del Close.
    Señal en la barra del 2 pivot.
    """
    n    = len(df)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    pl_pos = _pivot_indices(df['Close'], pivot_strength, 'low')
    ph_pos = _pivot_indices(df['Close'], pivot_strength, 'high')

    # Bullish: precio LL + RSI HL
    for k in range(1, len(pl_pos)):
        a, b = pl_pos[k - 1], pl_pos[k]
        if np.isnan(rsi[a]) or np.isnan(rsi[b]):
            continue
        if close[b] < close[a] and rsi[b] > rsi[a]:
            bull[b] = True

    # Bearish: precio HH + RSI LH
    for k in range(1, len(ph_pos)):
        a, b = ph_pos[k - 1], ph_pos[k]
        if np.isnan(rsi[a]) or np.isnan(rsi[b]):
            continue
        if close[b] > close[a] and rsi[b] < rsi[a]:
            bear[b] = True

    return bull, bear


# ─── Contextos de sweep para modelos #2 y #4 ─────────────────────────────────

def _build_sweep_contexts(sweeps: list, n: int) -> list:
    """
    Convierte la lista de Sweep en contextos listos para los modelos #2 y #4.
    Cada contexto contiene:
        direction  'bull' o 'bear'
        ss_i       indice entero del SS
        se_i       indice entero del SE
        ol_i       indice entero del OL
        next_se_i  indice del siguiente SE (limite de busqueda post-SE)
    """
    sorted_sweeps = sorted(sweeps, key=lambda s: s.se_bar)
    contexts = []
    for idx, sw in enumerate(sorted_sweeps):
        next_se = sorted_sweeps[idx + 1].se_bar if idx + 1 < len(sorted_sweeps) else n
        contexts.append({
            'direction': 'bull' if sw.direction == 'bullish' else 'bear',
            'ss_i':      sw.ss_bar,
            'se_i':      sw.se_bar,
            'ol_i':      sw.ol_bar,
            'next_se_i': next_se,
        })
    return contexts


# ─── Modelo #2: Sweep Divergence ──────────────────────────────────────────────

def _detect_model2(
    close: np.ndarray,
    rsi: np.ndarray,
    contexts: list,
    n: int,
) -> tuple:
    """
    Bull: Close[SE] vs Close[BB=min(Close[SE:next_SE])], RSI HL -> señal en BB.
    Bear: Close[SE] vs Close[AA=max(Close[SE:next_SE])], RSI LH -> señal en AA.
    """
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    for ctx in contexts:
        se_i      = ctx['se_i']
        next_se_i = ctx['next_se_i']
        direction = ctx['direction']

        if next_se_i <= se_i + 1:
            continue

        segment = close[se_i: next_se_i]

        if direction == 'bull':
            bb_i = se_i + int(np.argmin(segment))
            if bb_i == se_i:
                continue
            if np.isnan(rsi[se_i]) or np.isnan(rsi[bb_i]):
                continue
            if close[bb_i] < close[se_i] and rsi[bb_i] > rsi[se_i]:
                bull[bb_i] = True

        else:
            aa_i = se_i + int(np.argmax(segment))
            if aa_i == se_i:
                continue
            if np.isnan(rsi[se_i]) or np.isnan(rsi[aa_i]):
                continue
            if close[aa_i] > close[se_i] and rsi[aa_i] < rsi[se_i]:
                bear[aa_i] = True

    return bull, bear


# ─── Modelo #4: MSB Build Up Extended ────────────────────────────────────────

def _detect_model4(
    close: np.ndarray,
    rsi: np.ndarray,
    contexts: list,
    n: int,
) -> tuple:
    """
    Bull: cualquier par divergente (Close LL + RSI HL) entre ol_i y BB -> señal en BB.
    Bear: cualquier par divergente (Close HH + RSI LH) entre ol_i y AA -> señal en AA.
    """
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

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
            if bb_i <= ol_i:
                continue
            if _has_divergence_pair(close, rsi, ol_i, bb_i, 'bull'):
                bull[bb_i] = True

        else:
            aa_i = se_i + int(np.argmax(segment))
            if aa_i <= ol_i:
                continue
            if _has_divergence_pair(close, rsi, ol_i, aa_i, 'bear'):
                bear[aa_i] = True

    return bull, bear


# ─── Funcion principal ────────────────────────────────────────────────────────

def detect_rsi_divergence(
    data: pd.DataFrame,
    rsi_period: int = 14,
    pivot_strength: int = 5,
    liq_strength: int = 25,
) -> pd.DataFrame:
    """
    Detecta los 4 modelos de divergencia RSI.

    Parametros
    ----------
    data : pd.DataFrame
        Columnas requeridas: Close, High, Low (case-insensitive).
    rsi_period : int
        Periodo del RSI. Default = 14.
    pivot_strength : int
        Fuerza de pivot para Modelo #1 (pivot lows/highs del Close). Default = 5.
    liq_strength : int
        Fuerza de pivot para liquidity channels (SS/SE/OL). Default = 25.

    Retorna
    -------
    DataFrame original con columnas adicionales:

        div_m1_bull   Modelo #1 alcista
        div_m1_bear   Modelo #1 bajista
        div_m2_bull   Modelo #2 alcista — Sweep divergence, señal en BB
        div_m2_bear   Modelo #2 bajista — Sweep divergence, señal en AA
        div_m3_bull   Modelo #3 alcista — #1 AND #4 simultaneos
        div_m3_bear   Modelo #3 bajista — #1 AND #4 simultaneos
        div_m4_bull   Modelo #4 alcista — par divergente entre OL y BB
        div_m4_bear   Modelo #4 bajista — par divergente entre OL y AA
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]

    missing = {'High', 'Low', 'Close'} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    n     = len(df)
    close = df['Close'].values
    rsi   = compute_rsi(df['Close'], rsi_period).values

    # Liquidity channels + sweeps
    liq      = detect_liquidity_levels(df, strength=liq_strength)
    sweeps   = detect_sweeps(liq)
    contexts = _build_sweep_contexts(sweeps, n)

    # Modelos
    m1_bull, m1_bear = _detect_model1(df, close, rsi, pivot_strength)
    m2_bull, m2_bear = _detect_model2(close, rsi, contexts, n)
    m4_bull, m4_bear = _detect_model4(close, rsi, contexts, n)
    m3_bull          = m1_bull & m4_bull
    m3_bear          = m1_bear & m4_bear

    result = data.copy()
    result['div_m1_bull'] = m1_bull
    result['div_m1_bear'] = m1_bear
    result['div_m2_bull'] = m2_bull
    result['div_m2_bear'] = m2_bear
    result['div_m3_bull'] = m3_bull
    result['div_m3_bear'] = m3_bear
    result['div_m4_bull'] = m4_bull
    result['div_m4_bear'] = m4_bear

    return result


# ─── Test rapido ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n = 800
    price = 1.1000 + np.cumsum(np.random.randn(n) * 0.0002)
    df_test = pd.DataFrame({
        'High':  price + np.abs(np.random.randn(n) * 0.0003),
        'Low':   price - np.abs(np.random.randn(n) * 0.0003),
        'Close': price,
    })

    result = detect_rsi_divergence(df_test)

    print("=" * 50)
    print("RSI DIVERGENCE DETECTOR — Test")
    print("=" * 50)
    for col in [c for c in result.columns if c.startswith('div_')]:
        print(f"  {col:<20} {result[col].sum():>4} señales")
    print("=" * 50)
    print(f"  Total barras: {n}")
