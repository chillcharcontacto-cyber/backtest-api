"""
Liquidity Channels [TFO] — Python port
Timeframe: M5 | Pairs: EURUSD, GBPUSD

Glosario
--------
Pivot High / Pivot Low   Máximo/mínimo local detectado con ventana 'strength' barras
                         a cada lado. Origen: indicador Pine Script Liquidity Channels [TFO].

SS  (Sweep Start)        Barra donde se forma el pivot high o pivot low.
                         El precio de ese pivot es el Sweep Level.

Sweep Level              Precio EXACTO del pivot high o pivot low en SS.
                         Es la línea horizontal que une SS y SE — siempre el mismo precio.
                         SS, Sweep Level y SE comparten el mismo precio de referencia.

SE  (Sweep End)          Barra donde el precio cruza el Sweep Level.
                         · Setup de compra : low  < Sweep Level  (pivot low cruzado)
                         · Setup de venta  : high > Sweep Level  (pivot high cruzado)

OL  (Opposite Level)     Extremo opuesto del precio dentro del rango SS→SE.
                         · Setup de compra (bullish) : max(High[SS:SE]) — el TECHO
                         · Setup de venta  (bearish) : min(Low[SS:SE])  — el SUELO
                         Es el TP del trade. No tiene relación con pivots.

direction                'bullish' → pivot low cruzado, precio sube hacia OL
                         'bearish' → pivot high cruzado, precio baja hacia OL

IMPORTANTE — sweep_level correcto
----------------------------------
El sweep_level debe ser el precio EXACTO del pivot que se cruzó en SE,
no el nivel más cercano al precio en ese momento. Si hay múltiples pivots
activos, hay que identificar cuál fue el que se cruzó (taken).
Este archivo corrige ese bug respecto a versiones anteriores.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Estructuras ──────────────────────────────────────────────────────────────

@dataclass
class LiquidityLevel:
    price:     float
    bar_index: int
    taken:     bool          = False
    taken_at:  Optional[int] = None   # barra donde fue cruzado


@dataclass
class Sweep:
    direction:   str    # 'bullish' o 'bearish'
    ss_bar:      int    # Sweep Start — barra del pivot
    se_bar:      int    # Sweep End   — barra donde el precio cruza el Sweep Level
    sweep_level: float  # precio EXACTO del pivot (= precio en SS = precio en SE)
    ol_price:    float  # Opposite Level — extremo del precio en SS→SE
    ol_bar:      int    # barra donde se produce el OL


# ─── Pivot detection ──────────────────────────────────────────────────────────

def pivot_high(series: pd.Series, strength: int) -> pd.Series:
    """
    Replica ta.pivothigh(strength, strength) de Pine Script.
    Confirma el pivot en la barra central de la ventana [i-n, i+n].
    El valor aparece 'strength' barras después de que ocurrió el máximo real.
    """
    n      = strength
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(n, len(values) - n):
        window = values[i - n : i + n + 1]
        if values[i] == window.max():
            result.iloc[i] = values[i]
    return result


def pivot_low(series: pd.Series, strength: int) -> pd.Series:
    """
    Replica ta.pivotlow(strength, strength) de Pine Script.
    """
    n      = strength
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(n, len(values) - n):
        window = values[i - n : i + n + 1]
        if values[i] == window.min():
            result.iloc[i] = values[i]
    return result


# ─── Liquidity levels ─────────────────────────────────────────────────────────

def detect_liquidity_levels(
    data:          pd.DataFrame,
    strength:      int  = 25,
    del_untouched: bool = True,
    del_after:     int  = 1000,
) -> pd.DataFrame:
    """
    Detecta niveles de liquidez activos en cada barra.

    Cada pivot high genera un nivel de venta.
    Cada pivot low  genera un nivel de compra.
    Cuando el precio cruza un nivel, ese nivel queda marcado como 'taken'
    — esa barra es el SE del sweep correspondiente.

    Parámetros
    ----------
    data          : DataFrame con columnas High, Low, Close (case-insensitive).
    strength      : períodos de pivot. Default = 25.
    del_untouched : si True, descarta niveles no cruzados tras del_after barras.
    del_after     : vida máxima de un nivel no cruzado en barras. Default = 1000.

    Columnas añadidas
    -----------------
    ph_price    precio del pivot high activo más cercano
    ph_bar      barra de formación de ese pivot high
    ph_taken    True en la barra exacta en que high > ph_price  (= SE de venta)
    ph_count    número de pivot highs activos (no cruzados)

    pl_price    precio del pivot low activo más cercano
    pl_bar      barra de formación de ese pivot low
    pl_taken    True en la barra exacta en que low < pl_price   (= SE de compra)
    pl_count    número de pivot lows activos (no cruzados)

    all_ph      todos los pivot highs activos: [(price, bar_index), ...]
    all_pl      todos los pivot lows  activos: [(price, bar_index), ...]

    taken_ph_price  precio EXACTO del pivot high que se cruzó en esta barra
                    (NaN si no se cruzó ninguno). Usado por detect_sweeps().
    taken_ph_bar    barra de formación del pivot high cruzado en esta barra.
    taken_pl_price  precio EXACTO del pivot low  que se cruzó en esta barra.
    taken_pl_bar    barra de formación del pivot low  cruzado en esta barra.
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]

    missing = {'High', 'Low', 'Close'} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    ph = pivot_high(df['High'], strength)
    pl = pivot_low(df['Low'],  strength)

    n = len(df)

    out_ph_price       = np.full(n, np.nan)
    out_ph_bar         = np.full(n, np.nan)
    out_ph_taken       = np.zeros(n, dtype=bool)
    out_ph_count       = np.zeros(n, dtype=int)
    out_pl_price       = np.full(n, np.nan)
    out_pl_bar         = np.full(n, np.nan)
    out_pl_taken       = np.zeros(n, dtype=bool)
    out_pl_count       = np.zeros(n, dtype=int)
    out_all_ph         = [None] * n
    out_all_pl         = [None] * n

    # ── NUEVAS columnas: precio y barra EXACTOS del nivel cruzado ─────────────
    # Necesarias para que detect_sweeps() use el sweep_level correcto.
    out_taken_ph_price = np.full(n, np.nan)
    out_taken_ph_bar   = np.full(n, np.nan)
    out_taken_pl_price = np.full(n, np.nan)
    out_taken_pl_bar   = np.full(n, np.nan)

    active_ph: list[LiquidityLevel] = []
    active_pl: list[LiquidityLevel] = []

    high = df['High'].values
    low  = df['Low'].values
    ph_v = ph.values
    pl_v = pl.values

    for i in range(n):

        # 1. Registrar nuevos pivots confirmados en esta barra
        if not np.isnan(ph_v[i]):
            active_ph.append(LiquidityLevel(price=ph_v[i], bar_index=i))
        if not np.isnan(pl_v[i]):
            active_pl.append(LiquidityLevel(price=pl_v[i], bar_index=i))

        # 2. Actualizar pivot highs
        surviving_ph = []
        for lvl in active_ph:
            if not lvl.taken:
                if high[i] > lvl.price:
                    lvl.taken    = True
                    lvl.taken_at = i
                    out_ph_taken[i]       = True
                    # Guardar precio y barra EXACTOS del nivel cruzado
                    out_taken_ph_price[i] = lvl.price
                    out_taken_ph_bar[i]   = lvl.bar_index
                elif del_untouched and (i - lvl.bar_index) >= del_after:
                    continue
            surviving_ph.append(lvl)
        active_ph = surviving_ph

        # 3. Actualizar pivot lows
        surviving_pl = []
        for lvl in active_pl:
            if not lvl.taken:
                if low[i] < lvl.price:
                    lvl.taken    = True
                    lvl.taken_at = i
                    out_pl_taken[i]       = True
                    # Guardar precio y barra EXACTOS del nivel cruzado
                    out_taken_pl_price[i] = lvl.price
                    out_taken_pl_bar[i]   = lvl.bar_index
                elif del_untouched and (i - lvl.bar_index) >= del_after:
                    continue
            surviving_pl.append(lvl)
        active_pl = surviving_pl

        # 4. Snapshot de niveles activos (no cruzados)
        untaken_ph = [l for l in active_ph if not l.taken]
        untaken_pl = [l for l in active_pl if not l.taken]

        out_ph_count[i] = len(untaken_ph)
        out_pl_count[i] = len(untaken_pl)
        out_all_ph[i]   = [(l.price, l.bar_index) for l in untaken_ph]
        out_all_pl[i]   = [(l.price, l.bar_index) for l in untaken_pl]

        # 5. Nivel más cercano al precio actual (para referencia visual)
        if untaken_ph:
            closest         = min(untaken_ph, key=lambda l: abs(l.price - high[i]))
            out_ph_price[i] = closest.price
            out_ph_bar[i]   = closest.bar_index

        if untaken_pl:
            closest         = min(untaken_pl, key=lambda l: abs(l.price - low[i]))
            out_pl_price[i] = closest.price
            out_pl_bar[i]   = closest.bar_index

    df['ph_price']        = out_ph_price
    df['ph_bar']          = out_ph_bar
    df['ph_taken']        = out_ph_taken
    df['ph_count']        = out_ph_count
    df['pl_price']        = out_pl_price
    df['pl_bar']          = out_pl_bar
    df['pl_taken']        = out_pl_taken
    df['pl_count']        = out_pl_count
    df['all_ph']          = out_all_ph
    df['all_pl']          = out_all_pl
    df['taken_ph_price']  = out_taken_ph_price
    df['taken_ph_bar']    = out_taken_ph_bar
    df['taken_pl_price']  = out_taken_pl_price
    df['taken_pl_bar']    = out_taken_pl_bar

    return df


# ─── Sweep detection ──────────────────────────────────────────────────────────

def detect_sweeps(df: pd.DataFrame) -> list[Sweep]:
    """
    Detecta todos los sweeps completados en el DataFrame.
    Requiere la salida de detect_liquidity_levels().

    Corrección crítica respecto a versiones anteriores
    ---------------------------------------------------
    El sweep_level se obtiene de 'taken_ph_price' / 'taken_pl_price',
    que guardan el precio EXACTO del pivot que fue cruzado en el SE.
    Antes se usaba 'ph_price' / 'pl_price' (nivel más cercano), lo que
    podía dar un sweep_level incorrecto si había múltiples pivots activos.

    Por cada SE detectado construye un Sweep con:
      SS          = barra de formación del pivot cruzado
      sweep_level = precio EXACTO de ese pivot (= precio en SS = precio en SE)
      OL          = max(High[SS:SE]) en bullish / min(Low[SS:SE]) en bearish

    Retorna lista de Sweep ordenada cronológicamente por se_bar.
    """
    required = {'ph_taken', 'pl_taken',
                'taken_ph_price', 'taken_ph_bar',
                'taken_pl_price', 'taken_pl_bar'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas — ejecuta detect_liquidity_levels() primero: {missing}"
        )

    sweeps: list[Sweep] = []

    ph_taken       = df['ph_taken'].values
    pl_taken       = df['pl_taken'].values
    taken_ph_price = df['taken_ph_price'].values
    taken_ph_bar   = df['taken_ph_bar'].values
    taken_pl_price = df['taken_pl_price'].values
    taken_pl_bar   = df['taken_pl_bar'].values
    high           = df['High'].values
    low            = df['Low'].values

    for se in range(len(df)):

        # ── Setup de venta: pivot high cruzado (bearish sweep) ────────────────
        if ph_taken[se] and not np.isnan(taken_ph_price[se]):
            ss          = int(taken_ph_bar[se])
            sweep_level = taken_ph_price[se]     # precio EXACTO del pivot cruzado
            rng_low     = low[ss : se + 1]
            ol_bar      = ss + int(np.argmin(rng_low))
            ol_price    = rng_low.min()           # OL = mínimo del rango SS→SE

            sweeps.append(Sweep(
                direction   = 'bearish',
                ss_bar      = ss,
                se_bar      = se,
                sweep_level = sweep_level,
                ol_price    = ol_price,
                ol_bar      = ol_bar,
            ))

        # ── Setup de compra: pivot low cruzado (bullish sweep) ────────────────
        if pl_taken[se] and not np.isnan(taken_pl_price[se]):
            ss          = int(taken_pl_bar[se])
            sweep_level = taken_pl_price[se]     # precio EXACTO del pivot cruzado
            rng_high    = high[ss : se + 1]
            ol_bar      = ss + int(np.argmax(rng_high))
            ol_price    = rng_high.max()          # OL = máximo del rango SS→SE

            sweeps.append(Sweep(
                direction   = 'bullish',
                ss_bar      = ss,
                se_bar      = se,
                sweep_level = sweep_level,
                ol_price    = ol_price,
                ol_bar      = ol_bar,
            ))

    return sorted(sweeps, key=lambda s: s.se_bar)


def sweeps_to_dataframe(sweeps: list[Sweep]) -> pd.DataFrame:
    """Convierte la lista de Sweep a DataFrame para análisis o exportación."""
    if not sweeps:
        return pd.DataFrame(columns=[
            'direction', 'ss_bar', 'se_bar', 'sweep_level', 'ol_price', 'ol_bar'
        ])
    return pd.DataFrame([{
        'direction':   s.direction,
        'ss_bar':      s.ss_bar,
        'se_bar':      s.se_bar,
        'sweep_level': s.sweep_level,
        'ol_price':    s.ol_price,
        'ol_bar':      s.ol_bar,
    } for s in sweeps])


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n = 600
    price = 1.1000 + np.cumsum(np.random.randn(n) * 0.0002)
    df_test = pd.DataFrame({
        'High':  price + np.abs(np.random.randn(n) * 0.0003),
        'Low':   price - np.abs(np.random.randn(n) * 0.0003),
        'Close': price,
    })

    result   = detect_liquidity_levels(df_test, strength=25)
    sweeps   = detect_sweeps(result)
    df_sw    = sweeps_to_dataframe(sweeps)

    print(f"Total barras      : {len(result)}")
    print(f"Pivot highs taken : {result['ph_taken'].sum()}")
    print(f"Pivot lows  taken : {result['pl_taken'].sum()}")
    print(f"\nSweeps detectados : {len(sweeps)}")
    print(f"  Bajistas        : {sum(1 for s in sweeps if s.direction == 'bearish')}")
    print(f"  Alcistas        : {sum(1 for s in sweeps if s.direction == 'bullish')}")

    if not df_sw.empty:
        print("\nPrimeros sweeps:")
        print(df_sw.head().to_string(index=False))

    # Verificar que sweep_level == taken_ph/pl_price (no el más cercano)
    se_bars_ph = result[result['ph_taken']].index
    for idx in se_bars_ph[:3]:
        i = result.index.get_loc(idx)
        print(f"\nBar {i}: taken_ph_price={result['taken_ph_price'].iloc[i]:.5f} "
              f"ph_price={result['ph_price'].iloc[i]:.5f} "
              f"(iguales={result['taken_ph_price'].iloc[i] == result['ph_price'].iloc[i]})")
