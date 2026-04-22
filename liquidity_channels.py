"""
Liquidity Channels [TFO] — Python port
Timeframe: M5 | Pairs: EURUSD, GBPUSD

Glosario
--------
Pivot High / Pivot Low   Máximo/mínimo local detectado con ventana 'strength' barras
                         a cada lado. Origen: indicador Pine Script Liquidity Channels [TFO].

SS  (Sweep Start)        Barra donde el indicador genera un nivel (pivot high o pivot low).
                         El precio de ese pivot es el Sweep Level.

Sweep Level              Precio exacto del pivot high o pivot low generado en SS.
                         Es la línea horizontal que el precio debe cruzar para completar el sweep.

SE  (Sweep End)          Barra donde el precio cruza el Sweep Level.
                         · Setup de compra : low  < Sweep Level  (pivot low cruzado)
                         · Setup de venta  : high > Sweep Level  (pivot high cruzado)

OL  (Opposite Level)     Nivel de precio opuesto dentro del rango SS→SE.
                         · Setup de compra : max(High) en el rango SS→SE
                         · Setup de venta  : min(Low)  en el rango SS→SE
                         No tiene relación con pivots — es el extremo del precio crudo.

direction                'bullish' → pivot low cruzado, se espera subida hacia OL
                         'bearish' → pivot high cruzado, se espera bajada hacia OL
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Estructuras ──────────────────────────────────────────────────────────────

@dataclass
class LiquidityLevel:
    price: float
    bar_index: int
    taken: bool = False
    taken_at: Optional[int] = None  # barra donde fue cruzado


@dataclass
class Sweep:
    direction: str      # 'bullish' o 'bearish'
    ss_bar: int         # Sweep Start — barra del pivot
    se_bar: int         # Sweep End   — barra donde el precio cruza el Sweep Level
    sweep_level: float  # precio del pivot (SS)
    ol_price: float     # Opposite Level — extremo del precio en el rango SS→SE
    ol_bar: int         # barra donde se produce ese extremo


# ─── Pivot detection ──────────────────────────────────────────────────────────

def pivot_high(series: pd.Series, strength: int) -> pd.Series:
    """
    Replica ta.pivothigh(strength, strength) de Pine Script.
    Confirma el pivot en la barra central de la ventana [i-n, i+n].
    El valor aparece 'strength' barras después de que ocurrió el máximo real.
    """
    n = strength
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
    n = strength
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(n, len(values) - n):
        window = values[i - n : i + n + 1]
        if values[i] == window.min():
            result.iloc[i] = values[i]
    return result


# ─── Liquidity levels ─────────────────────────────────────────────────────────

def detect_liquidity_levels(
    data: pd.DataFrame,
    strength: int = 25,
    del_untouched: bool = True,
    del_after: int = 1000,
) -> pd.DataFrame:
    """
    Detecta niveles de liquidez activos en cada barra.

    Cada pivot high genera un nivel de venta (price_high).
    Cada pivot low  genera un nivel de compra (price_low).
    Cuando el precio cruza un nivel, ese nivel queda marcado como 'taken'
    — esa barra es el SE del sweep correspondiente.

    Parámetros
    ----------
    data : pd.DataFrame
        Columnas requeridas: High, Low, Close (case-insensitive).
    strength : int
        Períodos de pivot. Default = 25.
    del_untouched : bool
        Si True, descarta niveles no cruzados tras del_after barras.
    del_after : int
        Vida máxima de un nivel no cruzado en barras. Default = 1000.

    Columnas añadidas al DataFrame
    --------------------------------
    ph_price    precio del pivot high activo más cercano (futuro SS de venta)
    ph_bar      barra de formación de ese pivot high
    ph_taken    True en la barra exacta en que high > ph_price  (= SE de venta)
    ph_count    número de pivot highs activos (no cruzados)

    pl_price    precio del pivot low activo más cercano (futuro SS de compra)
    pl_bar      barra de formación de ese pivot low
    pl_taken    True en la barra exacta en que low < pl_price   (= SE de compra)
    pl_count    número de pivot lows activos (no cruzados)

    all_ph      lista de todos los pivot highs activos: [(price, bar_index), ...]
    all_pl      lista de todos los pivot lows  activos: [(price, bar_index), ...]
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]

    missing = {'High', 'Low', 'Close'} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    ph = pivot_high(df['High'], strength)
    pl = pivot_low(df['Low'], strength)

    n = len(df)
    out_ph_price = np.full(n, np.nan)
    out_ph_bar   = np.full(n, np.nan)
    out_ph_taken = np.zeros(n, dtype=bool)
    out_ph_count = np.zeros(n, dtype=int)
    out_pl_price = np.full(n, np.nan)
    out_pl_bar   = np.full(n, np.nan)
    out_pl_taken = np.zeros(n, dtype=bool)
    out_pl_count = np.zeros(n, dtype=int)
    out_all_ph   = [None] * n
    out_all_pl   = [None] * n

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
                if high[i] > lvl.price:       # SE de venta
                    lvl.taken = True
                    lvl.taken_at = i
                    out_ph_taken[i] = True
                elif del_untouched and (i - lvl.bar_index) >= del_after:
                    continue                   # expira sin cruzarse
            surviving_ph.append(lvl)
        active_ph = surviving_ph

        # 3. Actualizar pivot lows
        surviving_pl = []
        for lvl in active_pl:
            if not lvl.taken:
                if low[i] < lvl.price:        # SE de compra
                    lvl.taken = True
                    lvl.taken_at = i
                    out_pl_taken[i] = True
                elif del_untouched and (i - lvl.bar_index) >= del_after:
                    continue
            surviving_pl.append(lvl)
        active_pl = surviving_pl

        # 4. Snapshot de niveles activos (no cruzados) en esta barra
        untaken_ph = [l for l in active_ph if not l.taken]
        untaken_pl = [l for l in active_pl if not l.taken]

        out_ph_count[i] = len(untaken_ph)
        out_pl_count[i] = len(untaken_pl)
        out_all_ph[i]   = [(l.price, l.bar_index) for l in untaken_ph]
        out_all_pl[i]   = [(l.price, l.bar_index) for l in untaken_pl]

        # 5. Nivel más cercano al precio actual
        if untaken_ph:
            closest = min(untaken_ph, key=lambda l: abs(l.price - high[i]))
            out_ph_price[i] = closest.price
            out_ph_bar[i]   = closest.bar_index

        if untaken_pl:
            closest = min(untaken_pl, key=lambda l: abs(l.price - low[i]))
            out_pl_price[i] = closest.price
            out_pl_bar[i]   = closest.bar_index

    df['ph_price'] = out_ph_price
    df['ph_bar']   = out_ph_bar
    df['ph_taken'] = out_ph_taken
    df['ph_count'] = out_ph_count
    df['pl_price'] = out_pl_price
    df['pl_bar']   = out_pl_bar
    df['pl_taken'] = out_pl_taken
    df['pl_count'] = out_pl_count
    df['all_ph']   = out_all_ph
    df['all_pl']   = out_all_pl

    return df


# ─── Sweep detection ──────────────────────────────────────────────────────────

def detect_sweeps(df: pd.DataFrame) -> list[Sweep]:
    """
    Detecta todos los sweeps completados en el DataFrame.
    Requiere la salida de detect_liquidity_levels().

    Por cada SE detectado:
      - SS         = barra de formación del pivot cruzado
      - Sweep Level = precio de ese pivot
      - OL         = max(High[SS:SE]) en compras / min(Low[SS:SE]) en ventas

    Retorna lista de Sweep ordenada cronológicamente por se_bar.
    """
    required = {'ph_taken', 'pl_taken', 'ph_bar', 'pl_bar', 'ph_price', 'pl_price'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas — ejecuta detect_liquidity_levels() primero: {missing}")

    sweeps: list[Sweep] = []

    ph_taken = df['ph_taken'].values
    pl_taken = df['pl_taken'].values
    ph_bar   = df['ph_bar'].values
    pl_bar   = df['pl_bar'].values
    ph_price = df['ph_price'].values
    pl_price = df['pl_price'].values
    high     = df['High'].values
    low      = df['Low'].values

    for se in range(len(df)):

        # ── Setup de venta: pivot high cruzado ───────────────────────────────
        if ph_taken[se] and not np.isnan(ph_bar[se]):
            ss          = int(ph_bar[se])
            sweep_level = ph_price[se]
            rng_low     = low[ss : se + 1]
            ol_bar      = ss + int(np.argmin(rng_low))
            ol_price    = rng_low.min()

            sweeps.append(Sweep(
                direction   = 'bearish',
                ss_bar      = ss,
                se_bar      = se,
                sweep_level = sweep_level,
                ol_price    = ol_price,
                ol_bar      = ol_bar,
            ))

        # ── Setup de compra: pivot low cruzado ───────────────────────────────
        if pl_taken[se] and not np.isnan(pl_bar[se]):
            ss          = int(pl_bar[se])
            sweep_level = pl_price[se]
            rng_high    = high[ss : se + 1]
            ol_bar      = ss + int(np.argmax(rng_high))
            ol_price    = rng_high.max()

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


# ─── Uso ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # df = pd.read_csv("eurusd_m5.csv", parse_dates=["datetime"], index_col="datetime")

    np.random.seed(42)
    n = 600
    price = 1.1000 + np.cumsum(np.random.randn(n) * 0.0002)
    df_test = pd.DataFrame({
        'High':  price + np.abs(np.random.randn(n) * 0.0003),
        'Low':   price - np.abs(np.random.randn(n) * 0.0003),
        'Close': price,
    })

    # 1. Detectar niveles
    result = detect_liquidity_levels(df_test, strength=25)

    print(f"Total barras     : {len(result)}")
    print(f"Pivot highs taken: {result['ph_taken'].sum()}")
    print(f"Pivot lows  taken: {result['pl_taken'].sum()}")

    # 2. Detectar sweeps (SS, SE, Sweep Level, OL)
    sweeps    = detect_sweeps(result)
    df_sweeps = sweeps_to_dataframe(sweeps)

    print(f"\nSweeps detectados : {len(sweeps)}")
    print(f"  Bajistas        : {sum(1 for s in sweeps if s.direction == 'bearish')}")
    print(f"  Alcistas        : {sum(1 for s in sweeps if s.direction == 'bullish')}")

    if not df_sweeps.empty:
        print("\nPrimeros sweeps:")
        print(df_sweeps.head().to_string(index=False))
