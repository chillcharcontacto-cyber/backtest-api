"""
Cut Early Protocol — v1
========================
Timeframe : M5  |  Compatible con EURUSD, GBPUSD
Requiere  : market_structure_bos.py en el mismo directorio.

─── Glosario ────────────────────────────────────────────────────────────────
Estructura relevante
    LH relevante (trade bajista):
        High más alto entre dos LL consecutivos, con ≥1 vela alcista en el
        retroceso. Validación circular: solo se confirma cuando aparece el
        siguiente LL. Se invalida si una mecha posterior supera su High
        (High[j] > lh_price, j > lh_bar).

    HL relevante (trade alcista):
        Low más bajo entre dos HH consecutivos, con ≥1 vela bajista en el
        retroceso. Validación circular: solo se confirma cuando aparece el
        siguiente HH. Se invalida si una mecha posterior penetra su Low
        (Low[j] < hl_price, j > hl_bar).

    Implementado reutilizando detect_market_structure() de
    market_structure_bos.py — mismas reglas, misma validación circular,
    mismo orden intravela. Sin reimplementación propia.

MSB Against
    Trade bajista : Close > LH_relevante  (break AND close alcista)
                    El LH relevante debe estar por encima del entry_price.
    Trade alcista : Close < HL_relevante  (break AND close bajista)
                    El HL relevante debe estar por debajo del entry_price.

HH / LL del MSB Against
    High máximo (bear) / Low mínimo (bull) acumulado barra a barra desde
    la confirmación del MSB Against hasta (e incluyendo) la vela pullback.
    Puede estar en una vela distinta a la que confirma el break and close.

Pullback
    Bear : primera vela bajista (Close < Open) posterior al MSB Against,
           siempre que ninguna barra haya tocado con su Low el Restart Level.
    Bull : primera vela alcista (Close > Open) posterior al MSB Against,
           siempre que ninguna barra haya tocado con su High el Restart Level.

Cut Early Level
    Bear : High máximo acumulado desde MSB Against hasta pullback (inclusive).
    Bull : Low  mínimo acumulado desde MSB Against hasta pullback (inclusive).
    Se fija al confirmar el pullback.

Trigger
    Bear : High >= Cut Early Level  (toque de mecha, no requiere cierre)
    Bull : Low  <= Cut Early Level

Restart Level
    Bear : min(Low[entry_bar : msb_against_bar])  — mínimo de la zona verde.
    Bull : max(High[entry_bar : msb_against_bar]) — máximo de la zona verde.

Reset
    Si el precio toca el Restart Level (Low <= restart en bear /
    High >= restart en bull) antes de confirmarse el pullback, el MSB Against
    se cancela y el protocolo vuelve a IDLE desde la barra actual.
    El reset también aplica en CUT_ACTIVE.

─── Estados del protocolo ───────────────────────────────────────────────────
IDLE          Sin MSB Against activo. Monitoreando estructura relevante.
MSB_PENDING   MSB Against confirmado. Rastreando HH/LL barra a barra.
              Esperando pullback o reset.
CUT_ACTIVE    Pullback confirmado. Cut Early Level activo.
              Esperando trigger o reset.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from market_structure_bos import detect_market_structure


# ══════════════════════════════════════════════════════════════════════════════
# Estructura relevante — pre-computada con detect_market_structure
# ══════════════════════════════════════════════════════════════════════════════

def _build_relevant_levels(
    high:      np.ndarray,
    low:       np.ndarray,
    open_:     np.ndarray,
    close:     np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    Pre-computa el nivel relevante vigente en cada barra usando
    detect_market_structure() — misma lógica que market_structure_bos.py.

    Para direction='bear' → rastrea LH relevantes.
    Para direction='bull' → rastrea HL relevantes.

    Orden de operaciones por barra i (crítico):
        1. Invalidar nivel vigente si High[i] > cur_level (bear) o
           Low[i] < cur_level (bull). Esto ocurre ANTES de incorporar
           nuevos pivots, porque un pivot confirmado en la barra i no
           puede ser invalidado por esa misma barra — detect_market_structure
           ya garantiza su validez mediante el orden intravela.
        2. Incorporar pivots confirmados en la barra i. Si el pivot tiene
           un precio mayor/menor que el actual, pasa a ser el nivel vigente.

    Retorna
    -------
    np.ndarray float (n,) — nivel relevante vigente en cada barra.
    np.nan si no hay nivel confirmado o el último fue invalidado.
    """
    n      = len(high)
    points = detect_market_structure(high, low, open_, close)

    kind      = 'LH' if direction == 'bear' else 'HL'
    pivots    = [(p.price, p.bar) for p in points if p.kind == kind]
    relevant  = np.full(n, np.nan)
    pivot_idx = 0
    cur_level = np.nan

    for i in range(n):

        # 1 — Invalidar antes de incorporar
        if not np.isnan(cur_level):
            if direction == 'bear' and high[i] > cur_level:
                cur_level = np.nan
            elif direction == 'bull' and low[i] < cur_level:
                cur_level = np.nan

        # 2 — Incorporar pivots confirmados en la barra i
        while pivot_idx < len(pivots) and pivots[pivot_idx][1] <= i:
            cur_level = pivots[pivot_idx][0]
            pivot_idx += 1

        relevant[i] = cur_level

    return relevant


# ══════════════════════════════════════════════════════════════════════════════
# Función principal
# ══════════════════════════════════════════════════════════════════════════════

def compute_cut_early(
    data:   pd.DataFrame,
    params: dict,
) -> pd.Series:
    """
    Detecta el Cut Early Protocol y devuelve una señal de salida anticipada.

    Parámetros
    ----------
    data : pd.DataFrame
        Columnas requeridas: Open, High, Low, Close (case-insensitive).

    params : dict
        Claves requeridas:
            'direction'    : 'bear' | 'bull'  — dirección del trade abierto.
            'entry_bar'    : int              — índice entero de la barra de entrada.
            'entry_price'  : float            — precio de entrada del trade.
            'sl_price'     : float            — SL original del trade.
                                                Bear → AA (por encima del entry).
                                                Bull → BB (por debajo del entry).

    Retorna
    -------
    pd.Series (float), mismo índice que data.
        1.0 en la barra donde se activa el Cut Early.
        0.0 en el resto.
    """
    # ── Validación ────────────────────────────────────────────────────────
    required = {'direction', 'entry_bar', 'entry_price', 'sl_price'}
    if missing := required - set(params):
        raise ValueError(f"Faltan parámetros: {missing}")

    direction   = params['direction']
    entry_bar   = int(params['entry_bar'])
    entry_price = float(params['entry_price'])
    sl_price    = float(params['sl_price'])

    if direction not in ('bear', 'bull'):
        raise ValueError("direction debe ser 'bear' o 'bull'")
    if direction == 'bear' and sl_price <= entry_price:
        raise ValueError("Bear: sl_price debe ser > entry_price (AA)")
    if direction == 'bull' and sl_price >= entry_price:
        raise ValueError("Bull: sl_price debe ser < entry_price (BB)")

    # ── Arrays OHLC ───────────────────────────────────────────────────────
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if miss := {'Open', 'High', 'Low', 'Close'} - set(df.columns):
        raise ValueError(f"Faltan columnas: {miss}")

    n      = len(df)
    open_  = df['Open'].values
    high   = df['High'].values
    low    = df['Low'].values
    close  = df['Close'].values
    output = np.zeros(n, dtype=float)

    if entry_bar >= n:
        return pd.Series(output, index=data.index)

    # ── Niveles relevantes pre-computados ─────────────────────────────────
    relevant = _build_relevant_levels(high, low, open_, close, direction)

    # ── Variables de estado ───────────────────────────────────────────────
    state           = 'IDLE'       # IDLE | MSB_PENDING | CUT_ACTIVE
    restart_level   = np.nan       # mín/máx de la zona verde al detectar MSB Against
    pending_extreme = np.nan       # HH/LL acumulado en MSB_PENDING
    cut_early_level = np.nan       # nivel de salida activo en CUT_ACTIVE

    # Extremo a favor acumulado en IDLE (para calcular restart_level)
    running_extreme = low[entry_bar] if direction == 'bear' else high[entry_bar]

    # ── Simulación barra a barra ──────────────────────────────────────────
    for i in range(entry_bar, n):

        rel_level = relevant[i]

        # Actualizar extremo a favor solo mientras no hay MSB activo
        if state == 'IDLE':
            if direction == 'bear':
                running_extreme = min(running_extreme, low[i])
            else:
                running_extreme = max(running_extreme, high[i])

        # ── IDLE — buscar MSB Against ──────────────────────────────────────
        if state == 'IDLE':

            if np.isnan(rel_level):
                continue

            if direction == 'bear':
                # Break and close alcista sobre LH relevante en zona SL
                if close[i] > rel_level and rel_level > entry_price:
                    restart_level   = running_extreme   # mín zona verde
                    pending_extreme = high[i]           # HH inicial
                    state           = 'MSB_PENDING'

            else:  # bull
                # Break and close bajista bajo HL relevante en zona SL
                if close[i] < rel_level and rel_level < entry_price:
                    restart_level   = running_extreme   # máx zona verde
                    pending_extreme = low[i]            # LL inicial
                    state           = 'MSB_PENDING'

        # ── MSB_PENDING — acumular HH/LL, esperar pullback o reset ─────────
        elif state == 'MSB_PENDING':

            if direction == 'bear':
                pending_extreme = max(pending_extreme, high[i])  # acumular HH

                if low[i] <= restart_level:              # reset
                    state           = 'IDLE'
                    pending_extreme = np.nan
                    restart_level   = np.nan
                    running_extreme = low[i]
                    continue

                if close[i] < open_[i]:                  # pullback confirmado
                    cut_early_level = pending_extreme
                    state           = 'CUT_ACTIVE'

            else:  # bull
                pending_extreme = min(pending_extreme, low[i])   # acumular LL

                if high[i] >= restart_level:             # reset
                    state           = 'IDLE'
                    pending_extreme = np.nan
                    restart_level   = np.nan
                    running_extreme = high[i]
                    continue

                if close[i] > open_[i]:                  # pullback confirmado
                    cut_early_level = pending_extreme
                    state           = 'CUT_ACTIVE'

        # ── CUT_ACTIVE — esperar trigger o reset ───────────────────────────
        elif state == 'CUT_ACTIVE':

            if direction == 'bear':
                if low[i] <= restart_level:              # reset
                    state           = 'IDLE'
                    cut_early_level = np.nan
                    pending_extreme = np.nan
                    restart_level   = np.nan
                    running_extreme = low[i]
                    continue

                if high[i] >= cut_early_level:           # trigger
                    output[i] = 1.0
                    break

            else:  # bull
                if high[i] >= restart_level:             # reset
                    state           = 'IDLE'
                    cut_early_level = np.nan
                    pending_extreme = np.nan
                    restart_level   = np.nan
                    running_extreme = high[i]
                    continue

                if low[i] <= cut_early_level:            # trigger
                    output[i] = 1.0
                    break

    return pd.Series(output, index=data.index)


# ══════════════════════════════════════════════════════════════════════════════
# Test rápido
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(7)
    n     = 500
    price = 1.3500 + np.cumsum(np.random.randn(n) * 0.0003)
    opens = np.roll(price, 1); opens[0] = price[0]

    df_test = pd.DataFrame({
        'Open':  opens,
        'High':  price + np.abs(np.random.randn(n) * 0.0004),
        'Low':   price - np.abs(np.random.randn(n) * 0.0004),
        'Close': price,
    })

    for direction, sl_offset in [('bear', +0.0050), ('bull', -0.0050)]:
        entry_bar   = 50
        entry_price = df_test['Close'].iloc[entry_bar]
        sl_price    = entry_price + sl_offset

        signal = compute_cut_early(df_test, {
            'direction':   direction,
            'entry_bar':   entry_bar,
            'entry_price': entry_price,
            'sl_price':    sl_price,
        })

        print("=" * 50)
        print(f"CUT EARLY — {direction.upper()}")
        print("=" * 50)
        print(f"  Entry bar    : {entry_bar}")
        print(f"  Entry price  : {entry_price:.5f}")
        print(f"  SL price     : {sl_price:.5f}")
        print(f"  Cut Early    : {signal.sum():.0f} señal(es)")
        if signal.sum() > 0:
            bar = signal[signal == 1.0].index[0]
            print(f"  Activado en  : barra {df_test.index.get_loc(bar)}")
        print("=" * 50)
