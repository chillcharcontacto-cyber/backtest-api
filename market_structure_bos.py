"""
Market Structure & BOS (Break of Structure) — MCT Protocol
===========================================================
Timeframe : M5
Pairs     : EURUSD, GBPUSD
Requiere  : liquidity_channels.py en el mismo directorio.

═══════════════════════════════════════════════════════════════════════════════
GLOSARIO COMPLETO
═══════════════════════════════════════════════════════════════════════════════

Términos heredados de liquidity_channels.py
--------------------------------------------
SS   (Sweep Start)    Barra donde se forma el pivot high o pivot low.
                      Su precio es el Sweep Level.

Sweep Level           Precio exacto del pivot high/low en SS.
                      Es la línea horizontal que el precio debe cruzar.

SE   (Sweep End)      Barra donde el precio cruza el Sweep Level.
                      · Setup de compra : low  < Sweep Level  (pivot low cruzado)
                      · Setup de venta  : high > Sweep Level  (pivot high cruzado)

OL   (Opposite Level) Extremo del precio en el rango SS→SE.
                      · Compras : max(High[SS:SE]) — el TECHO del sweep, objetivo alcista.
                      · Ventas  : min(Low[SS:SE])  — el SUELO del sweep, objetivo bajista.

Sweep Range           Espacio de PRECIO entre Sweep Level y OL.
                      · Compras : entre sweep_level (suelo) y OL (techo).
                      · Ventas  : entre OL (suelo) y sweep_level (techo).
                      IMPORTANTE: es un rango de precio, no de barras.

Términos de este módulo
-----------------------
LL   (Lower Low)      Low más bajo entre dos LH consecutivos.
                      Sin condición de color de vela — solo importa el precio.

LH   (Lower High)     High más alto entre dos LL consecutivos.
                      CONDICIÓN OBLIGATORIA: en el retroceso entre los dos LL
                      debe haber al menos UNA vela alcista (close > open).
                      La vela alcista no tiene que ser la que marca el LH —
                      solo tiene que existir en ese tramo.
                      EMPATE de highs: se escoge el más reciente (derecha).

HH   (Higher High)    High más alto entre dos HL consecutivos.
                      Sin condición de color de vela.

HL   (Higher Low)     Low más bajo entre dos HH consecutivos.
                      CONDICIÓN OBLIGATORIA: en el retroceso entre los dos HH
                      debe haber al menos UNA vela bajista (close < open).
                      EMPATE de lows: se escoge el más reciente (derecha).

Validación circular   Un LH solo es LH si va seguido de un LL.
                      Un LL solo es LL si va seguido de un LH.
                      Ninguno se confirma hasta que el siguiente punto aparece.
                      Ídem para HH/HL.

Orden intravela       Convención para resolver ambigüedades dentro de una vela:
                      · Vela ALCISTA (close > open): primero ocurre el LOW, luego el HIGH.
                      · Vela BAJISTA (close < open): primero ocurre el HIGH, luego el LOW.
                      Esto permite que una sola vela sea simultáneamente LL y LH
                      (su low marca el LL y su high marca el LH del retroceso).

BB   (Below Below)    min(Low[SE : siguiente_SE]) — el mínimo post-SE en setup de compra.
                      Actúa como SL en el trade de compra.

AA   (Above Above)    max(High[SE : siguiente_SE]) — el máximo post-SE en setup de venta.
                      Actúa como SL en el trade de venta.

MSB Level             Nivel clave para el BOS. Se calcula así:

                      Para COMPRAS (sweep bullish):
                        Buscar en orden cronológico el primer LH cuyo High exacto
                        esté dentro del sweep range (sweep_level ≤ price ≤ OL).
                        Además, desde ese LH hasta el final del rango debe existir
                        un LL confirmado en el gráfico de LÍNEAS (Close), con al
                        menos una vela alcista en ese tramo.
                        MSB Level = High exacto del LH.

                      Para VENTAS (sweep bearish):
                        Buscar el primer HL cuyo Low exacto esté dentro del sweep
                        range (OL ≤ price ≤ sweep_level).
                        Además, desde ese HL debe existir un HH en el Close, con
                        al menos una vela bajista en ese tramo.
                        MSB Level = Low exacto del HL.

                      Filtro Close: evita setups donde las velas japonesas muestran
                      estructura pero el gráfico de líneas (Close) no la confirma.
                      Si el OHLC marca un LH pero el Close no tiene un LL posterior,
                      ese LH se descarta y se busca el siguiente candidato.

                      La vela candidata puede abrirse o cerrarse fuera del sweep
                      range — lo que importa es que el precio exacto del LH/HL
                      (High del LH o Low del HL) esté dentro del rango.

BOS  (Break of        Rotura de estructura. Ocurre cuando:
     Structure)
                      COMPRAS: una vela M5 CIERRA por encima del MSB Level.
                               → El cierre debe superar el nivel, no basta con tocarlo.

                      VENTAS: una vela M5 CIERRA por debajo del MSB Level.

                      El BOS confirma que el precio ha roto el nivel estructural
                      y activa la espera del retest.

Retest                Después del BOS, el precio vuelve a tocar el MSB Level:

                      COMPRAS: el LOW de cualquier vela posterior toca o baja del
                               MSB Level → ENTRADA LONG.

                      VENTAS: el HIGH de cualquier vela posterior toca o sube al
                              MSB Level → ENTRADA SHORT.

                      El retest se ejecuta con la vela ABIERTA (no se espera cierre).
                      No importa el color de la vela que hace el retest.
                      El retest NUNCA ocurre en la misma barra que el BOS.

SL   (Stop Loss)      COMPRAS: BB (min Low post-SE). Siempre por debajo de la entrada.
                      VENTAS:  AA (max High post-SE). Siempre por encima de la entrada.
                      Si el precio toca el SL después del retest = trade cerrado con pérdida.

TP   (Take Profit)    COMPRAS: OL (max High en SS→SE, el techo del sweep).
                      VENTAS:  OL (min Low en SS→SE, el suelo del sweep).

Invalidación          El setup se cancela (sin ejecutar entrada) si:
                      · El precio toca el OL antes del retest
                        (COMPRAS: high ≥ OL. VENTAS: low ≤ OL).
                        Esto significa que el precio llegó al objetivo sin pasar
                        por el nivel de entrada.
                      · No hay un cierre por encima/debajo del MSB Level y el
                        precio ya alcanzó el OL.

Prioridad global      Solo puede haber UN setup activo en cualquier momento,
                      independientemente de la dirección.
                      Si hay un setup activo (pendiente de retest) y aparece un
                      nuevo sweep (en cualquier dirección), ese nuevo sweep se
                      IGNORA completamente.
                      El setup activo solo se cierra por:
                        a) Retest ejecutado (entrada al trade).
                        b) Precio toca OL sin retest (invalidado).

═══════════════════════════════════════════════════════════════════════════════
FLUJO COMPLETO DEL SETUP — COMPRAS (MCT BULLISH)
═══════════════════════════════════════════════════════════════════════════════

  1. Se detecta un sweep bullish: low < sweep_level → SE confirmado.
     OL = max(High[SS:SE]).

  2. Dentro del sweep range [sweep_level, OL], buscar el primer LH que:
     a) Tenga su High exactamente dentro del rango de precio.
     b) Tenga un LL en el Close posterior (con ≥1 vela alcista) hasta BB.
     → Ese LH es el MSB Level.

  3. Esperar que una vela cierre por ENCIMA del MSB Level → BOS confirmado.

  4. En cualquier vela posterior, si el LOW toca el MSB Level → ENTRADA LONG.
     · SL = BB (mínimo post-SE).
     · TP = OL (máximo de SS→SE).

  5. Invalidación en cualquier momento: si HIGH ≥ OL antes del retest → cancelado.

Para VENTAS: todo simétrico con HH/HL, AA, y condiciones invertidas.

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from liquidity_channels import detect_liquidity_levels, detect_sweeps, Sweep


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class StructurePoint:
    """
    Punto de estructura de mercado confirmado.

    Atributos
    ---------
    kind  : 'LL' | 'LH' | 'HH' | 'HL'
    price : High exacto para LH/HH | Low exacto para LL/HL
    bar   : índice entero de la barra donde se produce el extremo de precio
    """
    kind:  str
    price: float
    bar:   int


@dataclass
class MSBSetup:
    """
    Setup MCT activo, pendiente de retest o invalidación.

    Atributos
    ---------
    direction : 'bullish' (compra) | 'bearish' (venta)
    msb_level : precio del MSB Level (High del LH en compras, Low del HL en ventas)
    msb_bar   : barra donde se formó el MSB Level
    bos_bar   : barra donde se confirmó el BOS (-1 si aún no hay BOS)
    sl_price  : Stop Loss — BB para compras, AA para ventas
    tp_price  : Take Profit — OL del sweep original
    ol_price  : precio del OL (mismo que tp_price, se guarda para la invalidación)
    """
    direction: str
    msb_level: float
    msb_bar:   int
    bos_bar:   int
    sl_price:  float
    tp_price:  float
    ol_price:  float


# ─── Orden intravela ──────────────────────────────────────────────────────────

def _events(i: int, open_: np.ndarray, high: np.ndarray,
            low: np.ndarray, close: np.ndarray) -> list:
    """
    Retorna los eventos de precio de la barra i en orden intravela.

    Convención
    ----------
    Vela ALCISTA (close >= open): primero LOW, luego HIGH.
    Vela BAJISTA (close <  open): primero HIGH, luego LOW.

    Esto resuelve la ambigüedad de qué extremo ocurrió primero dentro
    de la vela, y permite que una sola vela sea simultáneamente LL y LH.

    Retorna
    -------
    Lista de tuplas: [('low' | 'high', precio), ...]
    """
    if close[i] >= open_[i]:   # alcista o doji
        return [('low', low[i]), ('high', high[i])]
    else:                       # bajista
        return [('high', high[i]), ('low', low[i])]


# ─── Estructura de mercado ────────────────────────────────────────────────────

def detect_market_structure(
    high:  np.ndarray,
    low:   np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
) -> list[StructurePoint]:
    """
    Detecta todos los puntos de estructura (LL, LH, HH, HL) en el DataFrame,
    con validación circular completa y orden intravela.

    Algoritmo — Máquina de estados dual
    ------------------------------------
    Se ejecutan en PARALELO dos máquinas de estados independientes:
    una para la estructura bajista (LL/LH) y otra para la alcista (HH/HL).

    ESTRUCTURA BAJISTA
    ------------------
    Estado 'seek_ll':
        Busca un nuevo low que sea menor que el último LL confirmado.
        En cuanto aparece un evento 'low' más bajo → nuevo candidato LL.
        Transición inmediata a 'seek_lh'.

    Estado 'seek_lh':
        Busca el high más alto del retroceso desde el candidato LL.
        · En cada evento 'high': actualiza el candidato LH (empate → más reciente).
          Si la vela es alcista, activa el flag b_has_bull.
        · En cada evento 'low':
          - Si el nuevo low es menor que el candidato LL Y hay vela alcista
            en el retroceso (b_has_bull=True) Y hay un LH candidato válido:
            → CONFIRMAR par (LL, LH). Añadir ambos a points[].
            → El nuevo low se convierte en el siguiente candidato LL.
            → Permanecer en 'seek_lh' para el siguiente retroceso.
          - Si el nuevo low es menor pero no hay vela alcista:
            → Actualizar candidato LL al nuevo mínimo.
            → Resetear candidato LH (el retroceso anterior no era válido).

    ESTRUCTURA ALCISTA — simétrica
    --------------------------------
    Estado 'seek_hh': busca high > último HH confirmado.
    Estado 'seek_hl': busca el low más bajo del retroceso con ≥1 vela bajista.

    Parámetros
    ----------
    high, low, open_, close : arrays numpy de OHLC

    Retorna
    -------
    Lista de StructurePoint ordenada cronológicamente por barra.
    Los puntos confirmados siempre vienen en pares: (LL, LH) o (HH, HL).
    """
    n      = len(high)
    points = []

    # ── Estado bajista ────────────────────────────────────────────────────────
    b_phase       = 'seek_ll'
    b_last_ll     = np.inf     # precio del último LL confirmado (referencia)
    b_cand_ll     = np.inf     # precio del candidato LL actual
    b_cand_ll_bar = -1         # barra del candidato LL
    b_cand_lh     = -np.inf    # precio del candidato LH actual (máximo del retroceso)
    b_cand_lh_bar = -1         # barra del candidato LH (más reciente en empate)
    b_has_bull    = False      # flag: ≥1 vela alcista en el retroceso actual

    # ── Estado alcista ────────────────────────────────────────────────────────
    u_phase       = 'seek_hh'
    u_last_hh     = -np.inf
    u_cand_hh     = -np.inf
    u_cand_hh_bar = -1
    u_cand_hl     = np.inf
    u_cand_hl_bar = -1
    u_has_bear    = False

    for i in range(n):
        is_bull = close[i] >= open_[i]   # True si la vela es alcista o doji
        is_bear = close[i] <  open_[i]   # True si la vela es bajista

        for evt_type, evt_price in _events(i, open_, high, low, close):

            # ════════════════════════════════════════════════════════════════
            # ESTRUCTURA BAJISTA
            # ════════════════════════════════════════════════════════════════

            if b_phase == 'seek_ll':
                if evt_type == 'low' and evt_price < b_last_ll:
                    # Nuevo candidato LL encontrado → empezar a buscar LH
                    b_cand_ll     = evt_price
                    b_cand_ll_bar = i
                    b_phase       = 'seek_lh'
                    b_has_bull    = False
                    b_cand_lh     = -np.inf
                    b_cand_lh_bar = -1

            elif b_phase == 'seek_lh':
                if evt_type == 'high':
                    # Actualizar candidato LH con el máximo del retroceso
                    if is_bull:
                        b_has_bull = True   # registrar vela alcista en el retroceso
                    # Empate de highs → más reciente gana (>=)
                    if evt_price >= b_cand_lh:
                        b_cand_lh     = evt_price
                        b_cand_lh_bar = i

                elif evt_type == 'low':
                    if (evt_price < b_cand_ll        # nuevo mínimo = nuevo LL
                            and b_has_bull           # hubo ≥1 vela alcista en el retroceso
                            and b_cand_lh_bar >= 0): # hay un LH candidato válido
                        # ── CONFIRMAR par (LL anterior + LH del retroceso) ──
                        points.append(StructurePoint('LL', b_cand_ll, b_cand_ll_bar))
                        points.append(StructurePoint('LH', b_cand_lh, b_cand_lh_bar))
                        b_last_ll     = b_cand_ll   # actualizar referencia LL
                        # El nuevo low es el siguiente candidato LL
                        b_cand_ll     = evt_price
                        b_cand_ll_bar = i
                        b_cand_lh     = -np.inf
                        b_cand_lh_bar = -1
                        b_has_bull    = False
                        # Permanecer en 'seek_lh' para el siguiente retroceso

                    elif evt_price < b_cand_ll:
                        # LL más bajo pero sin vela alcista en el retroceso:
                        # el retroceso anterior no era válido → resetear todo
                        b_cand_ll     = evt_price
                        b_cand_ll_bar = i
                        b_cand_lh     = -np.inf
                        b_cand_lh_bar = -1
                        b_has_bull    = False

            # ════════════════════════════════════════════════════════════════
            # ESTRUCTURA ALCISTA (simétrica a la bajista)
            # ════════════════════════════════════════════════════════════════

            if u_phase == 'seek_hh':
                if evt_type == 'high' and evt_price > u_last_hh:
                    u_cand_hh     = evt_price
                    u_cand_hh_bar = i
                    u_phase       = 'seek_hl'
                    u_has_bear    = False
                    u_cand_hl     = np.inf
                    u_cand_hl_bar = -1

            elif u_phase == 'seek_hl':
                if evt_type == 'low':
                    if is_bear:
                        u_has_bear = True
                    # Empate de lows → más reciente gana (<=)
                    if evt_price <= u_cand_hl:
                        u_cand_hl     = evt_price
                        u_cand_hl_bar = i

                elif evt_type == 'high':
                    if (evt_price > u_cand_hh
                            and u_has_bear
                            and u_cand_hl_bar >= 0):
                        # ── CONFIRMAR par (HH anterior + HL del retroceso) ──
                        points.append(StructurePoint('HH', u_cand_hh, u_cand_hh_bar))
                        points.append(StructurePoint('HL', u_cand_hl, u_cand_hl_bar))
                        u_last_hh     = u_cand_hh
                        u_cand_hh     = evt_price
                        u_cand_hh_bar = i
                        u_cand_hl     = np.inf
                        u_cand_hl_bar = -1
                        u_has_bear    = False

                    elif evt_price > u_cand_hh:
                        # HH más alto pero sin vela bajista → resetear retroceso
                        u_cand_hh     = evt_price
                        u_cand_hh_bar = i
                        u_cand_hl     = np.inf
                        u_cand_hl_bar = -1
                        u_has_bear    = False

    points.sort(key=lambda p: p.bar)
    return points


# ─── Filtro Close (gráfico de líneas) ────────────────────────────────────────

def _has_close_ll(
    close:    np.ndarray,
    open_:    np.ndarray,
    from_bar: int,
    to_bar:   int,
) -> bool:
    """
    Verifica si existe un LL en el gráfico de LÍNEAS (Close) en el rango dado.

    Un LL en Close existe si hay un par de barras (a, b) con:
      from_bar ≤ a < b ≤ to_bar
      close[b] < close[a]                   → lower close (LL en líneas)
      ≥1 vela alcista (close≥open) en [a,b]  → la misma condición que en OHLC

    Por qué existe este filtro
    --------------------------
    Las velas japonesas pueden crear ilusiones de estructura que el gráfico
    de líneas no confirma. Por ejemplo: una vela alcista con una mecha inferior
    muy larga puede marcar un LH en OHLC, pero si el Close de esa vela está
    en la parte alta, el gráfico de líneas no muestra ningún retroceso.
    Este filtro garantiza que el MSB Level tiene respaldo tanto en OHLC
    como en el gráfico de líneas.

    Parámetros
    ----------
    close, open_ : arrays completos (se indexan con from_bar:to_bar+1)
    from_bar     : primera barra del rango (inclusivo)
    to_bar       : última barra del rango (inclusivo)

    Retorna
    -------
    True si existe al menos un par (a, b) que cumpla las condiciones.
    """
    seg_c = close[from_bar: to_bar + 1]
    seg_o = open_[from_bar: to_bar + 1]
    m     = len(seg_c)
    if m < 2:
        return False
    for a in range(m - 1):
        for b in range(a + 1, m):
            if seg_c[b] < seg_c[a]:                    # lower close
                for k in range(a, b + 1):
                    if seg_c[k] >= seg_o[k]:           # vela alcista en [a,b]
                        return True
    return False


def _has_close_hh(
    close:    np.ndarray,
    open_:    np.ndarray,
    from_bar: int,
    to_bar:   int,
) -> bool:
    """
    Versión simétrica de _has_close_ll para ventas.

    Un HH en Close existe si hay (a, b) con:
      close[b] > close[a]                    → higher close (HH en líneas)
      ≥1 vela bajista (close<open) en [a,b]
    """
    seg_c = close[from_bar: to_bar + 1]
    seg_o = open_[from_bar: to_bar + 1]
    m     = len(seg_c)
    if m < 2:
        return False
    for a in range(m - 1):
        for b in range(a + 1, m):
            if seg_c[b] > seg_c[a]:                    # higher close
                for k in range(a, b + 1):
                    if seg_c[k] < seg_o[k]:            # vela bajista en [a,b]
                        return True
    return False


# ─── Detección del MSB Level ──────────────────────────────────────────────────

def _find_msb_level(
    sweep:     Sweep,
    high:      np.ndarray,
    low:       np.ndarray,
    open_:     np.ndarray,
    close:     np.ndarray,
    structure: list[StructurePoint],
    end_bar:   int,
) -> Optional[tuple[float, int]]:
    """
    Busca el MSB Level para un sweep dado.

    Proceso de búsqueda
    -------------------
    1. Definir el sweep range en PRECIO:
       price_lo = min(sweep_level, ol_price)
       price_hi = max(sweep_level, ol_price)

    2. Filtrar los puntos de estructura del tipo correcto:
       · Compras (bullish): buscar LH con High dentro del sweep range.
       · Ventas  (bearish): buscar HL con Low  dentro del sweep range.
       Solo se consideran puntos con barra ≤ end_bar.

    3. Ordenar candidatos cronológicamente (el primero es el más antiguo).

    4. Para cada candidato, aplicar el filtro Close:
       · Compras: _has_close_ll(close, open_, cand.bar, end_bar)
       · Ventas:  _has_close_hh(close, open_, cand.bar, end_bar)
       El primer candidato que pase el filtro es el MSB Level.

    Nota sobre end_bar
    ------------------
    end_bar = min(siguiente_SE - 1, n - 1).
    Define hasta dónde buscamos el filtro de Close. El LL/HH en Close
    puede ocurrir hasta el final del rango post-SE.

    Parámetros
    ----------
    sweep     : objeto Sweep del módulo liquidity_channels
    structure : lista de StructurePoint de detect_market_structure()
    end_bar   : última barra del rango de búsqueda (inclusivo)

    Retorna
    -------
    (price, bar) del MSB Level, o None si no se encuentra ninguno válido.
    """
    lo = min(sweep.sweep_level, sweep.ol_price)   # límite inferior del sweep range
    hi = max(sweep.sweep_level, sweep.ol_price)   # límite superior del sweep range

    if sweep.direction == 'bullish':
        kind   = 'LH'
        flt_fn = _has_close_ll
    else:
        kind   = 'HL'
        flt_fn = _has_close_hh

    # Candidatos: puntos del tipo correcto dentro del sweep range en precio
    candidates = sorted(
        [p for p in structure
         if p.kind == kind
         and lo <= p.price <= hi
         and p.bar <= end_bar],
        key=lambda p: p.bar   # ordenar por barra para tomar el primero válido
    )

    for cand in candidates:
        # Aplicar filtro Close: debe haber un LL/HH en Close desde este punto
        if flt_fn(close, open_, cand.bar, end_bar):
            return (cand.price, cand.bar)

    return None   # ningún candidato pasó el filtro


# ─── AA y BB ──────────────────────────────────────────────────────────────────

def _aa_bb(
    high:    np.ndarray,
    low:     np.ndarray,
    se_bar:  int,
    next_se: int,
) -> tuple[float, int, float, int]:
    """
    Calcula AA y BB para el rango post-SE.

    AA (Above Above) = max(High[se_bar : next_se]) → SL para ventas.
    BB (Below Below) = min(Low[se_bar  : next_se]) → SL para compras.

    Parámetros
    ----------
    se_bar  : barra del SE (inclusiva)
    next_se : barra del siguiente SE (exclusiva) o n si es el último sweep

    Retorna
    -------
    (aa_price, aa_bar, bb_price, bb_bar)
    """
    h = high[se_bar: next_se]
    l = low[se_bar: next_se]
    if len(h) == 0:
        return np.nan, se_bar, np.nan, se_bar
    aa_off = int(np.argmax(h))
    bb_off = int(np.argmin(l))
    return (
        h[aa_off], se_bar + aa_off,
        l[bb_off], se_bar + bb_off,
    )


# ─── Detector principal ───────────────────────────────────────────────────────

def detect_bos(
    data:         pd.DataFrame,
    liq_strength: int = 25,
) -> pd.DataFrame:
    """
    Detecta todos los setups MCT (BOS + retest) en el DataFrame.

    Flujo interno
    -------------
    1. detect_liquidity_levels() → detecta pivot highs/lows activos.
    2. detect_sweeps()           → construye los objetos Sweep (SS, SE, OL).
    3. detect_market_structure() → detecta LL, LH, HH, HL globalmente.
    4. Por cada sweep, pre-calcular:
       · AA y BB del rango post-SE.
       · MSB Level (primer LH/HL válido en precio + filtro Close).
    5. Simulación barra a barra:
       · Al llegar al SE de un sweep: crear MSBSetup si hay MSB Level y no
         hay setup activo. Si ya hay uno activo, ignorar el sweep.
       · Con setup activo:
         - Fase 1 (sin BOS): esperar cierre por encima/debajo del MSB Level.
           Si el precio toca el OL antes → invalidar.
         - Fase 2 (BOS confirmado): esperar retest en barra posterior.
           Retest = low ≤ MSB Level (compras) / high ≥ MSB Level (ventas).
           Si el precio toca el OL antes del retest → invalidar.

    Parámetros
    ----------
    data         : pd.DataFrame con columnas Open, High, Low, Close
                   (case-insensitive).
    liq_strength : fuerza del pivot para liquidity channels. Default = 25.

    Retorna
    -------
    DataFrame original con columnas adicionales:

    bos_bull        bool   True en la barra donde hay cierre > MSB Level (compras)
    bos_bear        bool   True en la barra donde hay cierre < MSB Level (ventas)
    entry_bull      bool   True en la barra del retest bullish (entrada long)
    entry_bear      bool   True en la barra del retest bearish (entrada short)
    msb_level_bull  float  Precio del MSB Level activo para compras (NaN si no hay)
    msb_level_bear  float  Precio del MSB Level activo para ventas  (NaN si no hay)
    sl_bull         float  SL activo para compras = BB (NaN si no hay setup)
    sl_bear         float  SL activo para ventas  = AA (NaN si no hay setup)
    tp_bull         float  TP activo para compras = OL (NaN si no hay setup)
    tp_bear         float  TP activo para ventas  = OL (NaN si no hay setup)
    """
    df = data.copy()
    df.columns = [c.capitalize() for c in df.columns]

    required = {'Open', 'High', 'Low', 'Close'}
    if missing := required - set(df.columns):
        raise ValueError(f"Faltan columnas: {missing}")

    n     = len(df)
    high  = df['High'].values
    low   = df['Low'].values
    open_ = df['Open'].values
    close = df['Close'].values

    # ── 1. Sweeps ─────────────────────────────────────────────────────────────
    liq    = detect_liquidity_levels(df, strength=liq_strength)
    sweeps = detect_sweeps(liq)
    s_srt  = sorted(sweeps, key=lambda s: s.se_bar)

    # Mapa se_bar → barra del siguiente SE (para delimitar el rango post-SE)
    nxt_map = {
        sw.se_bar: (s_srt[i + 1].se_bar if i + 1 < len(s_srt) else n)
        for i, sw in enumerate(s_srt)
    }

    # ── 2. Estructura global ──────────────────────────────────────────────────
    structure = detect_market_structure(high, low, open_, close)

    # ── 3. Pre-calcular datos por sweep ──────────────────────────────────────
    sw_info: dict[int, dict] = {}
    for sw in s_srt:
        se  = sw.se_bar
        nxt = nxt_map[se]
        end = min(nxt - 1, n - 1)

        aa_p, _, bb_p, _ = _aa_bb(high, low, se, nxt)
        msb = _find_msb_level(sw, high, low, open_, close, structure, end)

        sw_info[se] = {
            'sw':    sw,
            'aa':    aa_p,   # SL para ventas
            'bb':    bb_p,   # SL para compras
            'msb_p': msb[0] if msb else np.nan,
            'msb_b': msb[1] if msb else -1,
        }

    # ── 4. Arrays de salida ───────────────────────────────────────────────────
    bos_bull   = np.zeros(n, dtype=bool)
    bos_bear   = np.zeros(n, dtype=bool)
    entry_bull = np.zeros(n, dtype=bool)
    entry_bear = np.zeros(n, dtype=bool)
    msb_bull   = np.full(n, np.nan)
    msb_bear   = np.full(n, np.nan)
    sl_bull    = np.full(n, np.nan)
    sl_bear    = np.full(n, np.nan)
    tp_bull    = np.full(n, np.nan)
    tp_bear    = np.full(n, np.nan)

    # ── 5. Simulación barra a barra ───────────────────────────────────────────
    active: Optional[MSBSetup] = None   # setup activo (solo uno globalmente)
    bos_ok  = False                      # True una vez confirmado el BOS

    for i in range(n):

        # ── Nuevo sweep en esta barra ─────────────────────────────────────────
        # Solo se activa si no hay setup activo (prioridad global).
        # Si hay un setup pendiente, el nuevo sweep se ignora completamente.
        if i in sw_info and active is None:
            sd = sw_info[i]
            if not np.isnan(sd['msb_p']):
                sw = sd['sw']
                if sw.direction == 'bullish':
                    active = MSBSetup(
                        direction = 'bullish',
                        msb_level = sd['msb_p'],
                        msb_bar   = sd['msb_b'],
                        bos_bar   = -1,
                        sl_price  = sd['bb'],       # SL = BB (mínimo post-SE)
                        tp_price  = sw.ol_price,    # TP = OL (máximo de SS→SE)
                        ol_price  = sw.ol_price,
                    )
                else:  # bearish
                    active = MSBSetup(
                        direction = 'bearish',
                        msb_level = sd['msb_p'],
                        msb_bar   = sd['msb_b'],
                        bos_bar   = -1,
                        sl_price  = sd['aa'],       # SL = AA (máximo post-SE)
                        tp_price  = sw.ol_price,    # TP = OL (mínimo de SS→SE)
                        ol_price  = sw.ol_price,
                    )
                bos_ok = False

        # Sin setup activo → nada que gestionar
        if active is None:
            continue

        s = active

        # Publicar niveles activos en esta barra
        if s.direction == 'bullish':
            msb_bull[i] = s.msb_level
            sl_bull[i]  = s.sl_price
            tp_bull[i]  = s.tp_price
        else:
            msb_bear[i] = s.msb_level
            sl_bear[i]  = s.sl_price
            tp_bear[i]  = s.tp_price

        # ── FASE 1: Esperando BOS ─────────────────────────────────────────────
        if not bos_ok:
            if s.direction == 'bullish':
                if close[i] > s.msb_level:
                    # BOS confirmado: vela cierra POR ENCIMA del MSB Level
                    bos_bull[i] = True
                    s.bos_bar   = i
                    bos_ok      = True
                elif high[i] >= s.ol_price:
                    # INVALIDACIÓN: precio alcanza el OL (techo) sin BOS previo.
                    # OL en compras es el máximo → high >= OL lo toca.
                    active = None

            else:  # bearish
                if close[i] < s.msb_level:
                    # BOS confirmado: vela cierra POR DEBAJO del MSB Level
                    bos_bear[i] = True
                    s.bos_bar   = i
                    bos_ok      = True
                elif low[i] <= s.ol_price:
                    # INVALIDACIÓN: precio alcanza el OL (suelo) sin BOS previo.
                    # OL en ventas es el mínimo → low <= OL lo toca.
                    active = None

        # ── FASE 2: BOS confirmado, esperando retest ──────────────────────────
        else:
            # El retest NUNCA ocurre en la misma barra que el BOS
            if i == s.bos_bar:
                continue

            if s.direction == 'bullish':
                if low[i] <= s.msb_level:
                    # RETEST: low de la vela toca o baja del MSB Level → ENTRADA LONG
                    # La vela está abierta — no se espera su cierre.
                    entry_bull[i] = True
                    active  = None
                    bos_ok  = False

                elif high[i] >= s.ol_price:
                    # INVALIDACIÓN: precio llega al OL (TP) sin haber dado el retest.
                    # El setup se cancela.
                    active  = None
                    bos_ok  = False

            else:  # bearish
                if high[i] >= s.msb_level:
                    # RETEST: high de la vela toca o supera el MSB Level → ENTRADA SHORT
                    entry_bear[i] = True
                    active  = None
                    bos_ok  = False

                elif low[i] <= s.ol_price:
                    # INVALIDACIÓN: precio llega al OL (TP) sin haber dado el retest.
                    active  = None
                    bos_ok  = False

    # ── 6. Construir resultado ────────────────────────────────────────────────
    result = data.copy()
    result['bos_bull']       = bos_bull
    result['bos_bear']       = bos_bear
    result['entry_bull']     = entry_bull
    result['entry_bear']     = entry_bear
    result['msb_level_bull'] = msb_bull
    result['msb_level_bear'] = msb_bear
    result['sl_bull']        = sl_bull
    result['sl_bear']        = sl_bear
    result['tp_bull']        = tp_bull
    result['tp_bear']        = tp_bear
    return result


# ─── Wrapper para evaluate_indicator ─────────────────────────────────────────

def evaluate_bos_indicator(
    data:   pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.Series:
    """
    Wrapper compatible con evaluate_indicator(data, indicator_name, params).

    Uso en el motor de backtesting
    --------------------------------
    signal = evaluate_bos_indicator(df, params={'liq_strength': 25, 'signal': 'both'})

    Retorna pd.Series con:
         1.0  → entrada long  (entry_bull en esa barra)
        -1.0  → entrada short (entry_bear en esa barra)
         0.0  → sin señal

    Parámetros opcionales (dict params)
    ------------------------------------
    liq_strength : int  Fuerza del pivot para liquidity channels. Default = 25.
    signal       : str  'both' → long y short | 'bull' → solo long | 'bear' → solo short.
                        Default = 'both'.
    """
    if params is None:
        params = {}

    liq_strength = params.get('liq_strength', 25)
    signal       = params.get('signal', 'both')

    r      = detect_bos(data, liq_strength=liq_strength)
    output = pd.Series(0.0, index=data.index)

    if signal in ('both', 'bull'):
        output[r['entry_bull'].values] = 1.0
    if signal in ('both', 'bear'):
        output[r['entry_bear'].values] = -1.0

    return output


# ─── Test rápido ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n     = 2000
    price = 1.1000 + np.cumsum(np.random.randn(n) * 0.0002)
    opens = np.roll(price, 1); opens[0] = price[0]

    df_test = pd.DataFrame({
        'Open':  opens,
        'High':  price + np.abs(np.random.randn(n) * 0.0004),
        'Low':   price - np.abs(np.random.randn(n) * 0.0004),
        'Close': price,
    })

    # Test estructura de mercado
    pts = detect_market_structure(
        df_test['High'].values,
        df_test['Low'].values,
        df_test['Open'].values,
        df_test['Close'].values,
    )

    print("=" * 55)
    print("MARKET STRUCTURE BOS — MCT Protocol")
    print("=" * 55)
    print(f"  Barras          : {n}")
    for k in ['LL', 'LH', 'HH', 'HL']:
        print(f"  {k} detectados   : {sum(1 for p in pts if p.kind == k)}")

    # Test BOS completo
    r = detect_bos(df_test)
    print(f"\n  BOS alcistas    : {r['bos_bull'].sum()}")
    print(f"  BOS bajistas    : {r['bos_bear'].sum()}")
    print(f"  Entradas long   : {r['entry_bull'].sum()}")
    print(f"  Entradas short  : {r['entry_bear'].sum()}")

    # Test wrapper
    sig = evaluate_bos_indicator(df_test)
    print(f"\n  Señales +1.0    : {(sig == 1.0).sum()}")
    print(f"  Señales -1.0    : {(sig == -1.0).sum()}")
    print("=" * 55)
