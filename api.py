"""
=============================================================
  Backtesting SaaS — FastAPI Backend con API Keys
=============================================================
Arrancar:
    uvicorn api:app --reload --port 8000

Gestionar keys desde variable de entorno VALID_API_KEYS:
    En Render → Environment → VALID_API_KEYS = key1,key2,key3

Generar una key nueva (ejecutar en terminal):
    python -c "import secrets; print('TEL-' + secrets.token_hex(16))"
=============================================================
"""

import os
import secrets
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backtest_rsi_ema import backtest, INPUT_SCHEMA, TIMEFRAME_FREQ

# ─────────────────────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────────────────────
# Las keys se leen de la variable de entorno VALID_API_KEYS
# Formato: "key1,key2,key3"
# En Render: Settings → Environment → añadir VALID_API_KEYS
#
# Si la variable no está definida, se usa una key de desarrollo
# IMPORTANTE: en producción siempre define VALID_API_KEYS en Render

def get_valid_keys() -> set:
    raw = os.environ.get("VALID_API_KEYS", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Valida la API key enviada en el header X-Api-Key."""
    valid_keys = get_valid_keys()

    if not valid_keys:
        # Sin keys configuradas → modo desarrollo, acceso libre
        return True

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key requerida. Incluye el header X-Api-Key con tu clave."
        )

    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="API key inválida o expirada. Contacta con soporte."
        )

    return True


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "TradingEdgeLabs — Backtesting API",
    description = "Motor de backtesting RSI + EMA. Requiere API key en header X-Api-Key.",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────
class BacktestConfig(BaseModel):
    ticker:     str   = Field("BTC-USD",    description="Activo. Forex: EURUSD=X")
    timeframe:  str   = Field("1d",         description="1m 5m 15m 30m 1h 2h 4h 1d 1wk 1mo")
    start:      str   = Field("2021-01-01", description="Fecha inicio YYYY-MM-DD")
    end:        str   = Field(...,          description="Fecha fin YYYY-MM-DD")
    rsi_period: int   = Field(14,  ge=2,    le=100)
    ema_period: int   = Field(50,  ge=2,    le=500)
    rsi_entry:  float = Field(30.0, ge=1,   le=49)
    rsi_exit:   float = Field(70.0, ge=51,  le=99)
    init_cash:  float = Field(10000.0, ge=1)
    fees:       float = Field(0.001, ge=0,  le=0.1)
    size:       float = Field(0.99,  ge=0.01, le=1.0)
    slippage:   float = Field(0.0005, ge=0, le=0.05)

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "BTC-USD", "timeframe": "1d",
                "start": "2024-01-01", "end": "2024-06-01",
                "rsi_period": 14, "ema_period": 20,
                "rsi_entry": 48, "rsi_exit": 52,
                "init_cash": 10000, "fees": 0.001,
            }
        }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    """Estado del servidor. No requiere autenticación."""
    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version":   "2.0.0",
        "auth":      "enabled" if get_valid_keys() else "disabled (dev mode)",
    }


@app.get("/schema", tags=["Sistema"])
def schema(x_api_key: Optional[str] = Header(None)):
    """Schema de input. Requiere API key."""
    verify_api_key(x_api_key)
    fields = {}
    for key, rules in INPUT_SCHEMA.items():
        fields[key] = {"type": rules["type"].__name__, "default": rules["default"]}
        if "min" in rules: fields[key]["min"] = rules["min"]
        if "max" in rules: fields[key]["max"] = rules["max"]
    return {
        "fields":     fields,
        "timeframes": list(TIMEFRAME_FREQ.keys()),
    }


@app.post("/backtest", tags=["Backtest"])
def run(config: BacktestConfig, x_api_key: Optional[str] = Header(None)):
    """
    Ejecuta un backtest RSI + EMA.
    Requiere API key en el header: X-Api-Key: tu-clave
    """
    verify_api_key(x_api_key)

    result = backtest(config.model_dump())

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ─────────────────────────────────────────────────────────────
# ARRANQUE
# ─────────────────────────────────────────────────────────────
@app.post("/export/pinescript", tags=["Export"])
def export_pinescript(config: StrategyConfig, x_api_key: Optional[str] = Header(None)):
    """
    Runs backtest and returns a Pine Script v6 file with all trades
    hardcoded as labels and lines at exact timestamps.
    """
    verify_api_key(x_api_key)
    from engine import run_strategy
    result = run_strategy(config.model_dump())
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["error"])

    trades  = result.get("trades", [])
    ticker  = config.market.get("ticker", "SYMBOL")
    tf      = config.market.get("timeframe", "5m")
    start   = config.market.get("start", "")
    end     = config.market.get("end", "")


    def ts(dt_str):
        """Convert datetime string to Pine timestamp(year,month,day,hour,minute)."""
        if not dt_str: return '"0"'
        try:
            from datetime import datetime as _dt
            if "T" in dt_str:
                d = _dt.fromisoformat(dt_str.replace("Z",""))
            elif " " in str(dt_str):
                d = _dt.strptime(str(dt_str)[:16], "%Y-%m-%d %H:%M")
            else:
                p = str(dt_str).split("-")
                return f'timestamp("{p[0]}, {p[1]}, {p[2]}, 0, 0")'
            return f'timestamp("{d.year}, {d.month}, {d.day}, {d.hour}, {d.minute}")'
        except Exception:
            return f'"{dt_str}"'

    lines = []
    lines.append('//@version=6')
    lines.append(f'indicator("MCT — {ticker} {tf} | {start}→{end} | {len(trades)} trades", overlay=true, max_labels_count=500, max_lines_count=500, max_boxes_count=200)')
    lines.append('')
    lines.append(f'// Auto-generated by TradingEdgeLabs | {ticker} {tf} {start}→{end} | {len(trades)} trades')
    lines.append('')

    if not trades:
        lines.append('label.new(bar_index, close, "No trades found", style=label.style_label_down)')
    else:
        for t in trades:
            ctx       = t.get("context", {})
            entry_dt  = t.get("entry_date", "")
            exit_dt   = t.get("exit_date",  "")
            direction = t.get("direction", "bull")
            entry_p   = t.get("entry_price", 0)
            sl_p      = t.get("sl_price", 0)
            tp_p      = t.get("tp_price", 0)
            ol_p      = t.get("ol_price", 0)
            ret       = t.get("return_pct", 0)
            status    = t.get("status", "loss")
            exit_type = t.get("exit_type", "normal")
            tid       = t.get("trade_id", 0)
            ss_ts     = ctx.get("ss_ts", "")
            ss_price  = ctx.get("ss_price", 0)
            se_ts     = ctx.get("se_ts", "")
            se_price  = ctx.get("se_price", 0)
            ol_ts     = ctx.get("ol_ts", "")
            ctx_ol    = ctx.get("ol_price", ol_p)
            div_ts    = ctx.get("div_ts", "")
            div_model = ctx.get("div_model", "")
            bos_ts    = ctx.get("bos_ts", entry_dt)
            bos_price = ctx.get("bos_price", entry_p)

            is_bull    = direction == "bull"
            col_entry  = "color.lime" if is_bull else "color.red"
            col_result = "color.lime" if status == "win" else "color.red"
            arrow      = "up" if is_bull else "down"
            arrow_sym  = "▲" if is_bull else "▼"
            result_sym = "WIN" if status == "win" else "LOSS"
            ret_str    = f"{ret:+.2f}%"
            lbl_up     = "label.style_label_up"
            lbl_dn     = "label.style_label_down"
            entry_style= lbl_up if is_bull else lbl_dn
            opp_style  = lbl_dn if is_bull else lbl_up

            lines.append(f'')
            lines.append(f'// ── Trade #{tid} {arrow_sym} {entry_dt}→{exit_dt} {result_sym} {ret_str} ──')

            # SS — Sweep Start
            if ss_ts:
                lines.append(f'if time == {ts(ss_ts)}')
                lines.append(f'    label.new(bar_index, {"low" if is_bull else "high"}, "#{tid} SS\\n{ss_price}", color=color.new(color.gray,40), textcolor=color.white, size=size.tiny, style={opp_style}, tooltip="Sweep Start")')

            # SE — Sweep End / Sweep Level
            if se_ts:
                se_col = "color.new(color.lime,30)" if is_bull else "color.new(color.red,30)"
                lines.append(f'if time == {ts(se_ts)}')
                lines.append(f'    label.new(bar_index, {"low" if is_bull else "high"}, "#{tid} SWEEP\\n{se_price}", color={se_col}, textcolor=color.black, size=size.tiny, style={opp_style}, tooltip="Sweep Level")')

            # OL — Opposite Level
            if ol_ts:
                lines.append(f'if time == {ts(ol_ts)}')
                lines.append(f'    label.new(bar_index, {"high" if is_bull else "low"}, "#{tid} OL\\n{ctx_ol}", color=color.new(color.orange,20), textcolor=color.black, size=size.tiny, style={entry_style}, tooltip="Opposite Level")')

            # DIV — Divergence model
            if div_ts and div_model:
                div_short = div_model.replace("Model ","M").replace(" — ","\\n")
                lines.append(f'if time == {ts(div_ts)}')
                lines.append(f'    label.new(bar_index, {"low" if is_bull else "high"}, "#{tid} DIV\\n{div_short}", color=color.new(color.purple,20), textcolor=color.white, size=size.tiny, style={entry_style}, tooltip="{div_model}")')

            # BOS
            if bos_ts:
                lines.append(f'if time == {ts(bos_ts)}')
                lines.append(f'    label.new(bar_index, {bos_price}, "#{tid} BOS {arrow_sym}", color={col_entry}, textcolor=color.black, size=size.small, style={entry_style}, tooltip="Break of Structure")')

            # ENTRY + lines
            div_short2 = (div_model[:20] if div_model else "n/a").replace('"',"'")
            entry_txt = f"#{tid} ENTRY {arrow_sym}\\nEntry:{entry_p}\\nSL:{sl_p}\\nTP:{tp_p}\\nOL:{ol_p}\\n{div_short2}"
            lines.append(f'if time == {ts(entry_dt)}')
            lines.append(f'    label.new(bar_index, {entry_p}, "{entry_txt}", color={col_entry}, textcolor=color.black, size=size.small, style={entry_style})')
            if sl_p:
                lines.append(f'    line.new(bar_index, {sl_p}, bar_index+30, {sl_p}, color=color.red, width=1, style=line.style_dashed)')
            if tp_p:
                lines.append(f'    line.new(bar_index, {tp_p}, bar_index+30, {tp_p}, color=color.lime, width=1, style=line.style_dashed)')
            if ol_p and round(float(ol_p),5) != round(float(tp_p),5):
                lines.append(f'    line.new(bar_index, {ol_p}, bar_index+30, {ol_p}, color=color.orange, width=1, style=line.style_dotted)')

            # EXIT
            exit_txt = f"#{tid} {result_sym}\\n{ret_str}\\n{exit_type}"
            lines.append(f'if time == {ts(exit_dt)}')
            lines.append(f'    label.new(bar_index, {tp_p if tp_p else entry_p}, "{exit_txt}", color={col_result}, textcolor=color.black, size=size.small, style={entry_style})')


    pine_code = '\n'.join(lines)

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=pine_code,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="MCT_{ticker}_{start}_{end}.pine"'}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


# ─────────────────────────────────────────────────────────────
# NUEVO — Motor universal de estrategias
# ─────────────────────────────────────────────────────────────
from engine import run_strategy, INDICATOR_CATALOG, CONDITION_LABELS


class StrategyConfig(BaseModel):
    market: dict   = Field(..., description="ticker, timeframe, start, end")
    risk:   dict   = Field(..., description="capital, fees, slippage, size")
    strategy: dict = Field(..., description="entry_confirmations, exit_confirmations")

    class Config:
        json_schema_extra = {
            "example": {
                "market":   {"ticker": "AAPL", "timeframe": "1d", "start": "2024-01-01", "end": "2024-06-01"},
                "risk":     {"capital": 10000, "fees": 0.001, "slippage": 0.0005, "size": 0.99},
                "strategy": {
                    "entry_confirmations": [
                        {"indicator": "rsi", "params": {"period": 14}, "condition": "crosses_above", "value": 48},
                        {"indicator": "ema", "params": {"period": 20}, "condition": "price_above"}
                    ],
                    "exit_confirmations": [
                        {"indicator": "rsi", "params": {"period": 14}, "condition": "crosses_below", "value": 52}
                    ]
                }
            }
        }


@app.get("/indicators", tags=["Builder"])
def get_indicators(x_api_key: Optional[str] = Header(None)):
    """Catálogo de indicadores disponibles para el builder."""
    verify_api_key(x_api_key)
    return {"indicators": INDICATOR_CATALOG, "conditions": CONDITION_LABELS}


@app.post("/strategy", tags=["Builder"])
def run_strategy_endpoint(config: StrategyConfig, x_api_key: Optional[str] = Header(None)):
    """
    Ejecuta un backtest con estrategia personalizada (N confirmaciones).
    """
    verify_api_key(x_api_key)
    result = run_strategy(config.model_dump())
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["error"])
    return result
