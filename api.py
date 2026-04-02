"""
=============================================================
  Backtesting SaaS — FastAPI Backend
=============================================================
Instalacion:
    pip install fastapi uvicorn

Arrancar el servidor:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /backtest        → ejecutar backtest
    GET  /health          → estado del servidor
    GET  /schema          → schema de input con defaults y validacion

Probar desde terminal:
    curl -X POST http://localhost:8000/backtest \
         -H "Content-Type: application/json" \
         -d '{"ticker":"EURUSD=X","timeframe":"1h","start":"2025-01-01","end":"2025-03-01","rsi_entry":40,"rsi_exit":60}'
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# Importar el motor de backtest
from backtest_rsi_ema import backtest, INPUT_SCHEMA, TIMEFRAME_FREQ

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Backtesting SaaS API",
    description = "Motor de backtesting RSI + EMA. Envía un config JSON, recibe resultados completos.",
    version     = "1.0.0",
)

# CORS — permite que la web frontend pueda llamar a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # En producción: poner solo tu dominio
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────
class BacktestConfig(BaseModel):
    ticker:     str   = Field("BTC-USD",    description="Activo. Forex: EURUSD=X, GBPUSD=X")
    timeframe:  str   = Field("1d",         description="1m 5m 15m 30m 1h 2h 4h 1d 1wk 1mo")
    start:      str   = Field("2021-01-01", description="Fecha inicio YYYY-MM-DD")
    end:        str   = Field(...,          description="Fecha fin YYYY-MM-DD")
    rsi_period: int   = Field(14,           ge=2,    le=100,  description="Periodo RSI")
    ema_period: int   = Field(50,           ge=2,    le=500,  description="Periodo EMA")
    rsi_entry:  float = Field(30.0,         ge=1,    le=49,   description="Nivel sobreventa (entry)")
    rsi_exit:   float = Field(70.0,         ge=51,   le=99,   description="Nivel sobrecompra (exit)")
    init_cash:  float = Field(10000.0,      ge=1,             description="Capital inicial USD")
    fees:       float = Field(0.001,        ge=0,    le=0.1,  description="Comision (0.001 = 0.1%)")
    size:       float = Field(0.99,         ge=0.01, le=1.0,  description="Fraccion equity por trade")
    slippage:   float = Field(0.0005,       ge=0,    le=0.05, description="Slippage")

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    """Comprueba que el servidor está activo."""
    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version":   "1.0.0",
    }


@app.get("/schema", tags=["Sistema"])
def schema():
    """Devuelve el schema de input con defaults, tipos y rangos."""
    fields = {}
    for key, rules in INPUT_SCHEMA.items():
        fields[key] = {
            "type":    rules["type"].__name__,
            "default": rules["default"],
        }
        if "min" in rules: fields[key]["min"] = rules["min"]
        if "max" in rules: fields[key]["max"] = rules["max"]

    return {
        "fields":     fields,
        "timeframes": list(TIMEFRAME_FREQ.keys()),
        "notes": {
            "forex":    "Añadir =X al par: EURUSD=X, GBPUSD=X, USDJPY=X",
            "intraday": "Timeframes < 1d tienen máximo ~60 días de histórico (yfinance gratuito)",
        }
    }


@app.post("/backtest", tags=["Backtest"])
def run(config: BacktestConfig):
    """
    Ejecuta un backtest con la estrategia RSI + EMA.

    Devuelve:
    - **summary**: métricas principales (return, sharpe, drawdown, winrate...)
    - **analysis**: métricas avanzadas (expectancy, avg R:R, racha perdedora, equity smoothness)
    - **trades**: lista completa de operaciones
    - **equity_curve**: curva de valor del portfolio lista para graficar
    """
    result = backtest(config.model_dump())

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ─────────────────────────────────────────────────────────────
# ARRANQUE DIRECTO
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
