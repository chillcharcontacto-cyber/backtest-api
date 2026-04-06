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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
