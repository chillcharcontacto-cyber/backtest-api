"""
Shared HTTP retry for transient exchange rate-limits / gateway blips (429/5xx).

Used by broker.py (Hyperliquid SDK calls) AND feed.py (Binance/HL candle fetches),
so a transient rate-limit on ANY exchange touchpoint is ridden out instead of aborting
the bar. This is what protects live ENTRIES from a 429 landing on the signal bar.

READS retry on 429 + 5xx (idempotent). WRITES (orders) retry on 429 ONLY — a 429 is
rejected before the matching engine so it cannot double-execute, whereas a 5xx could
hide a lost success.
"""
import random
import time
from typing import Optional

TRANSIENT_READ = frozenset({429, 500, 502, 503, 504})
TRANSIENT_WRITE = frozenset({429})


def http_status(exc) -> Optional[int]:
    """Best-effort HTTP status from a hyperliquid ClientError or a requests error."""
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code
    return None


def is_rate_limit(exc) -> bool:
    """True if `exc` is a transient rate-limit / gateway error (429 or 5xx)."""
    return http_status(exc) in TRANSIENT_READ


def retry(fn, statuses, tries: int = 6, base_delay: float = 1.0, max_delay: float = 15.0):
    """
    Call fn(); on a transient HTTP status (in `statuses`) retry with capped exponential
    backoff + jitter (~1,2,4,8,15,15s + up to 0.5s random). Any non-transient error —
    or the final attempt — raises, so a genuine outage still surfaces. Worst-case total
    wait ~45s, well within a 30-min bar. Jitter de-syncs retries so a shared-IP rate
    limit clears instead of being hammered in lockstep.
    """
    delay = base_delay
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries or http_status(e) not in statuses:
                raise
            time.sleep(min(delay, max_delay) + random.uniform(0, 0.5))
            delay *= 2
