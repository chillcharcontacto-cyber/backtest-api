"""
EMS decision brain — the server-side strategy oracle for Tier-4 partner bots.

A licensed thin client POSTs its current state (flat, or in-position with entry+SL);
the brain runs the EMS V3 decision (entry timing + HL-adapted SL, or the H1 exit) on
fresh candles and returns a SIGNED instruction. The strategy logic never leaves the
server: a stripped or forwarded client is inert without an APPROVED license key that is
BOUND to that client's own Hyperliquid address — and only the operator can issue one.

Split of responsibility:
  brain (here)  : WHEN to enter/exit + WHERE the stop goes (the secret edge)
  thin client   : sizing (its own risk_usd + equity), order placement, state, alerts

Signing uses eth_account (already a dependency via the Hyperliquid SDK — no new lib).
Each decision is signed over a canonical JSON payload; the client verifies the signer
is the operator's known address and that the decision is bound to its own account, not
expired, and matches the nonce it sent (blocks forged and replayed instructions).

Env (set on the web service):
  EMS_BRAIN_SIGNING_KEY : 0x… secp256k1 private key used to SIGN decisions.
                          Generate with scripts/gen_brain_keys.py; keep secret.
  EMS_LICENSE_KEYS      : JSON {"KEY": "0xBoundAddress", ...} — approved keys mapped to
                          the ONE HL address each may trade. Add a line = approve;
                          remove a line = revoke (kills that client on its next call).
"""
import json
import os
import time
from typing import Optional, Tuple

from eth_account import Account
from eth_account.messages import encode_defunct

from ems_live.config import LiveConfig
from ems_live.decider import build_ctx, check_entry_live, check_h1_exit
from ems_live.sl_adapter import adapt_sl_to_hl


BRAIN_TTL_SEC = 120          # a signed decision is valid for this many seconds


# --------------------------------------------------------------------------- #
#  Licensing (operator-controlled approval + per-account binding)              #
# --------------------------------------------------------------------------- #

def _license_map() -> dict:
    """{license_key: bound_hl_address} from env EMS_LICENSE_KEYS (JSON). {} if unset."""
    raw = os.environ.get("EMS_LICENSE_KEYS", "").strip()
    if not raw:
        return {}
    try:
        m = json.loads(raw)
        return m if isinstance(m, dict) else {}
    except json.JSONDecodeError:
        return {}


def verify_license(key: Optional[str], address: Optional[str]) -> Tuple[bool, str]:
    """
    A call is authorized iff `key` is approved AND `address` is the exact HL account
    bound to that key. Unknown key, or a key used from a different account, is denied —
    so a forwarded client (different account, or no valid key) cannot trade.
    """
    if not key or not address:
        return False, "missing license key or account address"
    bound = _license_map().get(key)
    if bound is None:
        return False, "license key not recognized (ask the operator to issue/approve one)"
    if bound.strip().lower() != address.strip().lower():
        return False, "license key is bound to a different account"
    return True, "ok"


# --------------------------------------------------------------------------- #
#  Signing (eth_account; client verifies against the operator's known address) #
# --------------------------------------------------------------------------- #

def _signing_key() -> str:
    k = os.environ.get("EMS_BRAIN_SIGNING_KEY")
    if not k:
        raise RuntimeError("EMS_BRAIN_SIGNING_KEY not set — cannot sign decisions")
    return k


def signer_address() -> str:
    """The public address the client must embed to verify decisions."""
    return Account.from_key(_signing_key()).address


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_payload(payload: dict) -> str:
    sig = Account.sign_message(encode_defunct(text=_canonical(payload)), _signing_key())
    return sig.signature.hex()


def verify_payload(payload: dict, signature: str, expected_signer: str) -> bool:
    """Reference verifier (used by tests and mirrored in the thin client)."""
    try:
        rec = Account.recover_message(encode_defunct(text=_canonical(payload)),
                                      signature=signature)
    except Exception:
        return False
    return rec.strip().lower() == expected_signer.strip().lower()


# --------------------------------------------------------------------------- #
#  Decision                                                                    #
# --------------------------------------------------------------------------- #

def _brain_cfg() -> LiveConfig:
    """Canonical EMS V3 config. testnet=False -> HL mainnet candles for SL adaptation."""
    return LiveConfig(testnet=False)


def load_frames(cfg: LiveConfig):
    """Fetch + decorate the Binance signal frames. Split out so tests can inject."""
    from ems_live.runner import load_signal_frames
    return load_signal_frames(cfg)


def decide(req: dict) -> dict:
    """
    Compute the signed decision for a licensed client.

    req: {license_key, account_address, coin, in_position, entry_price, sl_price, nonce}
    Returns {"payload": {...signed fields...}, "signature": "0x..."}.

    action in:
      enter_long : flat + fresh entry signal — `sl` is the HL-adapted stop to place
      none       : flat, no signal — do nothing
      exit       : in-position + H1 EMA100 exit fired — flatten
      hold       : in-position, no exit — keep managing (the resting stop still protects)
    Caller must have already passed verify_license().
    """
    cfg = _brain_cfg()
    m30, h1, h4 = load_frames(cfg)
    ctx = build_ctx(m30, h1, h4)
    i = len(m30) - 1
    bar_time = str(ctx.m30_times[i])

    if req.get("in_position"):
        ex = check_h1_exit(ctx, i, float(req["entry_price"]), float(req["sl_price"]))
        if ex is not None:
            action, sl, reason = "exit", None, ex.reason
        else:
            action, sl, reason = "hold", None, "no exit this bar"
    else:
        sig = check_entry_live(ctx, i, cfg, ctx.thirty_min)
        if sig is not None:
            sl_hl = adapt_sl_to_hl(sig.anchor_time, sig.crossover_time,
                                   cfg.coin, cfg.hl_api_url)
            action, sl, reason = "enter_long", float(sl_hl), "entry signal"
        else:
            action, sl, reason = "none", None, "no entry signal"

    now = int(time.time())
    payload = {
        "action":          action,
        "coin":            cfg.coin,
        "account_address": str(req["account_address"]).strip().lower(),
        "sl":              sl,
        "bar_time":        bar_time,
        "issued_at":       now,
        "expiry":          now + BRAIN_TTL_SEC,
        "nonce":           str(req["nonce"]),
        "reason":          reason,
    }
    return {"payload": payload, "signature": sign_payload(payload)}
