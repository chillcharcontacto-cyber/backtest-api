"""
EMS brain: license binding, decision signing, and decide() wiring.

The strategy predicates + feed are mocked, so these run offline — strategy
correctness itself is covered by the parity tests (test_live_parity.py). Here we
prove the BRAIN wiring: only bound license keys pass, and every decision is signed,
bound to the caller's account, nonce'd, and expiring.
"""
import os

from eth_account import Account

import ems_brain as brain

# One throwaway signing key for the whole module.
_ACCT = Account.create()
os.environ["EMS_BRAIN_SIGNING_KEY"] = _ACCT.key.hex()
SIGNER = _ACCT.address

OTHER = "0x0000000000000000000000000000000000000009"


# ----------------------------- signing ----------------------------- #

def test_sign_verify_roundtrip():
    p = {"action": "enter_long", "sl": 100.0, "nonce": "abc"}
    sig = brain.sign_payload(p)
    assert brain.verify_payload(p, sig, SIGNER)
    assert not brain.verify_payload({**p, "sl": 999.0}, sig, SIGNER)   # tamper -> fail
    assert not brain.verify_payload(p, sig, OTHER)                     # wrong signer -> fail


def test_signer_address_matches_key():
    assert brain.signer_address().lower() == SIGNER.lower()


# ----------------------------- licensing --------------------------- #

def test_license_binding(monkeypatch):
    addr = "0xAbC0000000000000000000000000000000000001"
    monkeypatch.setenv("EMS_LICENSE_KEYS", '{"KEY-good": "%s"}' % addr)
    assert brain.verify_license("KEY-good", addr)[0] is True
    assert brain.verify_license("KEY-good", addr.lower())[0] is True   # case-insensitive
    assert brain.verify_license("KEY-good", OTHER)[0] is False         # wrong account
    assert brain.verify_license("KEY-unknown", addr)[0] is False       # unknown key
    assert brain.verify_license(None, addr)[0] is False
    assert brain.verify_license("KEY-good", None)[0] is False


def test_empty_map_denies_all(monkeypatch):
    monkeypatch.delenv("EMS_LICENSE_KEYS", raising=False)
    assert brain.verify_license("anything", "0xabc")[0] is False


# ------------------------- decide() wiring ------------------------- #

class _FakeCtx:
    m30_times = [None] * 5 + ["2026-08-21 10:30:00+00:00"]   # index -1 is str-able
    thirty_min = "30m"


class _Entry:
    anchor_time = "a"
    crossover_time = "c"


class _ExitSig:
    reason = "H1_EMA100"


def _install(monkeypatch, entry=None, exit_=None):
    monkeypatch.setattr(brain, "load_frames", lambda cfg: ([0] * 6, None, None))
    monkeypatch.setattr(brain, "build_ctx", lambda m30, h1, h4: _FakeCtx())
    monkeypatch.setattr(brain, "check_entry_live", lambda ctx, i, cfg, iv: entry)
    monkeypatch.setattr(brain, "check_h1_exit", lambda ctx, i, e, s: exit_)
    monkeypatch.setattr(brain, "adapt_sl_to_hl", lambda a, c, coin, url: 63000.0)


def test_decide_enter_is_signed_and_bound(monkeypatch):
    _install(monkeypatch, entry=_Entry())
    out = brain.decide({"account_address": "0xABC", "nonce": "n1", "in_position": False})
    p = out["payload"]
    assert p["action"] == "enter_long" and p["sl"] == 63000.0
    assert p["account_address"] == "0xabc" and p["nonce"] == "n1"
    assert p["expiry"] > p["issued_at"]
    assert brain.verify_payload(p, out["signature"], SIGNER)


def test_decide_none_when_flat_no_signal(monkeypatch):
    _install(monkeypatch, entry=None)
    out = brain.decide({"account_address": "0xABC", "nonce": "n", "in_position": False})
    assert out["payload"]["action"] == "none" and out["payload"]["sl"] is None
    assert brain.verify_payload(out["payload"], out["signature"], SIGNER)


def test_decide_exit_when_in_position(monkeypatch):
    _install(monkeypatch, exit_=_ExitSig())
    out = brain.decide({"account_address": "0xABC", "nonce": "n",
                        "in_position": True, "entry_price": 64000, "sl_price": 63000})
    assert out["payload"]["action"] == "exit"
    assert brain.verify_payload(out["payload"], out["signature"], SIGNER)


def test_decide_hold_when_in_position_no_exit(monkeypatch):
    _install(monkeypatch, exit_=None)
    out = brain.decide({"account_address": "0xABC", "nonce": "n",
                        "in_position": True, "entry_price": 64000, "sl_price": 63000})
    assert out["payload"]["action"] == "hold"
    assert brain.verify_payload(out["payload"], out["signature"], SIGNER)
