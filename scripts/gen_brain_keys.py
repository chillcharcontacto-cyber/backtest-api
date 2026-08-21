"""
Generate the EMS brain signing keypair (one-time, by the operator).

    python scripts/gen_brain_keys.py

- The PRIVATE key goes in the web-service env var EMS_BRAIN_SIGNING_KEY (keep secret;
  it lets the brain sign decisions).
- The ADDRESS gets embedded in the thin client so it can verify decisions came from you.

Run it locally, never commit the private key. Rotating the key = regenerate, update the
env var, and ship the new address in the client.
"""
from eth_account import Account


def main():
    acct = Account.create()
    print("=" * 68)
    print("EMS BRAIN SIGNING KEYPAIR — generated locally, store carefully")
    print("=" * 68)
    print("\nSIGNING PRIVATE KEY  -> web service env  EMS_BRAIN_SIGNING_KEY")
    print("  (SECRET — never commit, never share, never put in the client)")
    print(f"  {acct.key.hex()}")
    print("\nSIGNER ADDRESS       -> embed in the thin client (EMS_BRAIN_SIGNER)")
    print("  (public — the client uses it to verify decisions are really yours)")
    print(f"  {acct.address}")
    print("\n" + "=" * 68)


if __name__ == "__main__":
    main()
