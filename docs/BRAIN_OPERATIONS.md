# EMS Brain — Operator Playbook (Tier-4)

How you (the operator) run the protected partner bot: the **brain** lives in *your* private
`backtest-api` web service and holds the strategy; each partner runs a **thin client**
(public repo `ems-thin-client`) that only executes your **signed** decisions on **their** HL
account. A partner's bot works only with a license key **you** issued and **bound to their
account** — so forwarded files are inert, and you can revoke anyone instantly.

Keep this doc PRIVATE (it's in backtest-api, not in the public thin-client repo).

---

## Architecture

```
partner's Render worker (ems-thin-client, PUBLIC, no strategy)
      │  POST /ems/decision   { license_key, my_HL_address, in_position, ... }
      ▼
your Render web service (backtest-api, PRIVATE)  ── /ems/decision ──►  ems_brain.py
      │  runs EMS V3 on fresh candles, SIGNS the decision                 (the strategy)
      ▼
      { payload:{action, sl, bound to that account, nonce, expiry}, signature }
      │  client verifies signature=you, account=mine, nonce, not expired
      ▼
partner's client sizes it locally + places the order on THEIR Hyperliquid account
```

You hold: the strategy, the signing key, the approved-keys list. Partners hold: their own
HL keys + funds. You never touch their money.

---

## One-time setup (≈15 min, once)

**1. Generate your signing keypair** (locally, never commit it):
```bash
py scripts/gen_brain_keys.py
```
It prints a **PRIVATE KEY** and a **SIGNER ADDRESS**. Copy both.

**2. Arm the brain** on the `backtest-api` web service (Render dashboard → the *web service*,
not the bot worker → **Environment**):
- `EMS_BRAIN_SIGNING_KEY` = the private key from step 1  (**sync:false / secret**)
- Leave `EMS_LICENSE_KEYS` unset for now (no one approved yet → every call is denied).
- **Save** (redeploys the web service).

The brain is now live but grants nobody. Verify it's up: open your API URL + `/health`
(find the URL at the top of the web service page, e.g. `https://backtest-api-xxxx.onrender.com`).

**3. Publish the thin-client template** (public repo `ems-thin-client`):
- In its `render.yaml`, set `EMS_BRAIN_URL` = your API URL, `EMS_BRAIN_SIGNER` = your signer
  address (step 1). In its `README.md`, replace `OPERATOR_GITHUB` in the Deploy button with
  your GitHub handle.
- `git init && git add -A && git commit -m "EMS thin client" ` → create the repo on GitHub as
  **public** → push.

Done. You can now issue keys.

---

## Issue a license key to a partner

1. Get the partner's **Hyperliquid main address** (`0x…`).
2. Mint a key:
   ```bash
   py -c "import secrets; print('EMS-' + secrets.token_hex(12))"
   ```
3. On the web service → **Environment** → `EMS_LICENSE_KEYS` (JSON; create it if absent) →
   add the partner's line, then **Save**:
   ```json
   {
     "EMS-aaaa1111...": "0xPartnerAAddress",
     "EMS-bbbb2222...": "0xPartnerBAddress"
   }
   ```
4. Send the partner **their key**. They follow `ems-thin-client/SETUP.md`.

That key now works **only** from that one HL account. A different account (or no key) → the
brain returns 403 and the client does nothing.

## Revoke / kill a partner

Remove their line from `EMS_LICENSE_KEYS` → **Save**. Their bot is denied on its next bar
(within ~30 min) and stops trading. (Their open position, if any, still has its own resting
stop on the exchange — revoking only stops NEW actions.)

---

## Smoke-test the whole chain (do this before onboarding anyone)

1. Issue a key bound to **your own** test HL address.
2. Deploy a thin client (or run locally: `EMS_DRY_RUN=true python -m thin.run once`) with your
   brain URL, signer, that key, and your address.
3. Watch it: it should fetch a signed decision and log `enter_long/none/hold/exit` with no
   signature/expiry/account errors. In dry-run it places no orders.
4. Flip a bad value (wrong signer, or revoke the key) and confirm the client refuses to act.

---

## Security & cost

- **`EMS_BRAIN_SIGNING_KEY` is the crown jewel** — anyone with it can sign decisions. Keep it
  only in the web-service env. To rotate: regenerate, update the env var, and ship the new
  signer address in the thin-client repo (partners redeploy to pick it up).
- The brain is **dormant** without both env vars set → safe to leave deployed.
- **Cost: ~$0** — the brain rides the existing `backtest-api` web service. Free tier is fine
  (a ~30–60s cold-start on the first call after idle is nothing at 30-min bars); upgrade that
  one service to Starter (~$7/mo) only if you want zero delay.

---

## Your activation checklist (the only things only you can do)

- [ ] `py scripts/gen_brain_keys.py` → save the private key + signer address
- [ ] Set `EMS_BRAIN_SIGNING_KEY` on the backtest-api **web service**; Save
- [ ] Confirm `/health` responds; note your API URL
- [ ] Fill `EMS_BRAIN_URL` + `EMS_BRAIN_SIGNER` in `ems-thin-client/render.yaml` + README, publish it public
- [ ] Smoke-test with your own address in dry-run
- [ ] Per partner: mint a key, add `{key: their_address}` to `EMS_LICENSE_KEYS`, Save, send them the key
