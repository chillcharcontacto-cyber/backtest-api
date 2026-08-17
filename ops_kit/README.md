# ops_kit — portable live-bot monitoring & safety

Drop-in operational system extracted from the EMS Hyperliquid bot, so a second bot gets
the same behavior: liveness, Telegram trade-follow, kill switch, crash-safe state, and
never-orphan execution.

**Start here → [`PORT_BRIEF.md`](PORT_BRIEF.md).** It explains the 5 layers, what copies
verbatim vs. what to re-author (with cTrader specifics), and a build order + acceptance
checklist.

Files:
- `daylimit.py` — daily-loss kill switch (verbatim)
- `position.py` — position state + boot reconcile (verbatim)
- `monitor.py` — Telegram + healthchecks senders (verbatim) and generic card formatters
- `loop.py` — the run-forever loop: reconcile → sleep+ping → tick → repeat (verbatim)
- `broker_protocol.py` — the method surface your venue adapter must implement (the seam)
- `render.worker.yaml` — Render worker block (disk + unbuffered + secrets)
- `.env.example` — local env template

This kit is venue-agnostic; the reference implementation it was lifted from is
`ems_live/` in this repo (read it alongside the brief).
