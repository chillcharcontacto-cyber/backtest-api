import csv
from typing import List

from .engine import Trade


def trades_to_csv(trades: List[Trade], path: str) -> None:
    """
    Write trade list to CSV using Samuel's 11-column Quantprove schema.

    Columns:
      trade_id  - sequential 1-based integer
      strategy  - from Trade.strategy ("EMA-Cross" | "EMA-Cross-H4F")
      date      - entry date DD/MM/YYYY UTC
      time      - entry time HH:MM UTC
      pair      - "BTC"
      direction - "Long"
      result    - "TP" if r_multiple > 0 else "SL"
      rr        - r_multiple rounded to 2dp
      duration  - hours between entry and exit, e.g. "14h"
      sl_size   - abs(entry_price - sl_price) in price units, rounded to 2dp
      exit_reason - "STRUCTURAL_SL" | "H1_EMA100"
    """
    if not trades:
        print("No trades to write.")
        return

    fieldnames = [
        "trade_id", "strategy", "date", "time",
        "pair", "direction", "result", "rr",
        "duration", "sl_size", "exit_reason",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t in enumerate(trades, start=1):
            hours    = (t.exit_time - t.entry_time).total_seconds() / 3600
            sl_size  = abs(t.entry_price - t.sl_price)
            result   = "TP" if t.r_multiple > 0 else "SL"
            writer.writerow({
                "trade_id":    idx,
                "strategy":    t.strategy,
                "date":        t.entry_time.strftime("%d/%m/%Y"),
                "time":        t.entry_time.strftime("%H:%M"),
                "pair":        "BTC",
                "direction":   "Long",
                "result":      result,
                "rr":          round(t.r_multiple, 2),
                "duration":    f"{int(hours)}h",
                "sl_size":     round(sl_size, 2),
                "exit_reason": t.exit_reason,
            })
