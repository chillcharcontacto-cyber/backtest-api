import csv
from typing import List

from .engine import Trade


def trades_to_csv(trades: List[Trade], path: str) -> None:
    """Write trade list to CSV. Columns: R-multiple focused per spec."""
    if not trades:
        print("No trades to write.")
        return

    fieldnames = [
        "entry_time",
        "exit_time",
        "entry_price",
        "sl_price",
        "exit_price",
        "exit_reason",
        "r_multiple",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "entry_time":  t.entry_time.isoformat(),
                "exit_time":   t.exit_time.isoformat(),
                "entry_price": t.entry_price,
                "sl_price":    t.sl_price,
                "exit_price":  t.exit_price,
                "exit_reason": t.exit_reason,
                "r_multiple":  t.r_multiple,
            })
