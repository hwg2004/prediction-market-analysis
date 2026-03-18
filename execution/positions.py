import time
from dataclasses import dataclass
from typing import Optional

from clients.kalshi import KalshiMarketClient


@dataclass
class PositionRow:
    ticker: str
    side: str               # "yes" (positive contracts) | "no" (negative)
    contracts: float
    entry_cost_dollars: float
    current_mid: Optional[int]   # cents
    unrealized_pnl: Optional[float]  # USD
    realized_pnl: float     # USD


def fetch_positions(client: KalshiMarketClient) -> list[PositionRow]:
    raw_positions = client.get_positions()
    rows = []

    for p in raw_positions:
        contracts = p["yes_contracts"]
        if contracts == 0:
            continue

        side = "yes" if contracts > 0 else "no"
        abs_contracts = abs(contracts)

        # entry cost from market_exposure_dollars if available
        raw = p.get("raw")
        entry_cost = float(getattr(raw, "market_exposure_dollars", None) or 0)

        # fetch current price
        current_mid = None
        unrealized = None
        try:
            snap = client.get_market(p["ticker"])
            if snap.yes_bid is not None and snap.yes_ask is not None:
                current_mid = (snap.yes_bid + snap.yes_ask) // 2
                # unrealized pnl: (current_mid - entry_price_per_contract) * contracts
                if entry_cost > 0 and abs_contracts > 0:
                    entry_price_cents = (entry_cost / abs_contracts) * 100
                    unrealized = (current_mid - entry_price_cents) * abs_contracts / 100
        except Exception:
            pass

        rows.append(PositionRow(
            ticker=p["ticker"],
            side=side,
            contracts=abs_contracts,
            entry_cost_dollars=entry_cost,
            current_mid=current_mid,
            unrealized_pnl=unrealized,
            realized_pnl=p["realized_pnl"],
        ))

    return rows


def display_positions(rows: list[PositionRow]) -> None:
    if not rows:
        print("  (no open positions)")
        return

    header = (
        f"{'TICKER':<35} {'SIDE':<5} {'CTRS':>5} "
        f"{'ENTRY$':>8} {'MID¢':>6} {'UNREAL$':>9} {'REAL$':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        mid_str = f"{r.current_mid}" if r.current_mid is not None else "  N/A"
        unreal_str = f"{r.unrealized_pnl:+.2f}" if r.unrealized_pnl is not None else "   N/A"
        print(
            f"{r.ticker:<35} {r.side:<5} {r.contracts:>5.1f} "
            f"{r.entry_cost_dollars:>8.2f} {mid_str:>6} {unreal_str:>9} {r.realized_pnl:>8.2f}"
        )

    print(sep)
    total_unreal = sum(r.unrealized_pnl for r in rows if r.unrealized_pnl is not None)
    total_real = sum(r.realized_pnl for r in rows)
    print(f"{'TOTAL':<35} {'':5} {'':5} {'':8} {'':6} {total_unreal:>+9.2f} {total_real:>8.2f}")


def run_monitor(
    client: KalshiMarketClient,
    interval_sec: int = 30,
    dry_run: bool = True,
) -> None:
    mode = "[DRY RUN] " if dry_run else ""
    print(f"{mode}Position monitor — refreshing every {interval_sec}s  (Ctrl+C to stop)\n")
    try:
        while True:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"\n=== {ts} ===")
            rows = fetch_positions(client)
            display_positions(rows)
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    from clients.kalshi import KalshiMarketClient
    run_monitor(KalshiMarketClient())
