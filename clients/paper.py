"""
Paper trading client — wraps KalshiMarketClient for real market data
but simulates balance, order fills, and positions in memory.

Usage:
    from clients.kalshi import KalshiMarketClient
    from clients.paper import PaperKalshiClient

    client = PaperKalshiClient(KalshiMarketClient(), starting_balance=1000.0)

    # Use exactly like KalshiMarketClient everywhere else
    # Orders print [PAPER] and deduct from the fake balance — nothing hits the API.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from clients.base import MarketClient, MarketSnapshot, OrderResult
from clients.kalshi import KalshiMarketClient


@dataclass
class _PaperPosition:
    ticker: str
    yes_contracts: float = 0.0
    no_contracts: float = 0.0
    total_cost: float = 0.0     # dollars paid
    realized_pnl: float = 0.0


class PaperKalshiClient(MarketClient):
    """
    Drop-in replacement for KalshiMarketClient in paper-trading mode.

    - get_market / get_markets / get_balance → real Kalshi API
    - place_order → simulated (deducts from paper balance, records position)
    - get_positions → returns paper positions with current market prices
    """

    platform = "paper_kalshi"

    def __init__(
        self,
        real_client: KalshiMarketClient,
        starting_balance: float = 1000.0,
    ):
        self._real = real_client
        self._balance = starting_balance
        self._starting_balance = starting_balance
        self._positions: dict[str, _PaperPosition] = {}
        self._order_count = 0

    # ------------------------------------------------------------------ #
    #  Delegated reads — real market data                                  #
    # ------------------------------------------------------------------ #

    def get_balance(self) -> float:
        return self._balance

    def get_market(self, ticker: str) -> MarketSnapshot:
        snap = self._real.get_market(ticker)
        snap.platform = self.platform
        return snap

    def get_markets(self, status: str = "open", limit: int = 1000) -> list[MarketSnapshot]:
        snaps = self._real.get_markets(status=status, limit=limit)
        for s in snaps:
            s.platform = self.platform
        return snaps

    # ------------------------------------------------------------------ #
    #  Simulated writes                                                    #
    # ------------------------------------------------------------------ #

    def place_order(
        self,
        ticker: str,
        side: str,
        price_cents: int,
        contracts: int,
        dry_run: bool = True,
    ) -> OrderResult:
        cost = contracts * price_cents / 100

        if cost > self._balance:
            print(f"[PAPER] REJECTED {ticker} — insufficient paper balance "
                  f"(need ${cost:.2f}, have ${self._balance:.2f})")
            return OrderResult(
                order_id="PAPER-REJECTED",
                ticker=ticker, side=side, price=price_cents,
                contracts=contracts, dry_run=True, platform=self.platform,
            )

        self._balance -= cost
        self._order_count += 1
        order_id = f"PAPER-{self._order_count:04d}"

        pos = self._positions.setdefault(ticker, _PaperPosition(ticker=ticker))
        if side == "yes":
            pos.yes_contracts += contracts
        else:
            pos.no_contracts += contracts
        pos.total_cost += cost

        pnl_change = self._starting_balance - self._balance
        print(
            f"[PAPER] {order_id}  {ticker} {side.upper()} {contracts}x @ {price_cents}¢  "
            f"cost=${cost:.2f}  paper_balance=${self._balance:.2f}  "
            f"total_deployed=${pnl_change:.2f}"
        )
        return OrderResult(
            order_id=order_id,
            ticker=ticker, side=side, price=price_cents,
            contracts=contracts, dry_run=False, platform=self.platform,
        )

    # ------------------------------------------------------------------ #
    #  Positions — paper positions with live unrealized PnL               #
    # ------------------------------------------------------------------ #

    def get_positions(self) -> list[dict]:
        rows = []
        for ticker, pos in self._positions.items():
            net = pos.yes_contracts - pos.no_contracts
            if net == 0:
                continue

            # Fetch current price to compute unrealized PnL
            exposure = pos.total_cost
            try:
                snap = self._real.get_market(ticker)
                if snap.yes_bid is not None and snap.yes_ask is not None:
                    mid_cents = (snap.yes_bid + snap.yes_ask) / 2
                    current_value = abs(net) * mid_cents / 100
                    pos.realized_pnl = current_value - exposure
            except Exception:
                pass

            rows.append({
                "ticker": ticker,
                "yes_contracts": float(net),
                "realized_pnl": pos.realized_pnl,
                "market_exposure_dollars": str(exposure),
                "platform": self.platform,
                "raw": None,
            })
        return rows

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        positions = self.get_positions()
        total_unrealized = sum(p["realized_pnl"] for p in positions)
        deployed = self._starting_balance - self._balance
        print(f"\n{'='*50}")
        print(f"  PAPER TRADING SUMMARY")
        print(f"{'='*50}")
        print(f"  Starting balance : ${self._starting_balance:>10.2f}")
        print(f"  Cash remaining   : ${self._balance:>10.2f}")
        print(f"  Deployed         : ${deployed:>10.2f}")
        print(f"  Unrealized PnL   : ${total_unrealized:>+10.2f}")
        print(f"  Open positions   : {len(positions)}")
        print(f"  Total orders     : {self._order_count}")
        print(f"{'='*50}\n")
        for p in positions:
            print(f"  {p['ticker']:<35}  {p['yes_contracts']:+.0f} contracts  "
                  f"pnl=${p['realized_pnl']:+.2f}")
        if positions:
            print()
