import logging
from dataclasses import dataclass
from typing import Optional

from clients.base import MarketClient, MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    ticker: str
    platform_a: str
    platform_b: str         # same as platform_a for single-platform detection
    side_a: str             # "yes"
    side_b: str             # "no"
    price_a: int            # yes_ask in cents (cost to buy YES)
    price_b: int            # no_ask in cents  (cost to buy NO = 100 - yes_bid)
    edge_cents: float       # 100 - price_a - price_b  (positive = free money)
    estimated_contracts: int = 1


class ArbitrageScanner:
    """
    Detects markets where buying both YES and NO costs less than 100¢ combined,
    locking in a risk-free profit.

    Binary identity:
        no_bid  = 100 - yes_ask
        no_ask  = 100 - yes_bid

    Arb condition:
        yes_ask + no_ask < 100
        yes_ask + (100 - yes_bid) < 100
        yes_bid > yes_ask            ← crossed spread

    Edge = yes_bid - yes_ask  (cents per contract pair)

    Cross-platform extension: when two clients are registered, scan for the same
    event priced differently across platforms.
    """

    def __init__(self, min_edge_cents: float = 3.0):
        self.min_edge_cents = min_edge_cents
        self._extra_clients: list[MarketClient] = []

    def add_platform(self, client: MarketClient) -> None:
        """Register a second platform client for future cross-platform scanning."""
        self._extra_clients.append(client)

    def _scan_single_platform(
        self, snapshots: list[MarketSnapshot]
    ) -> list[ArbitrageOpportunity]:
        opps = []
        for s in snapshots:
            if s.yes_bid is None or s.yes_ask is None:
                continue
            edge = s.yes_bid - s.yes_ask
            if edge >= self.min_edge_cents:
                no_ask = 100 - s.yes_bid
                opps.append(ArbitrageOpportunity(
                    ticker=s.ticker,
                    platform_a=s.platform,
                    platform_b=s.platform,
                    side_a="yes",
                    side_b="no",
                    price_a=s.yes_ask,
                    price_b=no_ask,
                    edge_cents=float(edge),
                    estimated_contracts=1,
                ))
        return opps

    def _scan_cross_platform(
        self,
        snapshots_a: list[MarketSnapshot],
        snapshots_b: list[MarketSnapshot],
    ) -> list[ArbitrageOpportunity]:
        """
        Match markets across platforms by title similarity (exact title match for now).
        Returns arb opportunities where the same event is mispriced.
        """
        logger.info("Cross-platform scan pending — market matching not yet implemented")
        # Future: fuzzy-match titles, then check yes_ask_a + no_ask_b < 100 - min_edge
        return []

    def scan(self, snapshots: list[MarketSnapshot]) -> list[ArbitrageOpportunity]:
        opps = self._scan_single_platform(snapshots)

        if self._extra_clients:
            logger.info(
                f"Cross-platform scan pending — "
                f"{len(self._extra_clients)} additional platform(s) registered"
            )

        return sorted(opps, key=lambda o: o.edge_cents, reverse=True)
