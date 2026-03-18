from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSnapshot:
    ticker: str
    title: str
    yes_bid: Optional[int]   # cents (0–100)
    yes_ask: Optional[int]
    last_price: Optional[int]
    volume: int
    open_interest: int
    platform: str            # "kalshi" | "polymarket"
    raw: object = None       # original SDK object


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    side: str                # "yes" | "no"
    price: int               # cents
    contracts: int
    dry_run: bool
    platform: str


class MarketClient(ABC):
    """Minimal interface every platform client must implement."""

    @abstractmethod
    def get_balance(self) -> float:
        """Return available balance in USD."""

    @abstractmethod
    def get_market(self, ticker: str) -> MarketSnapshot:
        """Return current snapshot for a single market."""

    @abstractmethod
    def get_markets(self, status: str = "open", limit: int = 1000) -> list[MarketSnapshot]:
        """Return a page of markets filtered by status."""

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        side: str,          # "yes" | "no"
        price_cents: int,
        contracts: int,
        dry_run: bool = True,
    ) -> OrderResult:
        """Place a limit order. dry_run=True logs but never hits the API."""

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Return open positions."""
