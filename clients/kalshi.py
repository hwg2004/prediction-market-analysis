import os
from dotenv import load_dotenv
from kalshi_python_sync import Configuration, KalshiClient

from clients.base import MarketClient, MarketSnapshot, OrderResult

load_dotenv()


def _build_client() -> KalshiClient:
    config = Configuration(host="https://api.elections.kalshi.com/trade-api/v2")
    with open(os.getenv("KALSHI_KEY_PATH", "kalshi.pem")) as f:
        config.private_key_pem = f.read()
    config.api_key_id = os.getenv("KALSHI_KEY_ID")
    return KalshiClient(config)


class KalshiMarketClient(MarketClient):
    def __init__(self):
        self._client = _build_client()

    # ------------------------------------------------------------------ #
    #  Account                                                             #
    # ------------------------------------------------------------------ #

    def get_balance(self) -> float:
        return self._client.get_balance().balance / 100

    # ------------------------------------------------------------------ #
    #  Markets                                                             #
    # ------------------------------------------------------------------ #

    def get_market(self, ticker: str) -> MarketSnapshot:
        m = self._client.get_market(ticker).market
        return self._to_snapshot(m)

    def get_markets(self, status: str = "open", limit: int = 1000) -> list[MarketSnapshot]:
        snapshots, cursor = [], None
        while True:
            resp = self._client.get_markets(limit=min(limit, 1000), cursor=cursor, status=status)
            snapshots.extend(self._to_snapshot(m) for m in resp.markets)
            cursor = resp.cursor
            if cursor is None or len(snapshots) >= limit:
                break
        return snapshots[:limit]

    # ------------------------------------------------------------------ #
    #  Trading                                                             #
    # ------------------------------------------------------------------ #

    def place_order(
        self,
        ticker: str,
        side: str,
        price_cents: int,
        contracts: int,
        dry_run: bool = True,
    ) -> OrderResult:
        result = OrderResult(
            order_id="DRY_RUN",
            ticker=ticker,
            side=side,
            price=price_cents,
            contracts=contracts,
            dry_run=dry_run,
            platform="kalshi",
        )
        if dry_run:
            print(f"[DRY RUN] {ticker} {side.upper()} {contracts}x @ {price_cents}¢")
            return result

        from kalshi_python_sync import CreateOrderRequest
        req = CreateOrderRequest(
            ticker=ticker,
            side=side,
            action="buy",
            type="limit",
            yes_price=price_cents if side == "yes" else 100 - price_cents,
            count=contracts,
        )
        resp = self._client.create_order(req)
        result.order_id = resp.order.order_id
        print(f"[ORDER] {ticker} {side.upper()} {contracts}x @ {price_cents}¢  id={result.order_id}")
        return result

    # ------------------------------------------------------------------ #
    #  Positions                                                           #
    # ------------------------------------------------------------------ #

    def get_positions(self) -> list[dict]:
        positions = self._client.get_positions().market_positions
        return [
            {
                "ticker": p.ticker,
                "yes_contracts": float(getattr(p, "position_fp", None) or 0),
                "realized_pnl": float(getattr(p, "realized_pnl_dollars", None) or 0),
                "platform": "kalshi",
                "raw": p,
            }
            for p in positions
        ]

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_snapshot(m) -> MarketSnapshot:
        return MarketSnapshot(
            ticker=m.ticker,
            title=getattr(m, "title", ""),
            yes_bid=getattr(m, "yes_bid", None),
            yes_ask=getattr(m, "yes_ask", None),
            last_price=getattr(m, "last_price", None),
            volume=getattr(m, "volume", 0) or 0,
            open_interest=getattr(m, "open_interest", 0) or 0,
            platform="kalshi",
            raw=m,
        )

    # Expose the raw SDK client for advanced use (e.g. HistoricalApi)
    @property
    def raw(self) -> KalshiClient:
        return self._client
