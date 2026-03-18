import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clients.base import MarketClient, OrderResult
from risk import RiskManager

LOG_PATH = Path(__file__).parent.parent / "data" / "orders_log.csv"
LOG_COLUMNS = [
    "timestamp", "ticker", "side", "contracts", "price_cents",
    "order_id", "dry_run", "platform", "status", "notes",
]

logger = logging.getLogger(__name__)


@dataclass
class OrderManager:
    client: MarketClient
    risk: RiskManager
    max_spend_usd: float = 50.0
    dry_run: bool = True

    def _log_order(
        self,
        result: OrderResult,
        status: str,
        notes: str = "",
    ) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": result.ticker,
                "side": result.side,
                "contracts": result.contracts,
                "price_cents": result.price,
                "order_id": result.order_id,
                "dry_run": result.dry_run,
                "platform": result.platform,
                "status": status,
                "notes": notes,
            })

    def _open_positions(self) -> dict[str, str]:
        """Return {ticker: side} for all currently open positions."""
        try:
            return {
                p["ticker"]: ("yes" if p["yes_contracts"] > 0 else "no")
                for p in self.client.get_positions()
                if p["yes_contracts"] != 0
            }
        except Exception:
            return {}

    def place(
        self,
        ticker: str,
        side: str,
        price_cents: int,
        contracts: int,
        notes: str = "",
    ) -> Optional[OrderResult]:
        _rejected = OrderResult(
            order_id="REJECTED",
            ticker=ticker,
            side=side,
            price=price_cents,
            contracts=contracts,
            dry_run=self.dry_run,
            platform=getattr(self.client, "platform", "unknown"),
        )

        trade_cost = contracts * price_cents / 100

        # 1. Balance check
        try:
            balance = self.client.get_balance()
        except Exception as e:
            reason = f"balance fetch failed: {e}"
            logger.warning(f"[ORDER REJECTED] {ticker} — {reason}")
            self._log_order(_rejected, "rejected", reason)
            return None

        if trade_cost > balance:
            reason = f"insufficient balance (need ${trade_cost:.2f}, have ${balance:.2f})"
            logger.warning(f"[ORDER REJECTED] {ticker} — {reason}")
            self._log_order(_rejected, "rejected", reason)
            return None

        # 2. Max spend check
        if trade_cost > self.max_spend_usd:
            reason = f"exceeds max_spend_usd ${self.max_spend_usd:.2f} (trade=${trade_cost:.2f})"
            logger.warning(f"[ORDER REJECTED] {ticker} — {reason}")
            self._log_order(_rejected, "rejected", reason)
            return None

        # 3. Deduplication
        open_pos = self._open_positions()
        if open_pos.get(ticker) == side:
            reason = f"duplicate: already have open {side.upper()} position on {ticker}"
            logger.info(f"[ORDER SKIPPED] {ticker} — {reason}")
            self._log_order(_rejected, "skipped", reason)
            return None

        # 4. Risk check
        current_positions = self.client.get_positions()
        approved, risk_reason = self.risk.check(
            ticker, contracts, price_cents, current_positions, balance
        )
        if not approved:
            logger.warning(f"[ORDER REJECTED] {ticker} — risk: {risk_reason}")
            self._log_order(_rejected, "rejected", f"risk: {risk_reason}")
            return None

        # 5. Place order
        result = self.client.place_order(
            ticker=ticker,
            side=side,
            price_cents=price_cents,
            contracts=contracts,
            dry_run=self.dry_run,
        )

        status = "dry_run" if self.dry_run else "placed"
        self._log_order(result, status, notes)
        return result
