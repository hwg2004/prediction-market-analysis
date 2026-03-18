from dataclasses import dataclass, field
from datetime import date
from typing import Optional


def kelly_size(
    p_win: float,
    b_win: float,
    bankroll: float,
    max_fraction: float = 0.10,
) -> int:
    """
    Full Kelly criterion, capped at max_fraction of bankroll.

    p_win  — estimated probability of winning (0–1)
    b_win  — net odds per $1 risked (for a binary market at price p¢:
             b_win = (100 - p) / p)
    bankroll — USD available

    Returns number of contracts (integer ≥ 0).
    Each contract is worth $1 at resolution; the caller applies the price
    to convert contracts → dollars at risk.
    """
    if p_win <= 0 or b_win <= 0 or bankroll <= 0:
        return 0
    kelly_fraction = p_win - (1 - p_win) / b_win
    kelly_fraction = max(0.0, min(kelly_fraction, max_fraction))
    dollars = kelly_fraction * bankroll
    return max(0, int(dollars))


@dataclass
class RiskManager:
    max_single_position_pct: float = 0.10   # max single trade as fraction of balance
    daily_loss_limit_pct: float = 0.20       # pause trading if daily loss exceeds this
    max_total_exposure_pct: float = 0.50     # max sum of all open position costs

    _daily_start_balance: float = field(default=0.0, init=False, repr=False)
    _daily_realized_loss: float = field(default=0.0, init=False, repr=False)
    _tracking_date: Optional[date] = field(default=None, init=False, repr=False)

    def reset_daily_if_needed(self, balance: float) -> None:
        today = date.today()
        if self._tracking_date != today:
            self._tracking_date = today
            self._daily_start_balance = balance
            self._daily_realized_loss = 0.0

    def record_pnl(self, delta_usd: float) -> None:
        """Call with a negative value to record a loss."""
        if delta_usd < 0:
            self._daily_realized_loss += abs(delta_usd)

    def _daily_loss_exceeded(self, balance: float) -> bool:
        if self._daily_start_balance <= 0:
            return False
        return self._daily_realized_loss >= self.daily_loss_limit_pct * self._daily_start_balance

    def check(
        self,
        ticker: str,
        contracts: int,
        price_cents: int,
        current_positions: list[dict],
        balance: float,
    ) -> tuple[bool, str]:
        """
        Returns (approved, reason).
        current_positions: list of dicts from KalshiMarketClient.get_positions(),
            each with 'yes_contracts' (float) and optionally 'market_exposure_dollars' (str).
        """
        if balance <= 0:
            return False, "zero balance"

        trade_cost = contracts * price_cents / 100

        # 1. Single position size
        if trade_cost > self.max_single_position_pct * balance:
            return False, (
                f"single position ${trade_cost:.2f} exceeds "
                f"{self.max_single_position_pct*100:.0f}% of balance (${balance:.2f})"
            )

        # 2. Total exposure
        existing_exposure = sum(
            float(p.get("market_exposure_dollars") or 0)
            for p in current_positions
        )
        if existing_exposure + trade_cost > self.max_total_exposure_pct * balance:
            return False, (
                f"total exposure ${existing_exposure + trade_cost:.2f} would exceed "
                f"{self.max_total_exposure_pct*100:.0f}% of balance (${balance:.2f})"
            )

        # 3. Daily loss limit
        if self._daily_loss_exceeded(balance):
            return False, (
                f"daily loss limit reached "
                f"(lost ${self._daily_realized_loss:.2f} of "
                f"${self._daily_start_balance:.2f})"
            )

        return True, "approved"
