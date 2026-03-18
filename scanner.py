"""
Main scanner — orchestrates all three strategies and surfaces trading signals.

Usage:
    from clients.kalshi import KalshiMarketClient
    from risk import RiskManager
    from execution.orders import OrderManager
    from scanner import Scanner

    client = KalshiMarketClient()
    risk   = RiskManager()
    orders = OrderManager(client=client, risk=risk, dry_run=True)
    s = Scanner(client=client, risk=risk, order_manager=orders)

    # One-shot
    signals = s.run_once()
    s.display_signals(signals)

    # Continuous loop (dry run, no auto-place)
    s.run_loop(interval_sec=60)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from clients.base import MarketClient, MarketSnapshot
from risk import RiskManager, kelly_size

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    ticker: str
    title: str
    strategy: str           # "hmm" | "news" | "arbitrage" | "copycat" | "merged"
    direction: str          # "yes" | "no"
    confidence: float       # 0–1
    suggested_price: int    # cents
    suggested_contracts: int


class Scanner:
    def __init__(
        self,
        client: MarketClient,
        risk: RiskManager,
        order_manager=None,
        use_hmm: bool = True,
        use_news: bool = True,
        use_arbitrage: bool = True,
        use_copycat: bool = True,
        min_confidence: float = 0.55,
        market_limit: int = 200,
    ):
        self._client = client
        self._risk = risk
        self._order_manager = order_manager
        self.min_confidence = min_confidence
        self.market_limit = market_limit

        self._hmm = None
        self._news_fetcher = None
        self._arb_scanner = None
        self._copycat = None
        self._init_strategies(use_hmm, use_news, use_arbitrage, use_copycat)

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def _init_strategies(
        self, use_hmm: bool, use_news: bool, use_arbitrage: bool, use_copycat: bool
    ) -> None:
        if use_hmm:
            try:
                from strategies.hmm import MarketHMM
                self._hmm = MarketHMM()
                self._hmm.fit()
            except Exception as e:
                logger.warning(f"HMM strategy disabled: {e}")

        if use_news:
            try:
                from strategies.news import NewsFetcher
                self._news_fetcher = NewsFetcher()
            except Exception as e:
                logger.warning(f"News strategy disabled: {e}")

        if use_arbitrage:
            try:
                from strategies.arbitrage import ArbitrageScanner
                self._arb_scanner = ArbitrageScanner()
            except Exception as e:
                logger.warning(f"Arbitrage strategy disabled: {e}")

        if use_copycat:
            try:
                from strategies.copycat import CopycatStrategy
                self._copycat = CopycatStrategy()
            except Exception as e:
                logger.warning(f"Copycat strategy disabled: {e}")

    # ------------------------------------------------------------------ #
    #  Per-market signal runners                                           #
    # ------------------------------------------------------------------ #

    def _run_hmm(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if self._hmm is None:
            return None
        try:
            from data.history import fetch_live_candlesticks, _build_api_client
            _, market_api = _build_api_client()
            df = fetch_live_candlesticks(market_api, snapshot.ticker)
        except Exception:
            df = None

        if df is None or df.empty:
            # fall back to a stub DataFrame from last_price
            import pandas as pd
            if snapshot.yes_bid is not None and snapshot.yes_ask is not None:
                df = pd.DataFrame([{
                    "yes_bid_close": snapshot.yes_bid,
                    "yes_ask_close": snapshot.yes_ask,
                    "volume": snapshot.volume,
                }])
            else:
                return None

        result = self._hmm.predict(df)
        fair_value = result["fair_value"]
        confidence = result["confidence"]

        if snapshot.yes_ask is None or snapshot.yes_bid is None:
            return None

        current_mid = (snapshot.yes_bid + snapshot.yes_ask) / 2
        if abs(fair_value - current_mid) < 3:   # less than 3¢ edge, skip
            return None

        direction = "yes" if fair_value > current_mid else "no"
        price = snapshot.yes_ask if direction == "yes" else 100 - snapshot.yes_bid

        return Signal(
            ticker=snapshot.ticker,
            title=snapshot.title,
            strategy="hmm",
            direction=direction,
            confidence=confidence,
            suggested_price=price,
            suggested_contracts=1,
        )

    def _run_news(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if self._news_fetcher is None:
            return None
        try:
            articles = self._news_fetcher.fetch_for_market(snapshot.ticker, snapshot.title)
            sentiment = self._news_fetcher.aggregate_sentiment(articles)
        except Exception as e:
            logger.debug(f"News error for {snapshot.ticker}: {e}")
            return None

        if abs(sentiment) < 0.10:
            return None

        direction = "yes" if sentiment > 0 else "no"
        confidence = min(abs(sentiment), 1.0)

        if snapshot.yes_ask is None or snapshot.yes_bid is None:
            return None

        price = snapshot.yes_ask if direction == "yes" else 100 - snapshot.yes_bid
        return Signal(
            ticker=snapshot.ticker,
            title=snapshot.title,
            strategy="news",
            direction=direction,
            confidence=confidence,
            suggested_price=price,
            suggested_contracts=1,
        )

    def _run_copycat(self, snapshots: list[MarketSnapshot], balance: float) -> list[Signal]:
        if self._copycat is None:
            return []
        try:
            return self._copycat.scan(my_balance=balance, kalshi_snapshots=snapshots)
        except Exception as e:
            logger.debug(f"Copycat scan error: {e}")
            return []

    def _run_arbitrage(self, snapshots: list[MarketSnapshot]) -> list[Signal]:
        if self._arb_scanner is None:
            return []
        try:
            opps = self._arb_scanner.scan(snapshots)
        except Exception as e:
            logger.debug(f"Arbitrage scan error: {e}")
            return []

        signals = []
        for opp in opps:
            confidence = min(opp.edge_cents / 10.0, 1.0)
            signals.append(Signal(
                ticker=opp.ticker,
                title=opp.ticker,
                strategy="arbitrage",
                direction=opp.side_a,
                confidence=confidence,
                suggested_price=opp.price_a,
                suggested_contracts=opp.estimated_contracts,
            ))
        return signals

    # ------------------------------------------------------------------ #
    #  Signal merging                                                      #
    # ------------------------------------------------------------------ #

    def _merge_signals(self, signals: list[Signal]) -> list[Signal]:
        """
        Group by ticker. For each ticker with multiple signals:
          - Require majority direction agreement
          - Merged confidence = mean of agreeing signals
          - Return merged Signal with strategy="merged"
        Single-strategy signals pass through as-is (not merged).
        """
        from collections import defaultdict
        by_ticker: dict[str, list[Signal]] = defaultdict(list)
        for s in signals:
            by_ticker[s.ticker].append(s)

        merged = []
        for ticker, group in by_ticker.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            yes_signals = [s for s in group if s.direction == "yes"]
            no_signals  = [s for s in group if s.direction == "no"]

            if len(yes_signals) > len(no_signals):
                agreeing = yes_signals
                direction = "yes"
            elif len(no_signals) > len(yes_signals):
                agreeing = no_signals
                direction = "no"
            else:
                # Tie → no signal (conservative)
                logger.debug(f"{ticker}: strategy disagreement, skipping")
                continue

            avg_confidence = sum(s.confidence for s in agreeing) / len(agreeing)
            avg_price = int(sum(s.suggested_price for s in agreeing) / len(agreeing))
            merged.append(Signal(
                ticker=ticker,
                title=group[0].title,
                strategy="merged",
                direction=direction,
                confidence=avg_confidence,
                suggested_price=avg_price,
                suggested_contracts=agreeing[0].suggested_contracts,
            ))

        return sorted(merged, key=lambda s: s.confidence, reverse=True)

    # ------------------------------------------------------------------ #
    #  Sizing                                                              #
    # ------------------------------------------------------------------ #

    def _size_signal(self, signal: Signal, balance: float) -> Signal:
        if signal.suggested_price <= 0 or balance <= 0:
            return signal
        p_win = signal.confidence
        b_win = (100 - signal.suggested_price) / signal.suggested_price
        contracts = kelly_size(p_win, b_win, balance)
        signal.suggested_contracts = max(1, contracts)
        return signal

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run_once(self) -> list[Signal]:
        snapshots = self._client.get_markets(status="open", limit=self.market_limit)

        try:
            balance = self._client.get_balance()
        except Exception:
            balance = 0.0

        all_signals: list[Signal] = []

        for snap in snapshots:
            for runner in (self._run_hmm, self._run_news):
                sig = runner(snap)
                if sig is not None:
                    all_signals.append(sig)

        all_signals.extend(self._run_arbitrage(snapshots))
        all_signals.extend(self._run_copycat(snapshots, balance))

        merged = self._merge_signals(all_signals)
        sized = [self._size_signal(s, balance) for s in merged]
        return [s for s in sized if s.confidence >= self.min_confidence]

    def display_signals(self, signals: list[Signal]) -> None:
        if not signals:
            print("  (no signals above confidence threshold)")
            return

        header = (
            f"{'TICKER':<35} {'DIR':<5} {'STRAT':<12} "
            f"{'CONF':>6} {'PRICE¢':>7} {'CTRS':>5}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for s in signals:
            print(
                f"{s.ticker:<35} {s.direction:<5} {s.strategy:<12} "
                f"{s.confidence:>6.2f} {s.suggested_price:>7} {s.suggested_contracts:>5}"
            )
        print(sep)

    def _collect_candles(self, snapshots: list[MarketSnapshot]) -> None:
        """
        Passively collect live candlestick data for open markets so the HMM
        accumulates training data over time. Runs once per loop iteration,
        samples up to 20 markets per call to keep latency low.
        Saves to data/candlesticks/<ticker>.csv (appending new rows).
        """
        try:
            from data.history import fetch_live_candlesticks, _build_api_client
            from pathlib import Path
            _, market_api = _build_api_client()
            candles_dir = Path("data/candlesticks")
            candles_dir.mkdir(exist_ok=True)

            # Prioritise markets with most volume — best training signal
            liquid = sorted(
                [s for s in snapshots if s.volume > 50],
                key=lambda s: s.volume, reverse=True
            )[:20]

            for snap in liquid:
                out_path = candles_dir / f"{snap.ticker}.csv"
                df = fetch_live_candlesticks(market_api, snap.ticker, lookback_hours=2)
                if df.empty:
                    continue
                if out_path.exists():
                    import pandas as pd
                    existing = pd.read_csv(out_path)
                    combined = pd.concat([existing, df]).drop_duplicates(subset=["ts"])
                    combined.to_csv(out_path, index=False)
                else:
                    df.to_csv(out_path, index=False)
        except Exception as e:
            logger.debug(f"Candle collection error: {e}")

    def _maybe_retrain_hmm(self) -> None:
        """Re-fit the HMM once enough new data has accumulated (every 50 loop iterations)."""
        if self._hmm is None:
            return
        if not hasattr(self, "_loop_count"):
            self._loop_count = 0
        self._loop_count += 1
        if self._loop_count % 50 == 0:
            logger.info("Re-fitting HMM on accumulated candlestick data...")
            self._hmm.fit()

    def run_loop(
        self,
        interval_sec: int = 60,
        auto_place: bool = False,
    ) -> None:
        mode = "AUTO-PLACE" if auto_place else "DRY RUN"
        print(f"Scanner starting [{mode}] — interval {interval_sec}s  (Ctrl+C to stop)\n")

        try:
            while True:
                from datetime import datetime, timezone
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                try:
                    balance = self._client.get_balance()
                    self._risk.reset_daily_if_needed(balance)
                except Exception:
                    balance = 0.0

                print(f"\n=== {ts}  balance=${balance:.2f} ===")
                signals = self.run_once()
                self.display_signals(signals)

                if auto_place and self._order_manager is not None:
                    for sig in signals:
                        self._order_manager.place(
                            ticker=sig.ticker,
                            side=sig.direction,
                            price_cents=sig.suggested_price,
                            contracts=sig.suggested_contracts,
                            notes=f"strategy={sig.strategy} conf={sig.confidence:.2f}",
                        )

                # Passively collect candlestick data for HMM training
                snapshots = self._client.get_markets(status="open", limit=self.market_limit)
                self._collect_candles(snapshots)
                self._maybe_retrain_hmm()

                time.sleep(interval_sec)

        except KeyboardInterrupt:
            # Print paper trading summary on exit if using PaperKalshiClient
            print("\nScanner stopped.")
            if hasattr(self._client, "print_summary"):
                self._client.print_summary()


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prediction market scanner")
    parser.add_argument("--paper", action="store_true",
                        help="Run in paper trading mode with a fake $1,000 balance")
    parser.add_argument("--paper-balance", type=float, default=1000.0,
                        help="Starting paper balance in USD (default: 1000)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Scan interval in seconds (default: 60)")
    args = parser.parse_args()

    from clients.kalshi import KalshiMarketClient
    from risk import RiskManager
    from execution.orders import OrderManager

    real_client = KalshiMarketClient()

    if args.paper:
        from clients.paper import PaperKalshiClient
        client = PaperKalshiClient(real_client, starting_balance=args.paper_balance)
        risk = RiskManager(max_single_position_pct=0.05)
        orders = OrderManager(client=client, risk=risk, dry_run=False,
                              max_spend_usd=args.paper_balance * 0.1)
        print(f"[PAPER MODE] Starting with ${args.paper_balance:.2f} fake balance\n")
    else:
        client = real_client
        risk = RiskManager(max_single_position_pct=0.03)
        orders = OrderManager(client=client, risk=risk, dry_run=False, max_spend_usd=10.0)

    scanner = Scanner(client=client, risk=risk, order_manager=orders)
    scanner.run_loop(interval_sec=args.interval, auto_place=True)
