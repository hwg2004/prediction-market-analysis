"""
Copycat strategy — follows specific Polymarket wallets and mirrors their trades on Kalshi.

How it works:
  1. Poll Polymarket CLOB API for recent BUY trades from tracked wallets
  2. Fetch the market question for each trade (Polymarket Gamma API)
  3. Match the question against open Kalshi markets using TF-IDF cosine similarity
  4. Size proportionally: (my_balance / their_estimated_portfolio) * their_trade_usd
  5. Emit a Signal for the matched Kalshi market

Matching approach (MarketMatcher):
  - TF-IDF vectorizer (1+2-grams, sublinear TF) fitted on all Kalshi market titles
  - Cosine similarity between Polymarket question and Kalshi title vectors
  - Index is rebuilt automatically when the Kalshi market universe changes
  - Much better than word overlap: IDF down-weights terms common across all markets
    (e.g. "will", "price", "above") and cosine handles length differences cleanly
  - Optional upgrade: install sentence-transformers for semantic embeddings
    (pip install sentence-transformers) — MarketMatcher will use it automatically

Configuration (.env):
  TRACKED_WALLETS=0xabc123,0xdef456,...

Notes:
  - Only BUY orders are copied (no shorting)
  - Trades already seen this session are deduplicated by trade ID
  - Market question cache TTL: 1hr  (questions rarely change)
  - Trade fetch cache TTL: 30s     (fresh enough for signal latency)
  - Portfolio estimate: rolling 7-day sum of that wallet's trade USD values
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logger = logging.getLogger(__name__)

_POLY_CLOB  = "https://clob.polymarket.com"
_POLY_GAMMA = "https://gamma-api.polymarket.com"
_SEVEN_DAYS = 7 * 86400
_SESSION    = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})

# Token normalisation — applied to both query and Kalshi titles before vectorising.
# Expands abbreviations so TF-IDF can align them. Sentence-transformers does this
# automatically (it understands synonyms semantically), but until that's installed
# this covers the most common prediction market assets.
_NORM: dict[str, str] = {
    # Crypto
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana", "xrp": "ripple",
    "bnb": "binance", "doge": "dogecoin", "ada": "cardano", "avax": "avalanche",
    "matic": "polygon", "dot": "polkadot", "link": "chainlink", "ltc": "litecoin",
    # Number shorthand
    "80k": "80000", "90k": "90000", "100k": "100000", "50k": "50000",
    "1m": "1000000", "1b": "1000000000",
    # Indices / finance
    "sp500": "s&p 500", "spx": "s&p 500", "nasdaq": "nasdaq", "djia": "dow jones",
    "fed": "federal reserve", "fomc": "federal reserve", "gdp": "gross domestic product",
    "cpi": "consumer price index", "pce": "personal consumption expenditures",
    # Politics
    "potus": "president", "gop": "republican", "dem": "democrat",
    "scotus": "supreme court",
    # Sports shorthand
    "nfl": "football", "nba": "basketball", "mlb": "baseball", "nhl": "hockey",
    "epl": "premier league", "ucl": "champions league",
}

import re as _re

def _normalise(text: str) -> str:
    """Lower-case, remove thousand-separator commas, expand abbreviations."""
    text = text.lower()
    text = _re.sub(r"(\d),(\d)", r"\1\2", text)  # 80,000 → 80000 (keep digits together)
    text = _re.sub(r"[$%]", " ", text)             # drop currency/percent symbols
    tokens = _re.findall(r"[a-z0-9&]+", text)
    tokens = [_NORM.get(t, t) for t in tokens]
    return " ".join(tokens)


# ------------------------------------------------------------------ #
#  Data classes                                                        #
# ------------------------------------------------------------------ #

@dataclass
class PolyTrade:
    trade_id: str
    wallet: str
    asset_id: str           # Polymarket YES/NO token ID
    condition_id: str       # Polymarket market condition ID
    question: str           # market question text (fetched separately)
    outcome: str            # "Yes" | "No"
    side: str               # "BUY" | "SELL"
    price: float            # 0–1 decimal
    size: float             # contracts (shares)
    usd_value: float        # price * size
    timestamp: int          # unix seconds


# ------------------------------------------------------------------ #
#  Main strategy class                                                 #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  Market matcher — TF-IDF cosine similarity                          #
# (upgrade path: install sentence-transformers for semantic matching) #
# ------------------------------------------------------------------ #

class MarketMatcher:
    """
    Matches a free-text question (from Polymarket) to the best Kalshi market title.

    Uses TF-IDF (1+2-grams, sublinear TF) + cosine similarity by default.
    If sentence-transformers is installed it will be used automatically instead,
    giving semantic understanding of paraphrases like:
      "Will BTC close above $80k?" ↔ "Bitcoin price over 80,000 at month end"

    To upgrade:
        pip install sentence-transformers
    """

    def __init__(self):
        self._use_embeddings = self._try_load_embedder()
        if not self._use_embeddings:
            self._vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 1),   # unigrams only — bigrams hurt on sparse market corpora
                min_df=1,
                sublinear_tf=True,
                strip_accents="unicode",
            )
        self._matrix = None
        self._tickers: list[str] = []
        self._index_hash: str = ""

    @staticmethod
    def _try_load_embedder() -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            return True
        except ImportError:
            return False

    def _embed(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def fit(self, snapshots: list) -> None:
        """
        Build (or rebuild) the index from a list of MarketSnapshot objects.
        Only rebuilds if the set of tickers has changed.
        """
        if not snapshots:
            return

        new_hash = hashlib.md5(
            ",".join(sorted(s.ticker for s in snapshots)).encode()
        ).hexdigest()
        if new_hash == self._index_hash:
            return   # nothing changed

        self._tickers = [s.ticker for s in snapshots]
        titles = [_normalise(s.title) for s in snapshots]   # normalise titles

        if self._use_embeddings:
            self._matrix = self._embed(titles)
            logger.info(f"MarketMatcher: built embedding index ({len(titles)} markets)")
        else:
            self._matrix = self._vectorizer.fit_transform(titles)
            logger.info(f"MarketMatcher: built TF-IDF index ({len(titles)} markets)")

        self._index_hash = new_hash

    def match(self, query: str, threshold: float = 0.15) -> Optional[tuple[str, float]]:
        """
        Return (kalshi_ticker, score) of the best match above threshold, or None.
        Threshold of 0.15 is appropriate for TF-IDF cosine; 0.50 for embeddings.
        """
        if self._matrix is None or not self._tickers:
            return None

        norm_query = _normalise(query)   # normalise query too

        if self._use_embeddings:
            q_vec = self._embed([norm_query])
            scores = cosine_similarity(q_vec, self._matrix)[0]
            effective_threshold = max(threshold, 0.50)
        else:
            q_vec = self._vectorizer.transform([norm_query])
            scores = cosine_similarity(q_vec, self._matrix)[0]
            effective_threshold = threshold

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= effective_threshold:
            return self._tickers[best_idx], best_score
        return None


class CopycatStrategy:
    def __init__(
        self,
        min_trade_usd: float = 20.0,
        match_threshold: float = 0.15,   # TF-IDF cosine cutoff (0.50 if embeddings active)
        portfolio_fallback_pct: float = 0.05,
    ):
        self.min_trade_usd = min_trade_usd
        self.match_threshold = match_threshold
        self.portfolio_fallback_pct = portfolio_fallback_pct

        self._wallets: list[str] = self._load_wallets()
        self._seen_trade_ids: set[str] = set()
        self._last_scan_ts: dict[str, int] = {}
        self._trade_cache: dict[str, tuple[float, list[PolyTrade]]] = {}
        self._question_cache: dict[str, tuple[float, str]] = {}
        self._matcher = MarketMatcher()

    # ------------------------------------------------------------------ #
    #  Config                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_wallets() -> list[str]:
        raw = os.getenv("TRACKED_WALLETS", "")
        wallets = [w.strip() for w in raw.split(",") if w.strip()]
        if not wallets:
            logger.info("TRACKED_WALLETS not set — copycat strategy idle")
        else:
            logger.info(f"Tracking {len(wallets)} Polymarket wallet(s)")
        return wallets

    # ------------------------------------------------------------------ #
    #  Polymarket API calls                                                #
    # ------------------------------------------------------------------ #

    def _fetch_wallet_trades(self, wallet: str, since_ts: int) -> list[PolyTrade]:
        """Fetch recent trades for a wallet from Polymarket CLOB API."""
        now = time.time()
        cache = self._trade_cache.get(wallet)
        if cache and now < cache[0]:
            return [t for t in cache[1] if t.timestamp >= since_ts]

        try:
            resp = _SESSION.get(
                f"{_POLY_CLOB}/data/trades",
                params={"maker_address": wallet, "limit": 50},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Polymarket trades fetch failed for {wallet[:10]}…: {e}")
            return []

        raw_trades = data if isinstance(data, list) else data.get("data", data.get("trades", []))
        trades = []
        for t in raw_trades:
            try:
                ts = int(t.get("timestamp") or 0)
                if isinstance(ts, str):
                    ts = int(datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc).timestamp())
                price = float(t.get("price", 0))
                size  = float(t.get("size", 0))
                trades.append(PolyTrade(
                    trade_id    = str(t.get("id") or t.get("trade_id", "")),
                    wallet      = wallet,
                    asset_id    = str(t.get("asset_id", "")),
                    condition_id= str(t.get("market", "")),
                    question    = "",   # filled in below
                    outcome     = str(t.get("outcome", "Yes")),
                    side        = str(t.get("side", "BUY")).upper(),
                    price       = price,
                    size        = size,
                    usd_value   = price * size,
                    timestamp   = ts,
                ))
            except Exception as e:
                logger.debug(f"Skipping malformed trade: {e}")

        self._trade_cache[wallet] = (now + 30, trades)   # 30s TTL
        return [t for t in trades if t.timestamp >= since_ts]

    def _fetch_question(self, asset_id: str) -> str:
        """Fetch Polymarket market question for a token asset_id. 1hr cache."""
        now = time.time()
        cached = self._question_cache.get(asset_id)
        if cached and now < cached[0]:
            return cached[1]

        question = ""
        try:
            resp = _SESSION.get(
                f"{_POLY_GAMMA}/markets",
                params={"clob_token_ids": asset_id},
                timeout=10,
            )
            resp.raise_for_status()
            markets = resp.json()
            if isinstance(markets, list) and markets:
                question = markets[0].get("question", "")
            elif isinstance(markets, dict):
                question = markets.get("question", "")
        except Exception as e:
            logger.debug(f"Question fetch failed for {asset_id[:12]}…: {e}")

        self._question_cache[asset_id] = (now + 3600, question)  # 1hr TTL
        return question

    # ------------------------------------------------------------------ #
    #  Market matching                                                     #
    # ------------------------------------------------------------------ #

    def _match_kalshi_ticker(
        self,
        question: str,
        kalshi_snapshots: list,
    ) -> Optional[tuple[str, float]]:
        """Fit/refresh the index then return (ticker, score) or None."""
        self._matcher.fit(kalshi_snapshots)
        return self._matcher.match(question, threshold=self.match_threshold)

    # ------------------------------------------------------------------ #
    #  Portfolio estimation                                                #
    # ------------------------------------------------------------------ #

    def _estimate_portfolio(self, wallet: str) -> float:
        """
        Rolling 7-day USD trade volume for the wallet as a proxy for portfolio size.
        Uses whatever is in the trade cache (no extra API call).
        """
        cutoff = int(time.time()) - _SEVEN_DAYS
        cached = self._trade_cache.get(wallet)
        if not cached:
            return 0.0
        trades = cached[1]
        return sum(t.usd_value for t in trades if t.timestamp >= cutoff)

    def _proportional_contracts(
        self,
        trade: PolyTrade,
        my_balance: float,
        their_portfolio: float,
        kalshi_price_cents: int,
    ) -> int:
        """
        Compute how many Kalshi contracts to buy, mirroring their proportional bet.
        Falls back to portfolio_fallback_pct of my_balance if their history is thin.
        """
        if kalshi_price_cents <= 0:
            return 0

        if their_portfolio > 0:
            trade_fraction = trade.usd_value / their_portfolio
        else:
            trade_fraction = self.portfolio_fallback_pct

        my_usd = trade_fraction * my_balance
        contracts = int(my_usd / (kalshi_price_cents / 100))
        return max(1, contracts)

    # ------------------------------------------------------------------ #
    #  Main scan                                                           #
    # ------------------------------------------------------------------ #

    def scan(
        self,
        my_balance: float,
        kalshi_snapshots: list,           # list[MarketSnapshot] from Scanner
    ) -> list:                            # list[Signal]
        from scanner import Signal        # avoid circular import at module level

        if not self._wallets:
            return []

        now_ts = int(time.time())
        signals = []

        # Build a quick title→snapshot lookup for matching
        snap_by_ticker = {s.ticker: s for s in kalshi_snapshots}

        for wallet in self._wallets:
            # Only look at trades since last scan (or last hour on first run)
            since = self._last_scan_ts.get(wallet, now_ts - 3600)
            trades = self._fetch_wallet_trades(wallet, since_ts=since)
            their_portfolio = self._estimate_portfolio(wallet)

            for trade in trades:
                # Skip already-seen trades and non-BUY orders
                if trade.trade_id in self._seen_trade_ids:
                    continue
                if trade.side != "BUY":
                    self._seen_trade_ids.add(trade.trade_id)
                    continue
                if trade.usd_value < self.min_trade_usd:
                    self._seen_trade_ids.add(trade.trade_id)
                    continue

                # Resolve the market question
                if not trade.question:
                    trade.question = self._fetch_question(trade.asset_id)
                if not trade.question:
                    logger.debug(f"No question for asset {trade.asset_id[:12]}…, skipping")
                    self._seen_trade_ids.add(trade.trade_id)
                    continue

                # Match to Kalshi
                match = self._match_kalshi_ticker(trade.question, kalshi_snapshots)
                if match is None:
                    logger.debug(
                        f"No Kalshi match for: '{trade.question[:60]}' "
                        f"(wallet {wallet[:10]}…)"
                    )
                    self._seen_trade_ids.add(trade.trade_id)
                    continue

                kalshi_ticker, match_score = match
                snap = snap_by_ticker.get(kalshi_ticker)
                if snap is None or snap.yes_ask is None or snap.yes_bid is None:
                    self._seen_trade_ids.add(trade.trade_id)
                    continue

                direction = "yes" if trade.outcome.lower() == "yes" else "no"
                price = snap.yes_ask if direction == "yes" else 100 - snap.yes_bid
                contracts = self._proportional_contracts(
                    trade, my_balance, their_portfolio, price
                )

                # Confidence: blend match quality + trade size signal
                # Higher match score → more confident; large trades from known wallet → more confident
                size_confidence = min(trade.usd_value / 500, 0.5)   # caps at 0.5 for $500+ trades
                confidence = round(match_score * 0.5 + size_confidence * 0.5, 3)

                logger.info(
                    f"[COPYCAT] {wallet[:10]}… bought {trade.outcome} on '{trade.question[:50]}' "
                    f"→ matched {kalshi_ticker} (score={match_score:.2f}) "
                    f"→ {direction} {contracts}x @ {price}¢  conf={confidence:.2f}"
                )

                signals.append(Signal(
                    ticker=kalshi_ticker,
                    title=snap.title,
                    strategy="copycat",
                    direction=direction,
                    confidence=confidence,
                    suggested_price=price,
                    suggested_contracts=contracts,
                ))
                self._seen_trade_ids.add(trade.trade_id)

            self._last_scan_ts[wallet] = now_ts

        return signals
