"""
News and RSS sentiment fetcher for prediction market signal generation.

Sentiment scorer priority:
  1. vaderSentiment (install: pip install vaderSentiment)
  2. textblob       (install: pip install textblob)
  3. keyword fallback (built-in, no install needed)
"""

import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import feedparser

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Sentiment scorer detection                                          #
# ------------------------------------------------------------------ #

_SCORER = "none"
_vader_analyzer = None
_TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_analyzer = SentimentIntensityAnalyzer()
    _SCORER = "vader"
except ImportError:
    try:
        from textblob import TextBlob as _TextBlobClass
        _TextBlob = _TextBlobClass
        _SCORER = "textblob"
    except ImportError:
        pass

logger.info(f"News sentiment scorer: {_SCORER}")

# ------------------------------------------------------------------ #
#  RSS feed catalogue                                                  #
# ------------------------------------------------------------------ #

RSS_FEEDS = {
    "reuters":  "https://feeds.reuters.com/reuters/topNews",
    "ap":       "https://rsshub.app/apnews/topics/apf-topnews",
    "bbc":      "https://feeds.bbci.co.uk/news/rss.xml",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "espn":     "https://www.espn.com/espn/rss/news",
}

_STOPWORDS = {
    "will", "the", "and", "for", "that", "this", "with", "from",
    "have", "been", "they", "their", "what", "when", "where",
    "which", "about", "than", "into", "over", "also", "more",
}


@dataclass
class Article:
    headline: str
    source: str
    published_at: str       # ISO string or ""
    sentiment_score: float  # -1 to 1
    url: str


@dataclass
class NewsFetcher:
    ttl_seconds: int = 300  # 5-minute cache

    _cache: dict = field(default_factory=dict, repr=False)
    _newsapi: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        key = os.getenv("NEWSAPI_KEY")
        if key:
            try:
                from newsapi import NewsApiClient
                self._newsapi = NewsApiClient(api_key=key)
                logger.info("NewsAPI client initialised")
            except ImportError:
                logger.warning("newsapi-python not installed; skipping NewsAPI")

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _score(self, text: str) -> float:
        if not text:
            return 0.0
        if _SCORER == "vader" and _vader_analyzer is not None:
            return _vader_analyzer.polarity_scores(text)["compound"]
        if _SCORER == "textblob" and _TextBlob is not None:
            return float(_TextBlob(text).sentiment.polarity)
        return self._keyword_score(text)

    @staticmethod
    def _keyword_score(text: str) -> float:
        positive = {"win", "rise", "up", "gain", "growth", "positive", "beat", "approve", "pass"}
        negative = {"loss", "fall", "down", "drop", "fail", "reject", "crash", "miss", "veto"}
        words = re.findall(r"\w+", text.lower())
        score = sum(1 for w in words if w in positive) - sum(1 for w in words if w in negative)
        return max(-1.0, min(1.0, score / max(len(words), 1) * 10))

    # ------------------------------------------------------------------ #
    #  Query terms                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _query_terms(ticker: str, title: str) -> list[str]:
        """Extract search terms from ticker prefix and market title."""
        # Ticker prefix (e.g. KXBTCD → BTC)
        prefix = ticker.split("-")[0].lstrip("KX").upper()

        # Significant words from title
        title_words = [
            w for w in re.findall(r"[A-Za-z]{5,}", title)
            if w.lower() not in _STOPWORDS
        ]

        terms = []
        if len(prefix) >= 3:
            terms.append(prefix)
        terms.extend(title_words[:3])
        return list(dict.fromkeys(terms))   # deduplicate, preserve order

    # ------------------------------------------------------------------ #
    #  Fetchers                                                            #
    # ------------------------------------------------------------------ #

    def _fetch_rss(self, query_terms: list[str]) -> list[Article]:
        articles = []
        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    headline = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "")
                    combined = f"{headline} {summary}".lower()

                    if not any(t.lower() in combined for t in query_terms):
                        continue

                    pub = getattr(entry, "published", "")
                    score = self._score(f"{headline} {summary}")
                    link = getattr(entry, "link", "")
                    articles.append(Article(
                        headline=headline,
                        source=source,
                        published_at=pub,
                        sentiment_score=score,
                        url=link,
                    ))
            except Exception as e:
                logger.debug(f"RSS error ({source}): {e}")
        return articles

    def _fetch_newsapi(self, query: str) -> list[Article]:
        if self._newsapi is None:
            return []
        try:
            resp = self._newsapi.get_everything(
                q=query, language="en", sort_by="publishedAt", page_size=20
            )
            articles = []
            for a in resp.get("articles", []):
                headline = a.get("title") or ""
                desc = a.get("description") or ""
                score = self._score(f"{headline} {desc}")
                articles.append(Article(
                    headline=headline,
                    source=a.get("source", {}).get("name", "newsapi"),
                    published_at=a.get("publishedAt", ""),
                    sentiment_score=score,
                    url=a.get("url", ""),
                ))
            return articles
        except Exception as e:
            logger.debug(f"NewsAPI error: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def fetch_for_market(self, ticker: str, title: str) -> list[Article]:
        cache_key = hashlib.md5(f"{ticker}:{title}".encode()).hexdigest()
        now = time.time()

        if cache_key in self._cache:
            expiry, cached_articles = self._cache[cache_key]
            if now < expiry:
                return cached_articles

        terms = self._query_terms(ticker, title)
        if not terms:
            return []

        query = " OR ".join(terms[:3])
        rss_articles = self._fetch_rss(terms)
        api_articles = self._fetch_newsapi(query)
        all_articles = rss_articles + api_articles

        # Sort by recency (rough: lexicographic on published_at string)
        all_articles.sort(key=lambda a: a.published_at, reverse=True)

        self._cache[cache_key] = (now + self.ttl_seconds, all_articles)
        return all_articles

    def aggregate_sentiment(self, articles: list[Article]) -> float:
        """Recency-weighted mean sentiment. Returns 0.0 for empty list."""
        if not articles:
            return 0.0

        now = datetime.now(timezone.utc)
        weighted_sum = 0.0
        weight_total = 0.0

        for a in articles:
            age_hours = 0.0
            if a.published_at:
                try:
                    from email.utils import parsedate_to_datetime
                    pub = parsedate_to_datetime(a.published_at)
                    age_hours = (now - pub.astimezone(timezone.utc)).total_seconds() / 3600
                except Exception:
                    pass
            weight = math.exp(-age_hours / 24)
            weighted_sum += a.sentiment_score * weight
            weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.0
