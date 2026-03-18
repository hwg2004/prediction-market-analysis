"""
Pulls historical resolved markets and their candlestick price series from Kalshi.

Output
------
data/settled_markets.csv   — one row per resolved market with outcome
data/candlesticks/          — one CSV per ticker: timestamped price series

Usage
-----
    python -m data.history
    python -m data.history --tickers KXBTCD-25JAN31-T95000 --interval 60
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from kalshi_python_sync import Configuration, KalshiClient, HistoricalApi, MarketApi
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent


def series_ticker(ticker: str) -> str:
    """Derive series ticker from market ticker: 'KXBTCD-25JAN31-T95000' → 'KXBTCD'."""
    return ticker.split("-")[0]
CANDLES_DIR = DATA_DIR / "candlesticks"
SETTLED_CSV = DATA_DIR / "settled_markets.csv"


# ------------------------------------------------------------------ #
#  Client bootstrap (no auth needed for public historical endpoints,  #
#  but private key is required for the authenticated ones)            #
# ------------------------------------------------------------------ #

def _build_api_client() -> tuple[HistoricalApi, MarketApi]:
    config = Configuration(host="https://api.elections.kalshi.com/trade-api/v2")
    key_path = os.getenv("KALSHI_KEY_PATH", "kalshi.pem")
    if Path(key_path).exists():
        with open(key_path) as f:
            config.private_key_pem = f.read()
    config.api_key_id = os.getenv("KALSHI_KEY_ID")
    client = KalshiClient(config)
    return HistoricalApi(client), MarketApi(client)


# ------------------------------------------------------------------ #
#  Step 1 — fetch all settled markets                                 #
# ------------------------------------------------------------------ #

def fetch_settled_markets(
    market_api: MarketApi,
    max_markets: int = 5000,
    min_settled_ts: int | None = None,
) -> pd.DataFrame:
    """
    Fetch resolved markets, newest first.
    Returns a DataFrame with columns:
        ticker, title, event_ticker, result, settled_time,
        yes_bid, yes_ask, last_price, volume, open_interest
    """
    rows, cursor = [], None
    print(f"Fetching settled markets (max {max_markets})...")

    while len(rows) < max_markets:
        kwargs = dict(limit=1000, cursor=cursor, status="settled", mve_filter="exclude")
        if min_settled_ts:
            kwargs["min_settled_ts"] = min_settled_ts
        resp = market_api.get_markets(**kwargs)

        for m in resp.markets:
            rows.append({
                "ticker": m.ticker,
                "title": getattr(m, "title", ""),
                "event_ticker": getattr(m, "event_ticker", ""),
                "result": getattr(m, "result", None),          # "yes" | "no" | None
                "settled_time": getattr(m, "settled_time", None),
                "close_time": getattr(m, "close_time", None),
                "open_time": getattr(m, "open_time", None),
                "yes_bid": getattr(m, "yes_bid", None),
                "yes_ask": getattr(m, "yes_ask", None),
                "last_price": getattr(m, "last_price", None),
                "volume": getattr(m, "volume", 0),
                "open_interest": getattr(m, "open_interest", 0),
            })

        cursor = resp.cursor
        print(f"  fetched {len(rows)} markets so far...")
        if cursor is None or len(resp.markets) == 0:
            break
        time.sleep(0.2)   # be polite to the API

    df = pd.DataFrame(rows)
    df.to_csv(SETTLED_CSV, index=False)
    print(f"Saved {len(df)} settled markets → {SETTLED_CSV}")
    return df


# ------------------------------------------------------------------ #
#  Step 2 — fetch candlestick history for a list of tickers          #
# ------------------------------------------------------------------ #

def fetch_candlesticks(
    hist_api: HistoricalApi,
    tickers: list[str],
    period_interval: int = 60,      # 1, 60, or 1440 minutes
    lookback_days: int = 90,
) -> dict[str, pd.DataFrame]:
    """
    Pull price/volume candlesticks for each ticker.
    Saves one CSV per ticker under data/candlesticks/.
    Returns dict of ticker -> DataFrame.
    """
    CANDLES_DIR.mkdir(exist_ok=True)
    results = {}

    now_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = now_ts - lookback_days * 86400

    for i, ticker in enumerate(tickers):
        out_path = CANDLES_DIR / f"{ticker}.csv"
        if out_path.exists():
            print(f"  [{i+1}/{len(tickers)}] {ticker} — cached, skipping")
            results[ticker] = pd.read_csv(out_path)
            continue

        try:
            resp = hist_api.get_market_candlesticks_historical(
                ticker=ticker,
                start_ts=start_ts,
                end_ts=now_ts,
                period_interval=period_interval,
            )
            candles = resp.candlesticks if hasattr(resp, "candlesticks") else []

            rows = []
            for c in candles:
                rows.append({
                    "ts": c.end_period_ts,
                    "datetime": datetime.fromtimestamp(c.end_period_ts, tz=timezone.utc).isoformat(),
                    # price distribution — take mid (open+close average) if available
                    "price_open": getattr(c.price, "open", None),
                    "price_close": getattr(c.price, "close", None),
                    "price_high": getattr(c.price, "high", None),
                    "price_low": getattr(c.price, "low", None),
                    # bid/ask distributions
                    "yes_bid_open": getattr(c.yes_bid, "open", None),
                    "yes_bid_close": getattr(c.yes_bid, "close", None),
                    "yes_ask_open": getattr(c.yes_ask, "open", None),
                    "yes_ask_close": getattr(c.yes_ask, "close", None),
                    "volume": float(c.volume),
                    "open_interest": float(c.open_interest),
                })

            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)
            print(f"  [{i+1}/{len(tickers)}] {ticker} — {len(df)} candles → {out_path.name}")
            results[ticker] = df

        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {ticker} — ERROR: {e}")
            results[ticker] = pd.DataFrame()

        time.sleep(0.15)

    return results


# ------------------------------------------------------------------ #
#  Step 2b — fetch live candlesticks for OPEN markets                #
# ------------------------------------------------------------------ #

def fetch_live_candlesticks(
    market_api: MarketApi,
    ticker: str,
    period_interval: int = 60,
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """
    Pull candlestick data for an open (live) market using the non-historical endpoint.
    Uses series_ticker derived from ticker prefix.
    Returns a DataFrame with the same column schema as fetch_candlesticks().
    Returns an empty DataFrame on error.
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = now_ts - lookback_hours * 3600
    s_ticker = series_ticker(ticker)

    try:
        resp = market_api.get_market_candlesticks(
            series_ticker=s_ticker,
            ticker=ticker,
            start_ts=start_ts,
            end_ts=now_ts,
            period_interval=period_interval,
        )
        candles = resp.candlesticks if hasattr(resp, "candlesticks") else []

        rows = []
        for c in candles:
            rows.append({
                "ts": c.end_period_ts,
                "datetime": datetime.fromtimestamp(c.end_period_ts, tz=timezone.utc).isoformat(),
                "price_open": getattr(c.price, "open", None),
                "price_close": getattr(c.price, "close", None),
                "price_high": getattr(c.price, "high", None),
                "price_low": getattr(c.price, "low", None),
                "yes_bid_open": getattr(c.yes_bid, "open", None),
                "yes_bid_close": getattr(c.yes_bid, "close", None),
                "yes_ask_open": getattr(c.yes_ask, "open", None),
                "yes_ask_close": getattr(c.yes_ask, "close", None),
                "volume": float(c.volume),
                "open_interest": float(c.open_interest),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  {ticker} live candles ERROR: {e}")
        return pd.DataFrame()


# ------------------------------------------------------------------ #
#  Step 3 — convenience: load settled markets and pull candles       #
# ------------------------------------------------------------------ #

def build_training_dataset(
    max_markets: int = 2000,
    min_volume: int = 100,
    period_interval: int = 60,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    High-level helper:
      1. Fetch settled markets
      2. Filter to liquid ones (volume >= min_volume)
      3. Pull candlestick history for each
      4. Return the settled markets DataFrame (candles saved to disk)
    """
    hist_api, market_api = _build_api_client()

    # Fetch or load settled markets
    if SETTLED_CSV.exists():
        print(f"Loading cached settled markets from {SETTLED_CSV}")
        markets_df = pd.read_csv(SETTLED_CSV)
    else:
        markets_df = fetch_settled_markets(market_api, max_markets=max_markets)

    # Filter to markets with known outcomes and sufficient liquidity
    df = markets_df.dropna(subset=["result"])
    df = df[df["volume"].fillna(0) >= min_volume]
    print(f"Filtered to {len(df)} liquid settled markets with known outcomes")

    tickers = df["ticker"].tolist()
    fetch_candlesticks(hist_api, tickers, period_interval=period_interval, lookback_days=lookback_days)

    return df


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Kalshi historical data")
    parser.add_argument("--tickers", nargs="*", help="Specific historical tickers to pull candles for")
    parser.add_argument("--live-ticker", help="Pull live candlesticks for a single open market ticker")
    parser.add_argument("--check-archive-cutoff", action="store_true",
                        help="Print the historical archive cutoff date and exit")
    parser.add_argument("--max-markets", type=int, default=2000)
    parser.add_argument("--min-volume", type=int, default=100)
    parser.add_argument("--interval", type=int, default=60, choices=[1, 60, 1440],
                        help="Candlestick interval in minutes (1, 60, or 1440)")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--lookback-hours", type=int, default=24,
                        help="Hours to look back for --live-ticker")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch settled markets even if cache exists")
    args = parser.parse_args()

    hist_api, market_api = _build_api_client()

    if args.check_archive_cutoff:
        cutoff = hist_api.get_historical_cutoff()
        print(f"Historical archive cutoff: {cutoff}")
        raise SystemExit(0)

    if args.live_ticker:
        df = fetch_live_candlesticks(market_api, args.live_ticker,
                                     period_interval=args.interval,
                                     lookback_hours=args.lookback_hours)
        print(df.to_string())
        raise SystemExit(0)

    if args.refresh and SETTLED_CSV.exists():
        SETTLED_CSV.unlink()

    if args.tickers:
        CANDLES_DIR.mkdir(exist_ok=True)
        fetch_candlesticks(hist_api, args.tickers, period_interval=args.interval,
                           lookback_days=args.lookback_days)
    else:
        build_training_dataset(
            max_markets=args.max_markets,
            min_volume=args.min_volume,
            period_interval=args.interval,
            lookback_days=args.lookback_days,
        )
