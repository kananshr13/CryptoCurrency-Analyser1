"""
Data Ingestion Module
=====================
Fetches:
  - Live cryptocurrency prices & market data from CoinGecko (free tier, no API key)
  - News headlines from CryptoPanic (free tier) or falls back to curated mock data

In a production pipeline this would also pull from:
  - Twitter/X API (social sentiment)
  - Reddit PRAW (community sentiment)
  - NewsAPI (broad financial news)
"""

import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Top coins tracked by this analyser
TRACKED_COINS = [
    {"id": "bitcoin",      "symbol": "BTC", "name": "Bitcoin"},
    {"id": "ethereum",     "symbol": "ETH", "name": "Ethereum"},
    {"id": "solana",       "symbol": "SOL", "name": "Solana"},
    {"id": "binancecoin",  "symbol": "BNB", "name": "BNB"},
    {"id": "ripple",       "symbol": "XRP", "name": "XRP"},
    {"id": "cardano",      "symbol": "ADA", "name": "Cardano"},
    {"id": "dogecoin",     "symbol": "DOGE", "name": "Dogecoin"},
    {"id": "polkadot",     "symbol": "DOT", "name": "Polkadot"},
]


class CoinGeckoClient:
    """Thin wrapper around CoinGecko public API with rate-limit handling."""

    def __init__(self, rate_limit_delay: float = 1.2):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.delay = rate_limit_delay  # CoinGecko free tier: ~10 req/min

    def _get(self, endpoint: str, params: dict = None, retries: int = 3) -> dict | list:
        url = f"{COINGECKO_BASE}/{endpoint}"
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                time.sleep(self.delay)
                return resp.json()
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning("Rate limited — waiting %ds", wait)
                    time.sleep(wait)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                logger.warning("Request failed (attempt %d): %s", attempt + 1, e)
                time.sleep(5)
        raise ConnectionError(f"Failed to fetch {url} after {retries} attempts")

    def get_prices(self, coin_ids: list[str] = None) -> pd.DataFrame:
        """
        Fetch current market data for tracked coins.
        Returns a DataFrame with price, volume, market cap, and 24h change.
        """
        ids = ",".join(coin_ids or [c["id"] for c in TRACKED_COINS])
        data = self._get("coins/markets", params={
            "vs_currency": "usd",
            "ids": ids,
            "order": "market_cap_desc",
            "per_page": 20,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "1h,24h,7d",
        })

        df = pd.DataFrame(data)[[
            "id", "symbol", "name", "current_price",
            "market_cap", "total_volume",
            "price_change_percentage_1h_in_currency",
            "price_change_percentage_24h_in_currency",
            "price_change_percentage_7d_in_currency",
            "high_24h", "low_24h", "circulating_supply",
        ]]
        df.columns = [
            "id", "symbol", "name", "price_usd",
            "market_cap_usd", "volume_24h_usd",
            "change_1h_pct", "change_24h_pct", "change_7d_pct",
            "high_24h", "low_24h", "circulating_supply",
        ]
        df["fetched_at"] = datetime.utcnow().isoformat()
        return df

    def get_ohlcv(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch OHLCV (Open/High/Low/Close/Volume) history.
        Useful for training price prediction models and correlation analysis.
        """
        data = self._get(f"coins/{coin_id}/ohlc", params={"vs_currency": "usd", "days": days})
        df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df["coin_id"] = coin_id
        return df.drop("timestamp_ms", axis=1)

    def get_market_chart(self, coin_id: str, days: int = 7) -> pd.DataFrame:
        """Hourly price series for the last N days."""
        data = self._get(f"coins/{coin_id}/market_chart", params={"vs_currency": "usd", "days": days})
        prices  = pd.DataFrame(data["prices"],  columns=["ts", "price"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])
        df = prices.merge(volumes, on="ts")
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        df["coin_id"] = coin_id
        return df.drop("ts", axis=1)


class NewsIngestion:
    """
    Fetches cryptocurrency news headlines for sentiment analysis.

    Live source: CryptoPanic free tier (no key needed for basic headlines)
    Fallback: curated realistic mock headlines (ensures pipeline always runs)
    """

    CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, cryptopanic_api_key: Optional[str] = None):
        self.api_key = cryptopanic_api_key
        self.session = requests.Session()

    def fetch_live(self, coin_symbol: str = None, limit: int = 20) -> list[dict]:
        """Fetch from CryptoPanic (requires free API key from cryptopanic.com)."""
        if not self.api_key:
            logger.info("No CryptoPanic API key — using mock data")
            return []
        params = {
            "auth_token": self.api_key,
            "kind": "news",
            "public": "true",
        }
        if coin_symbol:
            params["currencies"] = coin_symbol.upper()

        try:
            resp = self.session.get(self.CRYPTOPANIC_BASE, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])[:limit]
            return [
                {
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "source": r.get("source", {}).get("title", "Unknown"),
                    "published_at": r.get("published_at", ""),
                    "url": r.get("url", ""),
                    "coin": coin_symbol,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("CryptoPanic fetch failed: %s", e)
            return []

    def get_mock_news(self, coin_name: str) -> list[dict]:
        """
        Curated realistic news headlines covering bullish, bearish, and neutral signals.
        These are used for development and demonstration of the NLP pipeline.
        """
        templates = [
            {
                "title": f"{coin_name} sees massive institutional accumulation as ETF inflows hit record highs",
                "body": f"Major asset managers report unprecedented {coin_name} inflows, signalling strong institutional confidence.",
                "source": "CoinDesk", "sentiment_hint": "positive",
            },
            {
                "title": f"{coin_name} network upgrade boosts transaction throughput and reduces fees significantly",
                "body": f"The latest protocol upgrade has improved {coin_name}'s scalability, attracting new developer partnerships.",
                "source": "The Block", "sentiment_hint": "positive",
            },
            {
                "title": f"Regulatory concerns weigh on {coin_name} as SEC scrutiny intensifies amid crackdown fears",
                "body": f"Uncertainty around potential ban or regulations has created volatility and investor concern.",
                "source": "Bloomberg Crypto", "sentiment_hint": "negative",
            },
            {
                "title": f"{coin_name} rallies 12% as macro conditions improve and risk appetite returns",
                "body": f"Risk-on sentiment has pushed {coin_name} to a monthly high with analysts bullish on momentum.",
                "source": "Reuters", "sentiment_hint": "positive",
            },
            {
                "title": f"On-chain data shows {coin_name} long-term holders accumulating — historically bullish signal",
                "body": f"Exchange supply declining as conviction holders move assets to cold storage.",
                "source": "Glassnode", "sentiment_hint": "positive",
            },
            {
                "title": f"{coin_name} faces sell pressure amid broader market correction and liquidation cascade",
                "body": f"Macro headwinds have triggered a $200M liquidation event, with {coin_name} declining sharply.",
                "source": "FT", "sentiment_hint": "negative",
            },
            {
                "title": f"{coin_name} trading volume remains stable as market digests recent price action",
                "body": f"Analysts note consolidation in {coin_name} price as the market awaits clearer direction.",
                "source": "CoinTelegraph", "sentiment_hint": "neutral",
            },
            {
                "title": f"Major exchange reports {coin_name} hack — vulnerability exploited for $50M",
                "body": f"A security breach has caused significant {coin_name} outflows, raising concerns about custody risk.",
                "source": "Decrypt", "sentiment_hint": "negative",
            },
            {
                "title": f"{coin_name} partnership with global payment giant drives adoption milestone",
                "body": f"A strategic partnership could significantly expand real-world usage and strengthen fundamentals.",
                "source": "CoinDesk", "sentiment_hint": "positive",
            },
            {
                "title": f"Analysts divided on {coin_name} short-term outlook as dominance shifts",
                "body": f"Mixed signals from on-chain metrics and macro environment leave {coin_name} direction uncertain.",
                "source": "CryptoSlate", "sentiment_hint": "neutral",
            },
        ]

        now = datetime.utcnow()
        return [
            {
                "title": t["title"],
                "body": t["body"],
                "source": t["source"],
                "published_at": (now - timedelta(hours=i * 2)).isoformat(),
                "url": f"https://example.com/news/{coin_name.lower()}-{i}",
                "coin": coin_name,
                "sentiment_hint": t["sentiment_hint"],  # ground truth for evaluation
            }
            for i, t in enumerate(templates)
        ]

    def get_news(self, coin_name: str, coin_symbol: str, limit: int = 10) -> list[dict]:
        """Get news — live if API key available, mock otherwise."""
        live = self.fetch_live(coin_symbol, limit)
        return live if live else self.get_mock_news(coin_name)
