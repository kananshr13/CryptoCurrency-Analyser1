"""
CryptoCurrency Sentiment Analyser — Main Pipeline
==================================================
Orchestrates the full end-to-end flow:
  1. Fetch live prices from CoinGecko
  2. Ingest news headlines (live or mock)
  3. Run NLP sentiment analysis on each headline
  4. Aggregate per-coin sentiment scores
  5. Evaluate model accuracy against labeled data
  6. Generate visualisations and export report

Run:
    python main.py                     # full pipeline
    python main.py --coin bitcoin      # single coin deep-dive
    python main.py --eval              # evaluation only
    python main.py --no-finbert        # VADER-only (faster)
"""

import os
import json
import logging
import argparse
from datetime import datetime

import pandas as pd

from data.ingestion import CoinGeckoClient, NewsIngestion, TRACKED_COINS
from models.sentiment_model import CryptoSentimentAnalyser
from models.evaluation import evaluate_model, build_labeled_dataset, compare_vader_vs_ensemble
from utils.visualisation import (
    plot_sentiment_dashboard,
    plot_label_distribution,
    plot_confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
os.makedirs("outputs", exist_ok=True)


def run_pipeline(
    coins: list[dict] = None,
    use_finbert: bool = True,
    run_eval: bool = True,
    cryptopanic_key: str = None,
) -> dict:
    """
    Full analysis pipeline.

    Returns a dict with per-coin summaries and evaluation report.
    """
    coins = coins or TRACKED_COINS
    print("\n" + "="*60)
    print("  🪙  CryptoCurrency NLP Sentiment Analyser")
    print("  Built for: Management Trainee (AI-ML) — EIMA Analytics")
    print("="*60)

    # ── Step 1: Load models ────────────────────────────────────────────────────
    print("\n[1/5] Loading NLP models...")
    analyser = CryptoSentimentAnalyser(use_finbert=use_finbert)

    # ── Step 2: Fetch live prices ──────────────────────────────────────────────
    print("[2/5] Fetching cryptocurrency prices from CoinGecko...")
    gecko = CoinGeckoClient()
    try:
        prices_df = gecko.get_prices([c["id"] for c in coins])
        print(f"      ✓ Fetched prices for {len(prices_df)} coins")
    except Exception as e:
        logger.warning("Price fetch failed (%s) — using placeholder data", e)
        prices_df = pd.DataFrame(coins).rename(columns={"id": "id"})
        prices_df["price_usd"] = [62450, 3310, 178, 592, 0.58, 0.45, 0.15, 7.8][:len(coins)]
        prices_df["change_24h_pct"] = [2.4, 1.8, -1.2, 0.7, -2.1, -0.5, 3.2, -1.8][:len(coins)]

    # ── Step 3: Ingest news & run sentiment ────────────────────────────────────
    print("[3/5] Ingesting news and running sentiment analysis...")
    news_client = NewsIngestion(cryptopanic_api_key=cryptopanic_key)
    coin_summaries = []

    for coin in coins:
        news_items = news_client.get_news(coin["name"], coin["symbol"])
        texts = [f"{n['title']} {n['body']}" for n in news_items]
        results = analyser.analyse_batch(texts)
        summary = analyser.aggregate(results)

        # Attach price data
        price_row = prices_df[prices_df["id"] == coin["id"]]
        price_usd  = float(price_row["price_usd"].iloc[0])  if not price_row.empty else None
        change_24h = float(price_row["change_24h_pct"].iloc[0]) if not price_row.empty else None

        coin_summaries.append({
            "coin_id":    coin["id"],
            "symbol":     coin["symbol"],
            "name":       coin["name"],
            "price_usd":  price_usd,
            "change_24h_pct": change_24h,
            "sentiment":  summary,
            "n_articles": len(results),
            "top_keywords": summary["keywords"][:5],
        })

        signal  = summary["signal"]
        score   = summary["score"]
        pct_str = f"{change_24h:+.1f}%" if change_24h is not None else "N/A"
        price_str = f"${price_usd:,.2f}" if price_usd else "N/A"
        print(f"      {coin['symbol']:<6} {price_str:<12} {pct_str:<8}  Sentiment: {score:+.3f}  → {signal}")

    # ── Step 4: Visualisations ─────────────────────────────────────────────────
    print("[4/5] Generating visualisations...")
    plot_sentiment_dashboard(coin_summaries)

    # Label distribution for first coin (as example)
    first_coin = coins[0]
    news_items = news_client.get_news(first_coin["name"], first_coin["symbol"])
    texts = [f"{n['title']} {n['body']}" for n in news_items]
    results = analyser.analyse_batch(texts)
    plot_label_distribution(first_coin["name"], results)
    print("      ✓ Charts saved to outputs/")

    # ── Step 5: Model evaluation ───────────────────────────────────────────────
    eval_report = None
    if run_eval:
        print("[5/5] Running model evaluation on labeled test set...")
        labeled = build_labeled_dataset()
        eval_report = evaluate_model(analyser, labeled)
        eval_report.print_summary()
        plot_confusion_matrix(
            eval_report.confusion_matrix,
            labels=["Positive", "Negative", "Neutral"],
        )
        compare_vader_vs_ensemble(analyser, labeled)

    # ── Export JSON report ─────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "model": "CryptoSentimentAnalyser (VADER + FinBERT ensemble)",
        "coins_analysed": len(coin_summaries),
        "summaries": coin_summaries,
        "evaluation": {
            "accuracy": eval_report.accuracy if eval_report else None,
            "f1_macro": eval_report.f1_macro if eval_report else None,
        } if eval_report else None,
    }

    report_path = "outputs/report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Report saved to {report_path}")

    # ── Print final summary table ──────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  {'Coin':<12} {'Price':<12} {'24h':>7}  {'Score':>8}  {'Signal':<12}  Keywords")
    print("-"*65)
    for c in coin_summaries:
        price_str = f"${c['price_usd']:,.2f}" if c["price_usd"] else "N/A"
        chg_str   = f"{c['change_24h_pct']:+.1f}%" if c["change_24h_pct"] is not None else "N/A"
        kws = ", ".join(c["top_keywords"][:3]) if c["top_keywords"] else "-"
        print(f"  {c['symbol']:<12} {price_str:<12} {chg_str:>7}  {c['sentiment']['score']:>+.3f}   {c['sentiment']['signal']:<12}  {kws}")
    print("="*65)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CryptoCurrency NLP Sentiment Analyser")
    parser.add_argument("--coin",       type=str,  default=None,  help="Run for a single coin ID (e.g. bitcoin)")
    parser.add_argument("--eval",       action="store_true",       help="Run model evaluation only")
    parser.add_argument("--no-finbert", action="store_true",       help="Use VADER only (faster, no model download)")
    parser.add_argument("--api-key",    type=str,  default=None,  help="CryptoPanic API key for live news")
    args = parser.parse_args()

    target_coins = [c for c in TRACKED_COINS if c["id"] == args.coin] if args.coin else TRACKED_COINS

    if args.eval:
        analyser = CryptoSentimentAnalyser(use_finbert=not args.no_finbert)
        labeled  = build_labeled_dataset()
        report   = evaluate_model(analyser, labeled)
        report.print_summary()
    else:
        run_pipeline(
            coins=target_coins,
            use_finbert=not args.no_finbert,
            run_eval=True,
            cryptopanic_key=args.api_key,
        )
