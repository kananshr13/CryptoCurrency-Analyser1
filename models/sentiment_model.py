
# Sentiment Analysis Engine for Cryptocurrency News
#Uses a two-tier NLP approach:
#1. VADER (rule-based) for faster, domain-adapted for financial text
#2. HuggingFace FinBERT (transformer-based) for deep contextual understanding


import re
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Financial domain lexicon extension for VADER 
CRYPTO_LEXICON = {
    # Bullish signals
    "bullish": 3.2, "rally": 2.8, "surge": 2.9, "soar": 3.0, "breakout": 2.7,
    "adoption": 2.5, "partnership": 2.3, "upgrade": 2.1, "milestone": 2.4,
    "accumulate": 2.3, "institutional": 2.2, "etf": 2.4, "rebound": 2.2,
    "halving": 2.0, "staking": 1.8, "defi": 1.6, "ath": 2.5, "moon": 1.5,
    # Bearish signals
    "bearish": -3.1, "crash": -3.4, "plunge": -3.2, "dump": -2.8,
    "ban": -3.0, "hack": -3.3, "fraud": -3.5, "scam": -3.4, "lawsuit": -2.7,
    "crackdown": -2.9, "liquidation": -2.8, "bankruptcy": -3.3,
    "exploit": -3.1, "vulnerability": -2.5, "breach": -2.9, "collapse": -3.3,
    "delisting": -2.6, "rug": -3.4, "ponzi": -3.5,
}


@dataclass
class SentimentResult:
    """Structured output from the sentiment analysis pipeline."""
    text: str
    vader_compound: float        # VADER score: -1 to +1
    finbert_score: Optional[float]  # FinBERT score: -1 to +1 (None if unavailable)
    ensemble_score: float        # Weighted ensemble of both models
    label: str                   # "Positive" | "Negative" | "Neutral"
    confidence: float            # 0.0 – 1.0
    keywords: list[str]          # Matched domain keywords
    signal: str                  # "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class CryptoSentimentAnalyser:
    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert
        self._vader = None
        self._finbert = None
        self._load_models()

    def _load_models(self):
        """Load NLP models with graceful fallback."""
        # Load VADER with crypto domain extension
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self._vader.lexicon.update(CRYPTO_LEXICON)
            logger.info("✓ VADER loaded with %d crypto-domain terms", len(CRYPTO_LEXICON))
        except ImportError:
            logger.warning("vaderSentiment not installed — pip install vaderSentiment")

        # Load the FinBERT model (financial BERT fine-tuned on financial phrasebank)
        if self.use_finbert:
            try:
                from transformers import pipeline
                self._finbert = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    top_k=None,         # return all class scores
                    truncation=True,
                    max_length=512,
                )
                logger.info("✓ FinBERT loaded (ProsusAI/finbert)")
            except Exception as e:
                logger.warning("FinBERT unavailable (%s) — using VADER only", e)

    #Preprocessing

    def _preprocess(self, text: str) -> str:
        """Clean text for NLP processing."""
        text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
        text = re.sub(r"@\w+|#\w+", "", text)               # remove mentions/tags
        text = re.sub(r"[^\w\s\$\.\,\!\?\'%-]", " ", text)  # keep financial symbols
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_keywords(self, text: str) -> list[str]:
        """Find crypto-domain signal words present in the text."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t in CRYPTO_LEXICON]

    # Individual model scorers 

    def _vader_score(self, text: str) -> float:
        """Return VADER compound score in [-1, 1]."""
        if self._vader is None:
            return 0.0
        score = self._vader.polarity_scores(text)["compound"]
        print("\n===================")
        print("TEXT:",text)
        print("SCORE:",score)
        print("===================")
        
        return score 

    def _finbert_score(self, text: str) -> Optional[float]:
        """
        Return FinBERT score mapped to [-1, 1].
        FinBERT outputs: positive / negative / neutral with probabilities.
        """
        if self._finbert is None:
            return None
        try:
            results = self._finbert(text[:512])[0]  # truncate to model max
            scores = {r["label"].lower(): r["score"] for r in results}
            # Map to scalar: positive - negative (neutral dampens extremes)
            return scores.get("positive", 0) - scores.get("negative", 0)
        except Exception as e:
            logger.debug("FinBERT inference error: %s", e)
            return None

    #Ensemble & labelling

    def _ensemble(self, vader: float, finbert: Optional[float]):
        """Weighted ensemble → (score, confidence)."""
        if finbert is not None:
            score = 0.4 * vader + 0.6 * finbert
            agreement = 1 - abs(vader - finbert)/2
            confidence = max(0.30, min(0.95, agreement))
        else:
            score = vader
            confidence = max(0.30, min(0.85, abs(vader)))
        return round(score, 4), round(confidence, 4)

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score > 0.05:  return "Positive"
        if score < -0.05: return "Negative"
        return "Neutral"

    @staticmethod
    def _score_to_signal(score: float) -> str:
        if score >= 0.2:  
            return "Bullish"
        elif score <= -0.2:
            return "Bearish"
        else:
            return "Neutral"

    #Public API

    def analyse(self, text: str) -> SentimentResult:
        """Analyse a single piece of text and return a structured SentimentResult."""
        clean = self._preprocess(text)
        vader  = self._vader_score(clean)
        finbert = self._finbert_score(clean)
        score, confidence = self._ensemble(vader, finbert)
        keywords = self._extract_keywords(clean)

        return SentimentResult(
            text=text[:200],
            vader_compound=round(vader, 4),
            finbert_score=round(finbert, 4) if finbert is not None else None,
            ensemble_score=score,
            label=self._score_to_label(score),
            confidence=confidence,
            keywords=keywords,
            signal=self._score_to_signal(score),
        )

    def analyse_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyse a list of texts (e.g. all headlines for one coin)."""
        return [self.analyse(t) for t in texts]

    def aggregate(self, results: list[SentimentResult]) -> dict:
        
        #Aggregate multiple SentimentResults into a single coin-level summary.
        #Uses confidence-weighted averaging higher confidence predictions count more.
        
        if not results:
            return {"score": 0.0, "label": "Neutral", "signal": "Hold", "n": 0}

        weights = np.array([r.confidence for r in results])
        scores  = np.array([r.ensemble_score for r in results])
        weighted_score = float(np.average(scores, weights=weights))
        all_keywords = list({kw for r in results for kw in r.keywords})

        return {
            "score": round(weighted_score, 4),
            "label": self._score_to_label(weighted_score),
            "signal": self._score_to_signal(weighted_score),
            "confidence": round(float(weights.mean()), 4),
            "n_articles": len(results),
            "keywords": all_keywords[:10],
            "positive_pct": round(sum(1 for r in results if r.label == "Positive") / len(results) * 100, 1),
            "negative_pct": round(sum(1 for r in results if r.label == "Negative") / len(results) * 100, 1),
            "neutral_pct":  round(sum(1 for r in results if r.label == "Neutral")  / len(results) * 100, 1),
        }
