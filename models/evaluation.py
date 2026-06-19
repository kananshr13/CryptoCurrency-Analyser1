#created for evaluation of the sentiments from news headlines fetched from the web
import logging  
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class: dict        # precision/recall/f1 per label
    confusion_matrix: list
    n_samples: int
    model_name: str

    def print_summary(self):
        print(f"\n{'='*55}")
        print(f"  Evaluation Report — {self.model_name}")
        print(f"{'='*55}")
        print(f"  Samples    : {self.n_samples}")
        print(f"  Accuracy   : {self.accuracy:.1%}")
        print(f"  F1 (macro) : {self.f1_macro:.1%}")
        print(f"  F1 (wtd)   : {self.f1_weighted:.1%}")
        print(f"\n  Per-class metrics:")
        for cls, m in self.per_class.items():
            print(f"    {cls:<12} P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1-score']:.2f}  n={m['support']}")
        print(f"\n  Confusion Matrix:")
        labels = list(self.per_class.keys())
        df = pd.DataFrame(
            self.confusion_matrix,
            index=[f"True:{l}" for l in labels],
            columns=[f"Pred:{l}" for l in labels],
        )
        print(df.to_string())
        print(f"{'='*55}\n")


def evaluate_model(
    analyser,
    labeled_data: list[dict],
    model_name: str = "CryptoSentimentAnalyser",
) -> EvaluationReport: #evaluates the sentiment on the labelled examples where labelled data is the list of strings which are marked as positive, negative or neutral
    y_true, y_pred = [], []

    for item in labeled_data:
        result = analyser.analyse(item["text"])
        y_true.append(item["label"])
        y_pred.append(result.label)

    labels = ["Positive", "Negative", "Neutral"]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return EvaluationReport(
        accuracy=accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        per_class={k: v for k, v in report.items() if k in labels},
        confusion_matrix=cm,
        n_samples=len(labeled_data),
        model_name=model_name,
    )


def build_labeled_dataset() -> list[dict]: #ground truth labelled dataset for the evaluation 

    return [
        # Positive
        {"text": "Bitcoin ETF approval drives massive institutional adoption and record inflows", "label": "Positive"},
        {"text": "Ethereum network upgrade boosts throughput and reduces gas fees significantly", "label": "Positive"},
        {"text": "Solana rallies 20% on partnership announcement with global payment platform", "label": "Positive"},
        {"text": "Long-term holders accumulating Bitcoin — bullish on-chain signal detected", "label": "Positive"},
        {"text": "Crypto market surges as Fed signals rate cut boosting risk appetite", "label": "Positive"},
        {"text": "DeFi protocol hits all-time high in TVL as user adoption accelerates", "label": "Positive"},
        {"text": "Institutional investors rebound confidence in crypto after regulatory clarity", "label": "Positive"},
        {"text": "BNB breakout above resistance milestone signals strong momentum continuation", "label": "Positive"},
        # Negative
        {"text": "SEC sues major exchange in regulatory crackdown sending crypto crashing", "label": "Negative"},
        {"text": "Exchange hack drains $200M as vulnerability exploited by sophisticated attackers", "label": "Negative"},
        {"text": "Bitcoin plunges 15% amid liquidation cascade and bearish macro outlook", "label": "Negative"},
        {"text": "China announces ban on crypto trading triggering massive market sell-off", "label": "Negative"},
        {"text": "DeFi protocol collapse triggers fraud investigation and user fund losses", "label": "Negative"},
        {"text": "Crypto bankruptcy filing sparks contagion fears across the market", "label": "Negative"},
        {"text": "Stablecoin loses peg amid market panic causing widespread liquidations", "label": "Negative"},
        {"text": "XRP lawsuit ruling raises concerns about securities classification for altcoins", "label": "Negative"},
        # Neutral
        {"text": "Bitcoin trading volume remains stable as market awaits clearer direction", "label": "Neutral"},
        {"text": "Analysts divided on crypto short-term outlook amid mixed signals", "label": "Neutral"},
        {"text": "Ethereum developers outline roadmap for next six months of protocol work", "label": "Neutral"},
        {"text": "Crypto market consolidates after recent price movements", "label": "Neutral"},
        {"text": "Survey shows increasing retail awareness of cryptocurrency as asset class", "label": "Neutral"},
        {"text": "Crypto exchange announces scheduled maintenance downtime for system upgrades", "label": "Neutral"},
    ]


def compare_vader_vs_ensemble(analyser, labeled_data: list[dict]): #compares the VADER model alone with the full ensemble model and also demonstrates the value of adding FinBERT to the pipleine
    print("\n📊 VADER vs Ensemble Comparison")
    print(f"{'Text':<55} {'True':<10} {'VADER':<10} {'Ensemble':<12} {'Match?'}")
    print("-" * 105)

    vader_correct = 0
    ensemble_correct = 0

    for item in labeled_data[:12]:  # show first 12
        result = analyser.analyse(item["text"])
        # Reconstruct VADER-only label
        from models.sentiment_model import CryptoSentimentAnalyser
        vader_label = CryptoSentimentAnalyser._score_to_label(result.vader_compound)
        ensemble_label = result.label
        true_label = item["label"]

        v_ok = "✓" if vader_label == true_label else "✗"
        e_ok = "✓" if ensemble_label == true_label else "✗"
        if vader_label    == true_label: vader_correct += 1
        if ensemble_label == true_label: ensemble_correct += 1

        text_short = item["text"][:52] + "..." if len(item["text"]) > 52 else item["text"]
        print(f"{text_short:<55} {true_label:<10} {vader_label:<10} {ensemble_label:<12} V:{v_ok} E:{e_ok}")

    total = len(labeled_data[:12])
    print(f"\n  VADER accuracy (sample)   : {vader_correct/total:.0%}")
    print(f"  Ensemble accuracy (sample): {ensemble_correct/total:.0%}")
