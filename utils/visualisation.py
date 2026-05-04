"""
Visualisation Module
====================
Generates charts for:
  - Sentiment distribution per coin
  - Sentiment vs. price correlation
  - Sentiment over time (trend analysis)
  - Signal dashboard summary

Uses matplotlib + seaborn. All charts save to /outputs/ and can
be embedded in reports or a Streamlit dashboard.
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)
os.makedirs("outputs", exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "Positive": "#22c55e",
    "Negative": "#ef4444",
    "Neutral":  "#94a3b8",
    "bg":       "#0f172a",
    "surface":  "#1e293b",
    "text":     "#f1f5f9",
    "accent":   "#38bdf8",
    "grid":     "#334155",
}

SIGNAL_COLORS = {
    "Strong Buy":  "#00ff88",
    "Buy":         "#22c55e",
    "Hold":        "#facc15",
    "Sell":        "#f87171",
    "Strong Sell": "#ff3366",
}

def _dark_fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["surface"])
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.grid(color=COLORS["grid"], linestyle="--", linewidth=0.5, alpha=0.7)
    return fig, ax


def plot_sentiment_dashboard(coin_summaries: list[dict], save_path: str = "outputs/sentiment_dashboard.png"):
    """
    Bar chart showing the sentiment score and signal for each tracked coin.
    This is the primary summary chart — what a stakeholder would see first.
    """
    coins   = [c["name"] for c in coin_summaries]
    scores  = [c["sentiment"]["score"] for c in coin_summaries]
    signals = [c["sentiment"]["signal"] for c in coin_summaries]
    bar_colors = [SIGNAL_COLORS.get(s, COLORS["Neutral"]) for s in signals]

    fig, ax = _dark_fig(figsize=(14, 6))
    bars = ax.barh(coins, scores, color=bar_colors, edgecolor="none", height=0.6)

    # Add signal labels inside bars
    for bar, signal, score in zip(bars, signals, scores):
        x = score + (0.02 if score >= 0 else -0.02)
        ha = "left" if score >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{signal}  ({score:+.2f})", va="center", ha=ha,
                color=COLORS["text"], fontsize=9, fontweight="bold")

    ax.axvline(0, color=COLORS["text"], linewidth=1.2, alpha=0.6)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Ensemble Sentiment Score  ←  Bearish  |  Bullish  →", fontsize=11)
    ax.set_title("🪙 CryptoCurrency Sentiment Dashboard", fontsize=14, fontweight="bold", pad=15)

    # Legend
    legend_items = [mpatches.Patch(color=v, label=k) for k, v in SIGNAL_COLORS.items()]
    ax.legend(handles=legend_items, loc="lower right", framealpha=0.2,
              labelcolor=COLORS["text"], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)
    return save_path


def plot_sentiment_vs_price(
    coin_name: str,
    price_series: list[float],
    sentiment_series: list[float],
    timestamps: list[str],
    save_path: str = None,
):
    """
    Dual-axis chart overlaying price movement with sentiment score over time.
    This is the core analytical output — shows if sentiment leads/lags price.
    """
    save_path = save_path or f"outputs/{coin_name.lower()}_sentiment_vs_price.png"
    fig, ax1 = plt.subplots(figsize=(14, 6), facecolor=COLORS["bg"])
    ax1.set_facecolor(COLORS["surface"])

    # Price line
    ax1.plot(timestamps, price_series, color=COLORS["accent"], linewidth=2.5,
             label=f"{coin_name} Price (USD)", zorder=3)
    ax1.fill_between(timestamps, price_series,
                     alpha=0.1, color=COLORS["accent"])
    ax1.set_ylabel(f"{coin_name} Price (USD)", color=COLORS["accent"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax1.tick_params(axis="x", labelcolor=COLORS["text"], rotation=45, labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor(COLORS["grid"])

    # Sentiment line on secondary axis
    ax2 = ax1.twinx()
    ax2.set_facecolor(COLORS["surface"])
    sentiment_colors = [
        COLORS["Positive"] if s > 0.05 else COLORS["Negative"] if s < -0.05 else COLORS["Neutral"]
        for s in sentiment_series
    ]
    ax2.bar(timestamps, sentiment_series, color=sentiment_colors, alpha=0.5,
            width=0.8, label="Sentiment Score", zorder=2)
    ax2.axhline(0, color=COLORS["text"], linewidth=0.8, alpha=0.4)
    ax2.set_ylabel("Sentiment Score", color=COLORS["text"], fontsize=11)
    ax2.tick_params(axis="y", labelcolor=COLORS["text"])
    ax2.set_ylim(-1, 1)
    for spine in ax2.spines.values():
        spine.set_edgecolor(COLORS["grid"])

    ax1.set_title(f"📈 {coin_name} — Sentiment vs. Price Correlation",
                  fontsize=13, fontweight="bold", color=COLORS["text"], pad=12)
    ax1.grid(color=COLORS["grid"], linestyle="--", linewidth=0.4, alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               framealpha=0.2, labelcolor=COLORS["text"], fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)
    return save_path


def plot_label_distribution(coin_name: str, results: list, save_path: str = None):
    """Pie chart of Positive / Negative / Neutral distribution for a coin's news."""
    save_path = save_path or f"outputs/{coin_name.lower()}_label_dist.png"
    labels_list = [r.label for r in results]
    counts = {l: labels_list.count(l) for l in ["Positive", "Neutral", "Negative"]}

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=COLORS["bg"])
    wedges, texts, autotexts = ax.pie(
        counts.values(),
        labels=counts.keys(),
        colors=[COLORS[l] for l in counts.keys()],
        autopct="%1.0f%%",
        startangle=140,
        wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
    )
    for t in texts + autotexts:
        t.set_color(COLORS["text"])
    ax.set_title(f"{coin_name} — News Sentiment Distribution",
                 color=COLORS["text"], fontsize=12, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)
    return save_path


def plot_confusion_matrix(cm: list, labels: list, save_path: str = "outputs/confusion_matrix.png"):
    """Heatmap of the model evaluation confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=COLORS["bg"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=1, linecolor=COLORS["bg"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_facecolor(COLORS["surface"])
    ax.tick_params(colors=COLORS["text"])
    ax.set_xlabel("Predicted Label", color=COLORS["text"], fontsize=11)
    ax.set_ylabel("True Label",      color=COLORS["text"], fontsize=11)
    ax.set_title("Model Evaluation — Confusion Matrix",
                 color=COLORS["text"], fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)
    return save_path
