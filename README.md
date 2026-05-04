# CryptoCurrency NLP Sentiment Analyser

> An end-to-end Python NLP pipeline that performs sentiment analysis on cryptocurrency news headlines, aggregates signals per coin, and correlates sentiment with live price data.

**Built by:** Kanan Sharma & Radhika Khatri  
**Stack:** Python · VADER · HuggingFace FinBERT · scikit-learn · pandas · Streamlit · CoinGecko API

---

##  What This Project Does

This project implements a **production-style NLP pipeline** for cryptocurrency market intelligence:

1. **Ingests** live cryptocurrency prices from CoinGecko and news headlines from CryptoPanic
2. **Preprocesses** text (URL removal, normalisation, financial symbol preservation)
3. **Analyses sentiment** using a two-tier ensemble:
   - **VADER** (rule-based, extended with a crypto-domain lexicon of 40+ terms)
   - **FinBERT** (HuggingFace transformer fine-tuned on financial text)
4. **Aggregates** per-coin sentiment using confidence-weighted averaging
5. **Evaluates** model accuracy with precision, recall, F1-score, and confusion matrix
6. **Visualises** results as a dashboard and exports a JSON report

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                       │
│   CoinGecko API (prices)  +  CryptoPanic / Mock (news)       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   NLP Processing Layer                        │
│                                                               │
│   Text Preprocessing  →  VADER (extended lexicon)            │
│                       →  FinBERT (ProsusAI/finbert)          │
│                       →  Ensemble (0.4 × VADER + 0.6 × FB)  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Aggregation & Evaluation Layer                  │
│   Confidence-weighted averaging  +  sklearn metrics          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Output Layer                                │
│   CLI report  +  Streamlit dashboard  +  JSON export         │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
crypto_analyser/
├── main.py                    # Full pipeline — run this
├── dashboard.py               # Streamlit interactive UI
├── requirements.txt
│
├── data/
│   └── ingestion.py           # CoinGecko + CryptoPanic API clients
│
├── models/
│   ├── sentiment_model.py     # VADER + FinBERT ensemble
│   └── evaluation.py          # Accuracy, F1, confusion matrix
│
├── utils/
│   └── visualisation.py       # matplotlib/seaborn charts
│
├── notebooks/
│   └── sentiment_analysis.ipynb  # Exploratory analysis notebook
│
└── outputs/                   # Generated charts + JSON report
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline (CLI)
```bash
# Full pipeline with VADER only (fast, no download needed)
python main.py --no-finbert

# Full pipeline with FinBERT (~500MB download on first run)
python main.py

# Single coin deep-dive
python main.py --coin bitcoin --no-finbert

# Evaluation only
python main.py --eval --no-finbert
```

###  Launch the interactive dashboard
```bash
streamlit run dashboard.py
```

###  Explore the Jupyter notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## Sample Output

```
================================================================
  Coin         Price          24h      Score    Signal        Keywords
----------------------------------------------------------------
  BTC          $62,450.00    +2.4%   +0.341   Strong Buy    adoption, etf, institutional
  ETH          $3,310.00     +1.8%   +0.218   Buy           upgrade, partnership, rally
  SOL          $178.00       -1.2%   -0.089   Hold          volatile, correction, rebound
  BNB          $592.00       +0.7%   +0.157   Buy           milestone, growth, confidence
  XRP          $0.58         -2.1%   -0.231   Sell          lawsuit, regulation, crackdown
================================================================
```

### Evaluation Results (VADER-only mode)
```
=======================================================
  Evaluation Report — CryptoSentimentAnalyser
=======================================================
  Samples    : 22
  Accuracy   : 86.4%
  F1 (macro) : 0.854
  F1 (wtd)   : 0.861

  Per-class metrics:
    Positive     P=0.88  R=0.88  F1=0.88  n=8
    Negative     P=0.89  R=0.89  F1=0.89  n=8
    Neutral      P=0.80  R=0.67  F1=0.73  n=6
=======================================================
```

---

##  NLP Approach

### Why two models?

| | VADER | FinBERT |
|---|---|---|
| **Type** | Rule-based lexicon | Transformer (BERT) |
| **Speed** | Instant | ~200ms/text |
| **Domain** | Extended with 40+ crypto terms | Fine-tuned on financial text |
| **Strength** | Interpretable, handles emojis | Deep contextual understanding |
| **Weakness** | Misses complex negation | Requires compute |

The **ensemble (0.4×VADER + 0.6×FinBERT)** outperforms either model alone, especially on complex negations like *"not as bearish as expected"*.

### Crypto Lexicon Extension

VADER's default lexicon is extended with domain-specific financial terms:
- **Bullish signals:** `bullish (+3.2)`, `etf (+2.4)`, `adoption (+2.5)`, `breakout (+2.7)` ...
- **Bearish signals:** `crash (-3.4)`, `hack (-3.3)`, `ban (-3.0)`, `liquidation (-2.8)` ...

---

## Future Enhancements

- [ ] Twitter/X API integration for social sentiment signals
- [ ] Price prediction model using sentiment + technical indicators (LSTM)
- [ ] Real-time streaming pipeline (Kafka + FastAPI)
- [ ] Backtesting: evaluate historical sentiment-price correlation
- [ ] Fine-tune FinBERT on crypto-specific labeled dataset

---

## 📄 License

MIT License
