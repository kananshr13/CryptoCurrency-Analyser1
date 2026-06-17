# Crypto Market Sentiment Intelligence Platform

A real-time cryptocurrency sentiment analysis dashboard that combines live market data, multi-source crypto news aggregation, and Natural Language Processing (NLP) to provide actionable market intelligence.

## Live Demo

https://crypto-sentiment-analyser.streamlit.app

## Features

### Real-Time Market Data

* Live cryptocurrency prices powered by CoinGecko API
* 24-hour price change tracking
* 7-day price trend visualization
* Support for major cryptocurrencies including Bitcoin, Ethereum, Solana, BNB, and XRP

### Multi-Source News Aggregation

* CoinDesk RSS feeds
* CoinTelegraph RSS feeds
* Decrypt RSS feeds
* CryptoSlate RSS feeds
* Automatic article aggregation and filtering by cryptocurrency

### NLP-Powered Sentiment Analysis

* VADER sentiment analysis with custom cryptocurrency lexicon
* Optional FinBERT transformer-based financial sentiment model
* Confidence scoring mechanism
* Sentiment classification:

  * Bullish
  * Neutral
  * Bearish

### Interactive Analytics Dashboard

* Market-wide sentiment overview
* Coin-specific sentiment analysis
* News sentiment breakdown
* Source diversity tracking
* Live sentiment testing for custom headlines

## Dashboard Preview

Add screenshots here.

### Market Overview

Displays:

* Live cryptocurrency prices
* Sentiment scores
* Confidence metrics
* News article volume
* Source coverage

### Detailed Analysis

For each cryptocurrency:

* Sentiment score
* Confidence level
* Positive vs Negative article distribution
* 7-day price trend
* News headline analysis

## Technology Stack

### Backend

* Python
* Pandas
* NumPy

### NLP & Machine Learning

* VADER Sentiment Analysis
* FinBERT
* Scikit-Learn

### Data Sources

* CoinGecko API
* RSS News Feeds

### Visualization

* Streamlit
* Plotly

## Project Architecture

```text
RSS News Sources
        │
        ▼
News Aggregation Layer
        │
        ▼
NLP Pipeline
(VADER + FinBERT)
        │
        ▼
Sentiment Engine
        │
        ▼
Analytics Dashboard
        │
        ▼
Interactive User Interface
```

## Installation

Clone the repository:

```bash
git clone https://github.com/kananshr13/CryptoCurrency-Analyser1.git
cd CryptoCurrency-Analyser1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run dashboard.py
```

## Use Cases

* Cryptocurrency market monitoring
* Sentiment-driven investment research
* News impact analysis
* Financial NLP experimentation
* Data visualization and analytics

## Future Improvements

* Sentiment history tracking
* Historical trend analysis
* Advanced forecasting models
* Portfolio monitoring
* Additional cryptocurrency support
* Alert and notification system

## Author

Kanan Sharma

GitHub: https://github.com/kananshr13

## License

This project is intended for educational and portfolio purposes.

Market data and news content belong to their respective providers.

Disclaimer: This application is for informational and educational purposes only and does not constitute financial advice.
