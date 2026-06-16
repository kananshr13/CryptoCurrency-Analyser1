"""
Streamlit Dashboard — CryptoCurrency Sentiment Analyser
========================================================
Interactive web UI that runs the full NLP pipeline in real-time.

Run:
    streamlit run dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from data.ingestion import CoinGeckoClient, NewsIngestion, TRACKED_COINS
from models.sentiment_model import CryptoSentimentAnalyser

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Sentiment Analyser",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b; border-radius: 8px; padding: 12px; }
    .signal-pill {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

SIGNAL_COLORS = {
    "Strong Buy": "#00ff88", "Buy": "#22c55e",
    "Hold": "#facc15", "Sell": "#f87171", "Strong Sell": "#ff3366",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Configurations")
    use_finbert = st.toggle("Advanced Model", value=False,
                            help="Downloads ~500MB model on first run. VADER is instant.")
    selected_coins = st.multiselect(
        "Track coins",
        options=[c["name"] for c in TRACKED_COINS],
        default=["Bitcoin", "Ethereum", "Solana", "BNB", "XRP"],
    )
    cryptopanic_key = st.text_input("CryptoPanic API Key (optional)", type="password",
                                    help="Get free key at cryptopanic.com for live news")
    refresh = st.button("Refresh Data", use_container_width=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_analyser(use_finbert):
    return CryptoSentimentAnalyser(use_finbert=use_finbert)

@st.cache_data(ttl=60, show_spinner="Fetching live prices...")
def load_prices(coin_ids):
    try:
        return CoinGeckoClient().get_prices(coin_ids)
    except Exception:
        return pd.DataFrame()

def get_coin_data(coin, analyser, news_client):
    news = news_client.get_news(coin["name"], coin["symbol"])
    texts = [f"{n['title']} {n['body']}" for n in news]
    results = analyser.analyse_batch(texts)
    sentiment = analyser.aggregate(results)
    return {"coin": coin, "news": news, "results": results, "sentiment": sentiment}

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("Crypto Market Sentiment Intelligence")
st.markdown("""Real-time sentiment monitoring and market intelligence for major cryptocurrencies using NLP models and live market data.""")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  |  "
           f"Model: {'VADER + FinBERT Ensemble' if use_finbert else 'VADER (fast mode)'}")

target = [c for c in TRACKED_COINS if c["name"] in selected_coins]
if not target:
    st.warning("Select at least one coin in the sidebar.")
    st.stop()

analyser    = load_analyser(use_finbert)
news_client = NewsIngestion(cryptopanic_api_key=cryptopanic_key or None)
prices_df   = load_prices([c["id"] for c in target])

with st.spinner("Running NLP sentiment pipeline..."):
    coin_data = [get_coin_data(c, analyser, news_client) for c in target]

# ── Summary metrics row ────────────────────────────────────────────────────────
st.subheader("Market Overview")
cols = st.columns(len(coin_data))
for col, cd in zip(cols, coin_data):
    s = cd["sentiment"]
    coin = cd["coin"]
    price_row = prices_df[prices_df["id"] == coin["id"]] if not prices_df.empty else pd.DataFrame()
    price     = f"${float(price_row['price_usd'].iloc[0]):,.2f}" if not price_row.empty else "N/A"
    chg       = f"{float(price_row['change_24h_pct'].iloc[0]):+.1f}%" if not price_row.empty else ""
    color     = SIGNAL_COLORS.get(s["signal"], "#94a3b8")
    with col:
        st.metric(
            label=f"{coin['symbol']}  {coin['name']}",
            value=price,
            delta=chg,
        )
        st.markdown(
            f'<span class="signal-pill" style="background:{color}22;color:{color};border:1px solid {color}">'
            f'{s["signal"]}  {s["score"]:+.3f}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {s['confidence']:.0%}  |  {s['n_articles']} articles")

# ── Horizontal sentiment bar chart ────────────────────────────────────────────
st.markdown("---")
st.subheader("Sentiment Comparison")

chart_df = pd.DataFrame([
    {
        "Coin": cd["coin"]["symbol"],
        "Score": cd["sentiment"]["score"],
        "Signal": cd["sentiment"]["signal"],
        "Color": SIGNAL_COLORS.get(cd["sentiment"]["signal"], "#94a3b8"),
    }
    for cd in coin_data
]).sort_values("Score", ascending=True)

fig = go.Figure(go.Bar(
    x=chart_df["Score"], y=chart_df["Coin"],
    orientation="h",
    marker_color=chart_df["Color"],
    text=chart_df["Signal"],
    textposition="auto",
))
fig.update_layout(
    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
    font_color="#f1f5f9", height=max(300, len(coin_data) * 55),
    xaxis=dict(range=[-1, 1], gridcolor="#334155", title="Sentiment Score"),
    yaxis=dict(gridcolor="#334155"),
    margin=dict(l=10, r=10, t=10, b=10),
)
fig.add_vline(x=0, line_color="#64748b", line_width=1)
st.plotly_chart(fig, use_container_width=True)

# ── Per-coin deep dive ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Detailed Analysis")
tab_names = [cd["coin"]["symbol"] for cd in coin_data]
tabs = st.tabs(tab_names)

for tab, cd in zip(tabs, coin_data):
    with tab:
        s = cd["sentiment"]
        col1, col2 = st.columns([1, 2])

        with col1:
            # Gauge chart
            
            st.metric("Sentiment Score",f"{s['score']:+.3f}")
            st.progress((s["score"] + 1) / 2)
            st.metric("Signal",     s["signal"])
            st.metric("Confidence", f"{s['confidence']:.0%}")
            st.metric("Positive",   f"{s['positive_pct']:.0f}%")
            st.metric("Negative",   f"{s['negative_pct']:.0f}%")
            if s["keywords"]:
                st.caption("Top keywords: " + ", ".join(s["keywords"][:6]))

        with col2:
            st.markdown("**News Headlines & Sentiment**")
            for news_item, result in zip(cd["news"][:8], cd["results"][:8]):
                color = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"}[result.label]
                st.markdown(
                    f'<div style="border-left:3px solid {color};padding:6px 12px;margin:6px 0;'
                    f'background:#1e293b;border-radius:0 6px 6px 0">'
                    f'<b style="color:{color}">{result.label} ({result.ensemble_score:+.2f})</b> '
                    f'— <span style="color:#cbd5e1;font-size:13px">{news_item["title"]}</span>'
                    f'<br><span style="color:#64748b;font-size:11px">'
                    f'{news_item["source"]} · {news_item["published_at"][:10]}</span></div>',
                    unsafe_allow_html=True,
                )

# ── Text input for live analysis ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Live Sentiment Analysis ")
user_text = st.text_area(
    "Paste any crypto headline or text:",
    placeholder="e.g. Bitcoin ETF approved — massive institutional adoption expected...",
    height=100,
)
if user_text.strip():
    result = analyser.analyse(user_text)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Label",          result.label)
    c2.metric("Ensemble Score", f"{result.ensemble_score:+.3f}")
    c3.metric("VADER Score",    f"{result.vader_compound:+.3f}")
    c4.metric("Signal",         result.signal)
    if result.keywords:
        st.info(f" Keywords detected: {', '.join(result.keywords)}")
st.markdown("---")

st.caption(
    "Built by Kanan Sharma | Computer Science Engineering Student"
)

st.caption(
    "Python • NLP • Streamlit • FinBERT • VADER • CoinGecko API"
)