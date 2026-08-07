import os
import re
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from groq import Groq
import logging
from typing import Optional, Dict, Any, Tuple
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import yfinance as yf

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
    RSI_PERIOD = 14
    SMA_FAST = 20
    SMA_SLOW = 50
    MAX_HISTORY_DAYS = 365

st.set_page_config(
    page_title="BISTeknik PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 🎨 CSS — TEMİZ, OKUNUR, PROFESYONEL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #0b0f14;
    --panel: #111821;
    --panel2: #151e29;
    --border: #263646;
    --text: #e7edf4;
    --muted: #8b9aac;
    --blue: #4da3ff;
}

.stApp {
    background: var(--bg);
}

* {
    font-family: 'Inter', sans-serif !important;
}

.stApp, body {
    color: var(--text) !important;
}

.block-container {
    max-width: 1440px;
    padding: 2rem 3rem 4rem;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    background: none !important;
    -webkit-text-fill-color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

h1 { font-size: 2.15rem !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.2rem !important; }

.stMarkdown p, label, .stCaption {
    color: var(--muted) !important;
}

/* LOGO */
.logo-container {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    animation: none !important;
    padding: 0 !important;
    gap: 12px !important;
}

.logo-icon {
    width: 42px !important;
    height: 42px !important;
    border-radius: 8px !important;
    font-size: 24px !important;
    background: var(--blue) !important;
    box-shadow: none !important;
}

.logo-text {
    color: var(--text) !important;
    background: none !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size: 22px !important;
}

.logo-sub {
    color: var(--muted) !important;
    letter-spacing: 2px !important;
}

/* SEARCH */
.stTextInput input {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text) !important;
    font-size: 15px !important;
    padding: 12px 15px !important;
    box-shadow: none !important;
}

.stTextInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 1px var(--blue) !important;
}

.stTextInput input::placeholder {
    color: #667689 !important;
}

/* BUTTON */
.stButton button {
    min-height: 44px !important;
    background: var(--blue) !important;
    border: 0 !important;
    border-radius: 7px !important;
    color: #07111d !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    animation: none !important;
}

.stButton button:hover {
    background: #78baff !important;
    transform: none !important;
}

.stButton button * {
    color: #07111d !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 15px 17px !important;
    box-shadow: none !important;
}

[data-testid="stMetric"]:hover {
    transform: none !important;
    border-color: #38516b !important;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.73rem !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.45rem !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: var(--panel) !important;
    border-bottom: 1px solid var(--border);
    padding: 4px 5px 0;
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    border-radius: 5px 5px 0 0 !important;
    padding: 10px 16px !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--text) !important;
    background: var(--panel2) !important;
    border-bottom: 2px solid var(--blue) !important;
}

/* CHART */
[data-testid="stPlotlyChart"] {
    background: var(--panel) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 5px;
}

/* EXPANDER */
.streamlit-expanderHeader,
[data-testid="stExpander"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.streamlit-expanderContent {
    background: var(--panel) !important;
}

/* ALERT */
.stAlert {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
}

/* CHAT — SADECE BURASI SİYAH */
[data-testid="stChatMessage"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

[data-testid="stChatMessage"] * {
    color: var(--text) !important;
}

[data-testid="stChatInput"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
}

[data-testid="stChatInput"] input {
    color: var(--text) !important;
}

[data-testid="stChatInput"] button {
    background: var(--blue) !important;
    border-radius: 5px !important;
}

/* DIVIDER */
hr {
    border: 0 !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 1.25rem 0 !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: var(--bg);
}
::-webkit-scrollbar-thumb {
    background: #33465a;
    border-radius: 5px;
}

@keyframes pulse-glow {
    from, to { box-shadow: none; }
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.75rem 3rem;
    }
    h1 { font-size: 1.7rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 📍 YARDIMCI FONKSİYONLAR
# ============================================================

def sanitize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol in ["XU100", "BIST100", "BIST 100"]:
        return "^XU100"
    if not symbol.endswith(".IS") and not symbol.startswith("^"):
        if symbol.isalpha() and 3 <= len(symbol) <= 6:
            return f"{symbol}.IS"
    return symbol

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.where(loss != 0, 0.00001)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).clip(0, 100)

def get_rsi_comment(rsi_value):
    if rsi_value > 70:
        return "🔥 Aşırı Alım"
    elif rsi_value < 30:
        return "💎 Aşırı Satım"
    elif 40 <= rsi_value <= 60:
        return "⚖️ Nötr"
    elif rsi_value > 60:
        return "📈 Alım Bölgesi"
    else:
        return "📉 Satım Bölgesi"

# 📍 VERİ ÇEKME
@st.cache_data(ttl=300)
def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    clean_sym = sanitize_symbol(symbol)
    
    try:
        ticker = yf.Ticker(clean_sym)
        info = ticker.info
        
        if info and 'regularMarketPrice' in info:
            hist = ticker.history(period="1y")
            
            if not hist.empty:
                df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.index = pd.to_datetime(df.index)
                
                df['SMA20'] = df['Close'].rolling(20).mean()
                df['SMA50'] = df['Close'].rolling(50).mean()
                df['SMA200'] = df['Close'].rolling(200).mean()
                df['RSI'] = calculate_rsi(df['Close'])
                
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                last_price = float(df['Close'].iloc[-1])
                change = float(info.get('regularMarketChangePercent', 0))
                
                return {
                    "symbol": clean_sym,
                    "price": last_price,
                    "change": change,
                    "currency": "TRY",
                    "df": df,
                    "data_source": "Yahoo Finance"
                }
                
    except Exception as e:
        logger.error(f"Veri çekme hatası: {e}")
    
    return None

# 📍 HABER ÇEKME
@st.cache_data(ttl=300)
def fetch_news(symbol: str) -> str:
    clean_name = symbol.replace(".IS", "").replace("^", "")
    news_list = []
    
    try:
        url = f"https://news.google.com/search?q={clean_name}&hl=tr&gl=TR"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('a', class_='JtKRv')
            for item in headlines[:5]:
                title = item.get_text().strip()
                if title and len(title) > 10:
                    news_list.append(f"• {title}")
        
        if news_list:
            return "\n".join(news_list[:5])
        return "📡 Güncel haber bulunamadı."
            
    except Exception as e:
        return "📡 Haberler alınamıyor"

# 📍 RAKİP HİSSELER
def get_competitors(symbol: str) -> list:
    all_stocks = [
        "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "SISE.IS",
        "KCHOL.IS", "SAHOL.IS", "TUPRS.IS", "EREGL.IS", "PGSUS.IS"
    ]
    import random
    random.seed(hash(symbol))
    return random.sample([s for s in all_stocks if s != symbol], 3)

# 📍 AI ANALİZ
def analyze_with_ai(prompt: str, market_data: Dict, competitors_data: list, history: list, client) -> str:
    df = market_data['df']
    last_price = market_data['price']
    change = market_data['change']
    
    last_rsi = df['RSI'].iloc[-1] if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 50
    sma20 = df['SMA20'].iloc[-1] if 'SMA20' in df and not pd.isna(df['SMA20'].iloc[-1]) else last_price
    sma50 = df['SMA50'].iloc[-1] if 'SMA50' in df and not pd.isna(df['SMA50'].iloc[-1]) else last_price
    sma200 = df['SMA200'].iloc[-1] if 'SMA200' in df and not pd.isna(df['SMA200'].iloc[-1]) else last_price
    
    trend = "🚀 YÜKSELİŞ" if last_price > sma200 else "📉 DÜŞÜŞ" if last_price < sma200 else "➡️ YATAY"
    support = df['Low'].tail(50).min()
    resistance = df['High'].tail(50).max()
    
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5) if len(returns) > 0 else 0
    monthly_return = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100 if len(df) > 22 else 0
    yearly_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100 if len(df) > 0 else 0
    
    comp_text = ""
    if competitors_data:
        comp_text = "\n📊 **RAKİP KARŞILAŞTIRMA:**\n"
        for comp in competitors_data:
            if comp:
                comp_text += f"  • {comp['symbol']}: {comp['price']:.2f} (%{comp['change']:+.2f})\n"
    
    news = fetch_news(market_data['symbol'])
    
    data_str = f"""
📊 HİSSE ANALİZ VERİLERİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sembol: {market_data['symbol']}
Fiyat: {last_price:.2f} TRY
Günlük Değişim: %{change:+.2f}

📈 TEKNİK GÖSTERGELER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RSI(14): {last_rsi:.1f} ({get_rsi_comment(last_rsi)})
SMA20: {sma20:.2f} TRY
SMA50: {sma50:.2f} TRY
SMA200: {sma200:.2f} TRY
Trend: {trend}

🛡️ DESTEK/DİRENÇ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Destek: {support:.2f} TRY
Direnç: {resistance:.2f} TRY

📊 GETİRİ ANALİZİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Aylık Getiri: %{monthly_return:+.2f}
Yıllık Getiri: %{yearly_return:+.2f}
Volatilite: %{volatility*100:.2f}

{comp_text}

📰 GÜNCEL HABERLER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{news}
"""

    system_prompt = f"""
Sen, BISTeknik PRO yapay zeka analistisin.

📋 ANALİZ FORMATI (6 BAŞLIK):

1. ÖZET GÖRÜŞ - Hisse hakkında genel değerlendirme, kısa/orta/uzun vade beklenti
2. TEKNİK ANALİZ - RSI, SMA, MACD yorumu, destek/direnç seviyeleri
3. GETİRİ & RİSK - Aylık/Yıllık getiri, volatilite, risk/getiri potansiyeli
4. REKABET - Rakiplerle karşılaştırma, sektörel performans
5. HABER & GÜNDEM - Güncel haberlerin etkisi, makroekonomik faktörler
6. SONUÇ & STRATEJİ - Kısa/Orta/Uzun vade strateji, izlenecek seviyeler

⚠️ KURALLAR:
- Sadece GERÇEK VERİLERE dayan
- Yatırım tavsiyesi VERME
- Teknik terimleri doğru kullan
- Türkçe yaz

📊 VERİLER:
{data_str}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-3:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Analiz Hatası: {str(e)}"

# ============================================================
# 📱 ANA UYGULAMA
# ============================================================

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.text_input("🔑 Groq API Key:", type="password")

if not groq_api_key:
    st.warning("⚠️ Groq API Key girin")
    st.stop()

try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"❌ Client hatası: {e}")
    st.stop()

# LOGO
st.markdown("""
<div style="display:flex; justify-content:center; padding:20px 0 10px 0;">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:42px; height:42px; border-radius:8px; font-size:24px; background:#4da3ff; display:flex; align-items:center; justify-content:center;">📈</div>
        <div>
            <span style="color:#e7edf4; font-size:22px; font-weight:700;">BISTeknik PRO</span>
            <span style="color:#8b9aac; font-size:10px; letter-spacing:2px; margin-left:6px;">AI QUANT TERMINAL</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# SEARCH
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_input(
        "",
        placeholder="🔍 Hisse kodu girin... (Örnek: SASA, THYAO, GARAN)",
        label_visibility="collapsed",
        key="search_input"
    )

with col2:
    analyze_btn = st.button("🚀 ANALİZ ET", use_container_width=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "market_data" not in st.session_state:
    st.session_state.market_data = None

if "competitors" not in st.session_state:
    st.session_state.competitors = []

# ANALİZ
if analyze_btn and user_input:
    with st.spinner("🚀 Veriler çekiliyor..."):
        symbol = sanitize_symbol(user_input)
        market_data = fetch_market_data(symbol)
        
        if market_data:
            st.session_state.market_data = market_data
            
            comp_symbols = get_competitors(symbol)
            competitors = []
            for comp_sym in comp_symbols:
                comp_data = fetch_market_data(comp_sym)
                if comp_data:
                    competitors.append(comp_data)
            st.session_state.competitors = competitors
            
            prompt = f"{symbol} hissesi için detaylı profesyonel analiz yap."
            analysis = analyze_with_ai(
                prompt,
                market_data,
                competitors,
                st.session_state.messages,
                client
            )
            
            st.session_state.messages.append({"role": "assistant", "content": analysis})
            st.rerun()
        else:
            st.error(f"❌ {symbol} için veri bulunamadı")

# GÖSTERİM
if st.session_state.market_data:
    data = st.session_state.market_data
    df = data['df']
    
    # METRİKLER
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Fiyat",
            f"{data['price']:.2f} ₺",
            f"%{data['change']:+.2f}",
            delta_color="normal" if data['change'] >= 0 else "inverse"
        )
    
    with col2:
        last_rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
        st.metric("📊 RSI", f"{last_rsi:.1f}", get_rsi_comment(last_rsi))
    
    with col3:
        sma20 = df['SMA20'].iloc[-1] if 'SMA20' in df else 0
        st.metric("📈 SMA20", f"{sma20:.2f} ₺")
    
    with col4:
        sma50 = df['SMA50'].iloc[-1] if 'SMA50' in df else 0
        st.metric("📈 SMA50", f"{sma50:.2f} ₺")
    
    with col5:
        monthly_return = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100 if len(df) > 22 else 0
        st.metric("📆 Aylık", f"%{monthly_return:+.2f}")
    
    st.markdown("---")
    
    # GRAFİKLER
    tab1, tab2, tab3 = st.tabs(["📈 Fiyat & SMA", "📊 Teknik Göstergeler", "📉 Karşılaştırma"])
    
    with tab1:
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df.index[-90:],
            open=df['Open'].iloc[-90:],
            high=df['High'].iloc[-90:],
            low=df['Low'].iloc[-90:],
            close=df['Close'].iloc[-90:],
            name="Fiyat",
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index[-90:],
            y=df['SMA20'].iloc[-90:],
            mode='lines',
            name='SMA 20',
            line=dict(color='#f59e0b', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index[-90:],
            y=df['SMA50'].iloc[-90:],
            mode='lines',
            name='SMA 50',
            line=dict(color='#3b82f6', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index[-90:],
            y=df['SMA200'].iloc[-90:],
            mode='lines',
            name='SMA 200',
            line=dict(color='#a78bfa', width=1.5, dash='dash')
        ))
        
        fig.update_layout(
            title=f"<b>{data['symbol']}</b> — Fiyat ve Hareketli Ortalamalar",
            template="plotly_dark",
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="rgba(255,255,255,0.7)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.6, 0.4])
        
        fig.add_trace(go.Scatter(
            x=df.index[-90:],
            y=df['MACD'].iloc[-90:],
            mode='lines',
            name='MACD',
            line=dict(color='#60a5fa', width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index[-90:],
            y=df['Signal'].iloc[-90:],
            mode='lines',
            name='Sinyal',
            line=dict(color='#f59e0b', width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index[-90:],
            y=df['RSI'].iloc[-90:],
            mode='lines',
            name='RSI',
            line=dict(color='#a78bfa', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.3, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", opacity=0.3, row=2, col=1)
        
        fig.update_layout(
            title=f"<b>{data['symbol']}</b> — MACD ve RSI",
            template="plotly_dark",
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="rgba(255,255,255,0.7)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if st.session_state.competitors:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index[-90:],
                y=df['Close'].iloc[-90:],
                mode='lines',
                name=data['symbol'],
                line=dict(color='#3b82f6', width=3)
            ))
            
            colors = ['#22c55e', '#f59e0b', '#ef4444']
            for i, comp in enumerate(st.session_state.competitors[:3]):
                if comp and comp.get('df') is not None:
                    comp_df = comp['df']
                    fig.add_trace(go.Scatter(
                        x=comp_df.index[-90:],
                        y=comp_df['Close'].iloc[-90:],
                        mode='lines',
                        name=comp['symbol'],
                        line=dict(color=colors[i % len(colors)], width=1.5, dash='dash')
                    ))
            
            fig.update_layout(
                title=f"<b>{data['symbol']}</b> — Rakiplerle Karşılaştırma",
                template="plotly_dark",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(color="rgba(255,255,255,0.7)")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Rakip verileri yüklenemedi.")
    
    # HABERLER
    with st.expander("📰 GÜNCEL HABERLER", expanded=True):
        news = fetch_news(data['symbol'])
        st.markdown(news)
    
    # AI ANALİZ
    st.markdown("---")
    st.markdown('<h3 style="text-align:center; font-weight:700;">🤖 AI QUANT ANALİZ</h3>', unsafe_allow_html=True)
    st.markdown("---")
    
    for msg in st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    
    # KULLANICI SORUSU
    user_question = st.chat_input("📝 Hisse hakkında detaylı soru sorun...")
    
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        with st.spinner("🧠 Derinlemesine analiz yapılıyor..."):
            analysis = analyze_with_ai(
                user_question,
                st.session_state.market_data,
                st.session_state.competitors,
                st.session_state.messages,
                client
            )
            
            st.session_state.messages.append({"role": "assistant", "content": analysis})
            st.rerun()

else:
    # BAŞLANGIÇ EKRANI
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:72px; margin-bottom:20px;">📈</div>
        <h2 style="font-size:2rem; margin-bottom:10px;">Profesyonel AI Quant Terminal</h2>
        <p style="color:#8b9aac; font-size:1.1rem; max-width:500px; margin:0 auto;">
            Yukarıdaki arama kutusuna bir hisse kodu yazın ve anında detaylı analiz alın.
        </p>
        <div style="margin-top:40px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.05); padding:10px 24px; border-radius:30px; border:1px solid rgba(255,255,255,0.06); color:#e7edf4;">📊 SASA</span>
            <span style="background:rgba(255,255,255,0.05); padding:10px 24px; border-radius:30px; border:1px solid rgba(255,255,255,0.06); color:#e7edf4;">✈️ THYAO</span>
            <span style="background:rgba(255,255,255,0.05); padding:10px 24px; border-radius:30px; border:1px solid rgba(255,255,255,0.06); color:#e7edf4;">🏦 GARAN</span>
            <span style="background:rgba(255,255,255,0.05); padding:10px 24px; border-radius:30px; border:1px solid rgba(255,255,255,0.06); color:#e7edf4;">🔧 ASELS</span>
            <span style="background:rgba(255,255,255,0.05); padding:10px 24px; border-radius:30px; border:1px solid rgba(255,255,255,0.06); color:#e7edf4;">🛢️ TUPRS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
