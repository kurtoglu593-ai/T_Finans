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

# .env dosyasını yükle
load_dotenv()

# 📍 Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📍 CONFIG
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"
    TV_SCAN_URL = "https://scanner.tradingview.com/turkey/scan"
    STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
    FX_BASE_URL = "https://api.frankfurter.app"
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
    
    RSI_PERIOD = 14
    SMA_FAST = 20
    SMA_SLOW = 50
    MAX_HISTORY_DAYS = 365  # 1 yıllık veri

# 📍 SAYFA AYARLARI
st.set_page_config(
    page_title="BISTeknik PRO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 📍 TEMA
st.markdown("""
<style>
    /* ANA ARKA PLAN */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 50%, #0a0e1a 100%);
    }
    
    /* TÜM YAZILAR BEYAZ */
    * {
        color: #ffffff !important;
    }
    
    /* INPUT ALANLARI */
    .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 18px !important;
        padding: 12px 20px !important;
    }
    
    /* BUTONLAR */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* METRİK KARTLARI */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    
    /* EXPANDER */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 20px !important;
    }
    
    /* CHAT */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
    }
    
    [data-testid="stChatInput"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stChatInput"] input {
        color: #ffffff !important;
        font-size: 16px !important;
    }
    
    /* BAŞLIKLAR */
    h1, h2, h3, h4 {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 900 !important;
    }
    
    /* GRAFİK */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* DİV KARTLARI */
    .pro-card {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px) !important;
        margin: 10px 0 !important;
    }
    
    .glass-card {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin: 12px 0 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 📍 YARDIMCI FONKSİYONLAR
def sanitize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol in ["XU100", "BIST100", "BIST 100", "^XU100"]:
        return "^XU100"
    if symbol in ["XBANA", "BIST ANA"]:
        return "XBANA.IS"
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
        return "Aşırı Alım 🔴"
    elif rsi_value < 30:
        return "Aşırı Satım 🟢"
    elif 40 <= rsi_value <= 60:
        return "Nötr 🟡"
    elif rsi_value > 60:
        return "Alım Bölgesi 📈"
    else:
        return "Satım Bölgesi 📉"

# 📍 VERİ ÇEKME
@st.cache_data(ttl=300)
def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Çoklu kaynaktan veri çek"""
    clean_sym = sanitize_symbol(symbol)
    
    try:
        # 1. Yahoo Finance
        ticker = yf.Ticker(clean_sym)
        info = ticker.info
        
        if info and 'regularMarketPrice' in info:
            hist = ticker.history(period="1y")
            
            if not hist.empty:
                df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.index = pd.to_datetime(df.index)
                
                # Teknik göstergeler
                df['SMA20'] = df['Close'].rolling(20).mean()
                df['SMA50'] = df['Close'].rolling(50).mean()
                df['SMA200'] = df['Close'].rolling(200).mean()
                df['RSI'] = calculate_rsi(df['Close'])
                
                # MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['Signal']
                
                # Bollinger
                df['BB_Middle'] = df['Close'].rolling(20).mean()
                bb_std = df['Close'].rolling(20).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                
                last_price = float(df['Close'].iloc[-1])
                change = float(info.get('regularMarketChangePercent', 0))
                
                return {
                    "symbol": clean_sym,
                    "price": last_price,
                    "change": change,
                    "currency": "TRY",
                    "df": df,
                    "data_source": "Yahoo Finance",
                    "info": info
                }
        
        # 2. Stooq (fallback)
        stooq_code = clean_sym.replace(".IS", ".TR").replace("^", "").lower()
        stooq_url = f"https://stooq.com/q/d/l/?s={stooq_code}&i=d"
        df_stooq = pd.read_csv(stooq_url)
        
        if not df_stooq.empty and 'Close' in df_stooq.columns:
            df_stooq['Date'] = pd.to_datetime(df_stooq['Date'])
            df_stooq.set_index('Date', inplace=True)
            df_stooq.sort_index(inplace=True)
            
            df = df_stooq[['Open', 'High', 'Low', 'Close']].copy()
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            
            last_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else last_price
            change = ((last_price - prev_price) / prev_price) * 100 if prev_price else 0
            
            return {
                "symbol": clean_sym,
                "price": last_price,
                "change": change,
                "currency": "TRY",
                "df": df,
                "data_source": "Stooq",
                "info": None
            }
            
    except Exception as e:
        logger.error(f"Veri çekme hatası: {e}")
    
    return None

# 📍 HABER ÇEKME
@st.cache_data(ttl=300)
def fetch_news(symbol: str) -> str:
    """Güncel haberleri çek"""
    clean_name = symbol.replace(".IS", "").replace("^", "")
    news_list = []
    
    try:
        # Google Haberler
        url = f"https://news.google.com/search?q={clean_name}&hl=tr&gl=TR"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('a', class_='JtKRv')
            for item in headlines[:5]:
                title = item.get_text().strip()
                if title and len(title) > 10:
                    news_list.append(f"• {title}")
        
        # TradingView
        tv_url = f"https://www.tradingview.com/symbols/BIST:{clean_name}/news/"
        response = requests.get(tv_url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('a', class_='news-item')
            for item in headlines[:3]:
                title = item.get_text().strip()
                if title and len(title) > 10:
                    news_list.append(f"• {title}")
        
        if news_list:
            return "\n".join(news_list[:5])
        else:
            return "📡 Güncel haber bulunamadı."
            
    except Exception as e:
        return f"📡 Haberler alınamıyor: {str(e)[:50]}"

# 📍 RAKİP HİSSELER
def get_competitors(symbol: str, market_data: Dict) -> list:
    """Rakip hisseleri belirle"""
    competitors = []
    all_stocks = [
        "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "SISE.IS",
        "KCHOL.IS", "SAHOL.IS", "TUPRS.IS", "EREGL.IS", "PGSUS.IS"
    ]
    
    # Rastgele 3 rakip seç
    import random
    random.seed(hash(symbol))
    competitors = random.sample([s for s in all_stocks if s != symbol], 3)
    
    return competitors

# 📍 GELİŞMİŞ AI ANALİZİ
def analyze_with_ai(prompt: str, market_data: Dict, competitors_data: list, history: list, client) -> str:
    """ULTRA PRO AI ANALİZ"""
    
    df = market_data['df']
    last_price = market_data['price']
    change = market_data['change']
    
    # Teknik veriler
    last_rsi = df['RSI'].iloc[-1] if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 50
    sma20 = df['SMA20'].iloc[-1] if 'SMA20' in df and not pd.isna(df['SMA20'].iloc[-1]) else last_price
    sma50 = df['SMA50'].iloc[-1] if 'SMA50' in df and not pd.isna(df['SMA50'].iloc[-1]) else last_price
    sma200 = df['SMA200'].iloc[-1] if 'SMA200' in df and not pd.isna(df['SMA200'].iloc[-1]) else last_price
    
    # Trend
    trend = "YÜKSELİŞ" if last_price > sma200 else "DÜŞÜŞ" if last_price < sma200 else "YATAY"
    
    # Destek/Direnç
    support = df['Low'].tail(50).min()
    resistance = df['High'].tail(50).max()
    
    # Volatilite
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5) if len(returns) > 0 else 0
    
    # Kazanç hesaplamaları
    monthly_return = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100 if len(df) > 22 else 0
    yearly_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100 if len(df) > 0 else 0
    
    # Rakip karşılaştırma
    comp_text = ""
    if competitors_data:
        comp_text = "\n📊 **RAKİP KARŞILAŞTIRMA:**\n"
        for comp in competitors_data:
            if comp:
                comp_text += f"  • {comp['symbol']}: {comp['price']:.2f} (%{comp['change']:+.2f})\n"
    
    # Haberler
    news = fetch_news(market_data['symbol'])
    
    # Kapsamlı veri metni
    data_str = f"""
📊 **HİSSE ANALİZ VERİLERİ**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Sembol: {market_data['symbol']}
🔹 Fiyat: {last_price:.2f} TRY
🔹 Günlük Değişim: %{change:+.2f}
🔹 Veri Kaynağı: {market_data['data_source']}

📈 **TEKNİK GÖSTERGELER**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔸 RSI(14): {last_rsi:.1f} ({get_rsi_comment(last_rsi)})
🔸 SMA20: {sma20:.2f} TRY
🔸 SMA50: {sma50:.2f} TRY
🔸 SMA200: {sma200:.2f} TRY
🔸 Trend: {trend}

🛡️ **DESTEK/DİRENÇ**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 Destek: {support:.2f} TRY
🔴 Direnç: {resistance:.2f} TRY

📊 **GETİRİ ANALİZİ**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📆 Aylık Getiri: %{monthly_return:+.2f}
📅 Yıllık Getiri: %{yearly_return:+.2f}
⚡ Volatilite: %{volatility*100:.2f}

{comp_text}

📰 **GÜNCEL HABERLER**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{news}
"""

    # ULTRA PRO SYSTEM PROMPT
    system_prompt = f"""
Sen, **BISTeknik ULTRA PRO** yapay zeka analistisin. Wall Street seviyesinde profesyonel bir quant analistsin.

🎯 **GÖREVİN:**
Kullanıcının verdiği hisse hakkında DETAYLI, KAPSAYICI ve PROFESYONEL bir analiz yap.

📋 **ANALİZ FORMATI (5-6 PARAGRAF):**

1. **ÖZET GÖRÜŞ** (1 paragraf)
   - Hisse hakkında genel değerlendirme
   - Kısa, orta, uzun vade beklenti
   - 1-2 cümlede özet

2. **TEKNİK ANALİZ** (1-2 paragraf)
   - RSI yorumu ve olası hareketler
   - SMA'ların konumu ve trend durumu
   - MACD ve Bollinger Band yorumu
   - Destek/Direnç seviyeleri ve kırılma senaryoları

3. **GETİRİ VE RİSK ANALİZİ** (1 paragraf)
   - Aylık/Yıllık getiri değerlendirmesi
   - Volatilite analizi
   - Risk/Getiri potansiyeli
   - Olası hedef fiyatlar

4. **REKABET ANALİZİ** (1 paragraf)
   - Rakiplerle karşılaştırma
   - Sektörel performans
   - Piyasa konumu

5. **HABER VE GÜNDEM** (1 paragraf)
   - Güncel haberlerin etkisi
   - Makroekonomik faktörler
   - Yakın vadeli katalizörler

6. **SONUÇ VE STRATEJİ** (1 paragraf)
   - Kısa/Orta/Uzun vade strateji
   - İzlenecek seviyeler
   - Son değerlendirme

⚠️ **KURALLAR:**
- Sadece verilen GERÇEK VERİLERE dayan
- Yatırım tavsiyesi VERME, analiz yap
- Teknik terimleri doğru kullan
- Türkçe yaz, İngilizce terimlerle destekle
- Objektif ve profesyonel ol

📊 **MEVCUT VERİLER:**
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

# Groq API Key
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

# Başlık
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 style="font-size: 3rem; margin: 0;">🚀 BISTeknik PRO</h1>
    <p style="color: rgba(255,255,255,0.5); font-size: 1rem; margin: 0;">Ultra Profesyonel AI Quant Terminal</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 📍 ANA SEARCH
search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    user_input = st.text_input(
        "",
        placeholder="🔍 Hisse kodu veya ismi girin... (Örnek: SASA, THYAO, GARAN)",
        label_visibility="collapsed",
        key="search_input"
    )

with search_col2:
    analyze_btn = st.button("🎯 ANALİZ ET", use_container_width=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "market_data" not in st.session_state:
    st.session_state.market_data = None

if "competitors" not in st.session_state:
    st.session_state.competitors = []

# Analiz yap
if analyze_btn and user_input:
    with st.spinner("🚀 Veriler çekiliyor ve analiz yapılıyor..."):
        # Veri çek
        symbol = sanitize_symbol(user_input)
        market_data = fetch_market_data(symbol)
        
        if market_data:
            st.session_state.market_data = market_data
            
            # Rakipleri çek
            comp_symbols = get_competitors(symbol, market_data)
            competitors = []
            for comp_sym in comp_symbols:
                comp_data = fetch_market_data(comp_sym)
                if comp_data:
                    competitors.append(comp_data)
            st.session_state.competitors = competitors
            
            # AI Analizi
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
            st.error(f"❌ {symbol} için veri bulunamadı. Lütfen geçerli bir hisse kodu girin.")

# 📊 ANA GÖSTERİM
if st.session_state.market_data:
    data = st.session_state.market_data
    df = data['df']
    
    # 📍 ÜST KARTLAR
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Fiyat",
            f"{data['price']:.2f} TRY",
            f"%{data['change']:+.2f}",
            delta_color="normal" if data['change'] >= 0 else "inverse"
        )
    
    with col2:
        last_rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
        st.metric("📊 RSI (14)", f"{last_rsi:.1f}", get_rsi_comment(last_rsi))
    
    with col3:
        sma20 = df['SMA20'].iloc[-1] if 'SMA20' in df else 0
        st.metric("📈 SMA20", f"{sma20:.2f}")
    
    with col4:
        sma50 = df['SMA50'].iloc[-1] if 'SMA50' in df else 0
        st.metric("📈 SMA50", f"{sma50:.2f}")
    
    with col5:
        monthly_return = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100 if len(df) > 22 else 0
        st.metric("📆 Aylık Getiri", f"%{monthly_return:+.2f}")
    
    st.markdown("---")
    
    # 📍 GRAFİKLER
    tab1, tab2, tab3 = st.tabs(["📈 Fiyat ve SMA", "📊 Teknik Göstergeler", "📉 Karşılaştırma"])
    
    with tab1:
        fig = go.Figure()
        
        # Mum grafiği
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
        
        # SMA'lar
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
            line=dict(color='#8b5cf6', width=1.5, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{data['symbol']} — Fiyat ve Hareketli Ortalamalar",
            template="plotly_dark",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            row_heights=[0.6, 0.4])
        
        # MACD
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
        
        # RSI
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
            title=f"{data['symbol']} — MACD ve RSI",
            template="plotly_dark",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if st.session_state.competitors:
            fig = go.Figure()
            
            # Ana hisse
            fig.add_trace(go.Scatter(
                x=df.index[-90:],
                y=df['Close'].iloc[-90:],
                mode='lines',
                name=data['symbol'],
                line=dict(color='#3b82f6', width=3)
            ))
            
            # Rakipler
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
                title=f"{data['symbol']} — Rakiplerle Karşılaştırma",
                template="plotly_dark",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rakip verileri yüklenemedi.")
    
    # 📍 HABERLER
    with st.expander("📰 GÜNCEL HABERLER", expanded=True):
        news = fetch_news(data['symbol'])
        st.markdown(news)
    
    # 📍 AI ANALİZ
    st.markdown("---")
    st.markdown("### 🤖 AI QUANT ANALİZ")
    
    for msg in st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    
    # Kullanıcı sorusu
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
    # Başlangıç ekranı
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <div style="font-size: 72px; margin-bottom: 20px;">🚀</div>
        <h2 style="font-size: 2rem; margin-bottom: 10px;">Profesyonel AI Quant Terminal</h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 1.1rem;">
            Yukarıdaki arama kutusuna bir hisse kodu yazın ve analiz edin.
        </p>
        <div style="margin-top: 30px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.05); padding: 8px 20px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);">📊 SASA</span>
            <span style="background: rgba(255,255,255,0.05); padding: 8px 20px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);">✈️ THYAO</span>
            <span style="background: rgba(255,255,255,0.05); padding: 8px 20px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);">🏦 GARAN</span>
            <span style="background: rgba(255,255,255,0.05); padding: 8px 20px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);">🔧 ASELS</span>
            <span style="background: rgba(255,255,255,0.05); padding: 8px 20px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);">🛢️ TUPRS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
