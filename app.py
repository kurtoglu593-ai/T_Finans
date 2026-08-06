import os
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from groq import Groq

# Model Tanımlamaları
MODEL_70B = "llama-3.3-70b-versatile"
MODEL_8B = "llama-3.1-8b-instant"

# Plotly Tema Ayarı
pio.templates.default = "plotly_dark"

# Sayfa Yapılandırması
st.set_page_config(
    page_title="T - Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN PREMIUM FINANS TERMINALI CUSTOM CSS ---
st.markdown("""
<style>
    /* Genel Arka Plan ve Tipografi */
    .stApp {
        background: radial-gradient(circle at top left, #0d1117, #05070a);
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Yan Menü (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #090d14 !important;
        border-right: 1px solid #1f293d !important;
    }
    
    header {visibility: hidden;}
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 98%;
    }
    
    /* Metrik Kartları (Ticker Tape) */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.7), rgba(13, 17, 23, 0.8));
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #388bfd;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Chat Mesaj Kutuları */
    .stChatMessage {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Input & Buton Stilleri */
    .stTextInput input {
        background-color: #090d14 !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTextInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.3);
    }
    .stButton button {
        background: linear-gradient(180deg, #1f6feb, #1158c7) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: linear-gradient(180deg, #388bfd, #1f6feb) !important;
        box-shadow: 0 4px 16px rgba(56, 139, 253, 0.5);
        transform: translateY(-1px);
    }
    
    /* Bölme Çizgileri */
    hr {
        border-color: #21262d !important;
    }
    
    /* Başlık Vurguları */
    h3 {
        color: #f0f6fc !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)

# --- FINANSAL HESAPLAMALAR VE VERİ ÇEKME ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6m")
        if df.empty:
            return None
        
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)

        info = ticker.fast_info
        last_p = float(info.last_price)
        prev_p = float(info.previous_close)
        pct_chg = ((last_p - prev_p) / prev_p) * 100.0
        curr = getattr(info, 'currency', 'TL')

        return {
            "symbol": symbol,
            "price": last_p,
            "change": pct_chg,
            "currency": curr,
            "df": df
        }
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_quick_market_data():
    tickers = {
        "BIST 100": "^XU100",
        "USD/TRY": "USDTRY=X",
        "EUR/TRY": "EURTRY=X",
        "ONS ALTIN": "GC=F"
    }
    data = {}
    for name, sym in tickers.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                last = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                chg = ((last - prev) / prev) * 100
                data[name] = (last, chg)
            elif len(hist) == 1:
                data[name] = (hist['Close'].iloc[-1], 0.0)
        except Exception:
            pass
    return data

def analyze_with_ai(user_prompt, market_data, history, client):
    data_str = "Canlı piyasa verisi çekilemedi."
    if market_data:
        last_rsi = market_data['df']['RSI'].iloc[-1] if 'RSI' in market_data['df'] else 0
        data_str = f"Varlık: {market_data['symbol']} | Son Fiyat: {market_data['price']:.2f} {market_data['currency']} | Değişim: %{market_data['change']:+.2f} | RSI(14): {last_rsi:.1f}"

    system_instruction = (
        "Sen 'T' adında profesyonel bir borsa ve finans analistisin. "
        "Canlı Piyasa Verisi: " + data_str + " "
        "Teknik indikatörleri temel alarak kısa, net, otoriter ve borsa terminali üslubuyla yanıt ver."
    )

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        res = client.chat.completions.create(
            model=MODEL_70B,
            messages=messages,
            temperature=0.2
        )
        return res.choices[0].message.content
    except Exception as err:
        return f"⚠️ Analiz Hatası: {err}"

def detect_symbol_with_ai(user_input, history, client):
    prompt = "Geçmiş: " + str(history[-2:]) + "\nSon Mesaj: '" + str(user_input) + "'\nBorsa/Kripto sembolünü döndür. (Örn: THYAO.IS, BTC-USD, GARAN.IS). Yoksa 'YOK' yaz."
    try:
        res = client.chat.completions.create(
            model=MODEL_8B,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        code = res.choices[0].message.content.strip().upper()
        if "YOK" in code or len(code) > 12:
            return None
        return code
    except Exception:
        return None

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.markdown("## ⚡ **T — TERMINAL**")
    st.caption("Pro Quantitative Trading Assistant")
    st.divider()

    groq_api_key = st.text_input("Groq API Key:", type="password", help="console.groq.com adresinden alabilirsiniz")
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.divider()
    st.markdown("#### 📌 **İzleme Listesi**")
    watchlist_input = st.text_input("Semboller:", value="THYAO.IS, ASELS.IS, BTC-USD")
    if st.button("🔄 Verileri Güncelle"):
        symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="2d")
                if len(hist) >= 1:
                    last_p = hist['Close'].iloc[-1]
                    chg_p = ((last_p - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) >= 2 else 0.0
                    st.metric(label=sym, value=f"{last_p:,.2f}", delta=f"%{chg_p:+.2f}")
            except Exception:
                st.caption(f"⚠️ {sym} okunamadı.")

if not groq_api_key:
    st.info("👈 **Başlamak İçin:** Sol menüye **Groq API Key** giriniz.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- ANA EKRAN DÜZENİ (DASHBOARD) ---

# 1. Üst Piyasa Bandı (Ticker Tape)
market_summary = get_quick_market_data()
if market_summary:
    cols = st.columns(len(market_summary))
    for idx, (name, (val, chg)) in enumerate(market_summary.items()):
        cols[idx].metric(label=name, value=f"{val:,.2f}", delta=f"%{chg:+.2f}")

st.divider()

# 2. İki Sütunlu Terminal Düzeni
col_left, col_right = st.columns([1.65, 1.0], gap="medium")

# SOHBET GEÇMİŞİ BAŞLATMA
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "⚡ **T Terminal Çevrimiçi.** Analiz etmek istediğiniz sembolü yazın (Örn: `THYAO`, `BTC-USD`)."}
    ]

# SOL SÜTUN: Grafik & Teknik Panel
with col_left:
    st.markdown("### 📊 **Piyasa Analitiği & Grafik**")
    
    last_user_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "THYAO.IS")
    active_symbol = detect_symbol_with_ai(last_user_query, st.session_state.messages, client) or "THYAO.IS"
    
    market_data = fetch_data(active_symbol)
    
    if market_data and market_data.get("df") is not None:
        df = market_data["df"].tail(90)
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=(f"{market_data['symbol']} — Fiyat Trendi & SMA", "RSI Indikatörü"),
            row_heights=[0.72, 0.28]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Fiyat",
            increasing_line_color='#00e676', decreasing_line_color='#ff5252',
            increasing_fillcolor='rgba(0, 230, 118, 0.2)', decreasing_fillcolor='rgba(255, 82, 82, 0.2)'
        ), row=1, col=1)

        # SMAs
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='#ffab00', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='#40c4ff', width=1.5)), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#e040fb', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff5252", opacity=0.6, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e676", opacity=0.6, row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=580,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13, 17, 23, 0.7)",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#21262d")
        fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Lütfen sohbet paneline analiz etmek istediğiniz varlığı yazın.")

# SAĞ SÜTUN: AI Analist & Chat Paneli
with col_right:
    st.markdown("### 🤖 **T — Yapay Zeka Analisti**")
    
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Soru sorun veya teknik analiz isteyin..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Piyasa taranıyor..."):
            symbol = detect_symbol_with_ai(prompt, st.session_state.messages, client)
            current_market_data = fetch_data(symbol) if symbol else None
            ai_response = analyze_with_ai(prompt, current_market_data, st.session_state.messages, client)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.rerun()
