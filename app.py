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

# --- RADİKAL DÖNÜŞÜM: AGRESİF TERMINAL STİLİ ---
st.markdown("""
<style>
    /* Global Reset & Dark Theme */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #06080c !important;
        color: #d1d5db !important;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid #1a2332 !important;
    }

    header, footer { visibility: hidden !important; }

    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 98% !important;
    }

    /* Streamlit varsayılan kutularını sıfırlama ve özelleştirme */
    [data-testid="stMetric"] {
        background: #0d121d !important;
        border: 1px solid #1e293b !important;
        border-left: 3px solid #00e676 !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6) !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
    }

    /* Chat Alanı Tasarımı */
    [data-testid="stChatMessage"] {
        background-color: #0d121d !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
    }

    /* Input & Butonlar */
    .stTextInput input {
        background-color: #0d121d !important;
        color: #00e676 !important;
        border: 1px solid #1e293b !important;
        border-radius: 6px !important;
    }

    .stTextInput input:focus {
        border-color: #00e676 !important;
        box-shadow: 0 0 10px rgba(0, 230, 118, 0.2) !important;
    }

    .stButton button {
        background: #101726 !important;
        color: #38bdf8 !important;
        border: 1px solid #1e293b !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }

    .stButton button:hover {
        background: #38bdf8 !important;
        color: #06080c !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4) !important;
    }

    /* Kartlar için Özel Hazırlanmış Terminal Başlığı */
    .terminal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0d121d;
        border: 1px solid #1e293b;
        border-radius: 6px 6px 0 0;
        padding: 8px 14px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.5px;
    }

    hr { border-color: #1a2332 !important; }
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
    st.markdown("<h3 style='color: #00e676; font-size: 1.1rem; margin-bottom: 0;'>⚡ T — TERMINAL</h3>", unsafe_allow_html=True)
    st.caption("Algorithmic Trading & Analytics Engine")
    st.divider()

    groq_api_key = st.text_input("Groq API Key:", type="password")
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.divider()
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #64748b;'>WATCHLIST</p>", unsafe_allow_html=True)
    watchlist_input = st.text_input("Semboller:", value="THYAO.IS, ASELS.IS, BTC-USD")
    if st.button("🔄 Güncelle"):
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
    st.info("👈 Sol menüden **Groq API Key** girerek terminali aktif edin.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- ANA EKRAN DÜZENİ (DASHBOARD) ---

# 1. Ticker Tape (Piyasa Akışı)
market_summary = get_quick_market_data()
if market_summary:
    cols = st.columns(len(market_summary))
    for idx, (name, (val, chg)) in enumerate(market_summary.items()):
        cols[idx].metric(label=name, value=f"{val:,.2f}", delta=f"%{chg:+.2f}")

st.divider()

# 2. İki Sütunlu Terminal Yapısı
col_left, col_right = st.columns([1.65, 1.0], gap="small")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "⚡ **T Terminal Çevrimiçi.** Bir hisse/kripto kodu girin (Örn: `THYAO`, `BTC-USD`)."}
    ]

# SOL SÜTUN: Grafik & Piyasa Paneli
with col_left:
    st.markdown("<div class='terminal-header'><span>📊 TECHNICAL CHART & FEED</span><span>LIVE</span></div>", unsafe_allow_html=True)
    
    last_user_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "THYAO.IS")
    active_symbol = detect_symbol_with_ai(last_user_query, st.session_state.messages, client) or "THYAO.IS"
    
    market_data = fetch_data(active_symbol)
    
    if market_data and market_data.get("df") is not None:
        df = market_data["df"].tail(90)
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=(f"{market_data['symbol']} — CANDLESTICK & SMA", "RSI (14) OSCILLATOR"),
            row_heights=[0.72, 0.28]
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Fiyat",
            increasing_line_color='#00e676', decreasing_line_color='#ff3366',
            increasing_fillcolor='rgba(0, 230, 118, 0.1)', decreasing_fillcolor='rgba(255, 51, 102, 0.1)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='#ffb703', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='#2196f3', width=1.2)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#a855f7', width=1.2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e676", opacity=0.5, row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=540,
            paper_bgcolor="#0d121d",
            plot_bgcolor="#06080c",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        )
        fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
        fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Lütfen sohbet paneline analiz etmek istediğiniz varlığı yazın.")

# SAĞ SÜTUN: AI Analist & Chat Paneli
with col_right:
    st.markdown("<div class='terminal-header'><span>🤖 AI ANALYST ENGINE</span><span>MODEL 70B</span></div>", unsafe_allow_html=True)
    
    chat_container = st.container(height=455)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Soru veya sembol yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("İşleniyor..."):
            symbol = detect_symbol_with_ai(prompt, st.session_state.messages, client)
            current_market_data = fetch_data(symbol) if symbol else None
            ai_response = analyze_with_ai(prompt, current_market_data, st.session_state.messages, client)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.rerun()
