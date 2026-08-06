import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from groq import Groq

# Model Tanımlamaları
MODEL_70B = "llama-3.3-70b-versatile"
MODEL_8B = "llama-3.1-8b-instant"

# Plotly Tema Ayarı (Aydınlık Finans Teması)
pio.templates.default = "plotly_white"

# Sayfa Yapılandırması
st.set_page_config(
    page_title="T - Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FERAH VE AYDINLIK FİNANS TEMASI (LIGHT MODE CSS) ---
st.markdown("""
<style>
    /* Global Aydınlık Zemin ve Font Yapısı */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    header, footer { display: none !important; }

    .main .block-container {
        padding: 1rem 1.5rem !important;
        max-width: 99% !important;
    }

    /* Streamlit Varsayılan Kutu Temizleme */
    [data-testid="stVerticalBlock"] > div {
        background: transparent !important;
        border: none !important;
    }

    /* Metrik Kartları */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
    }

    /* Chat Balonları ve Uyarı Kutuları */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    
    [data-testid="stAlert"] {
        background-color: #e0f2fe !important;
        color: #0369a1 !important;
        border: 1px solid #bae6fd !important;
        border-radius: 8px !important;
    }

    /* Input Alanları */
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        background-color: transparent !important;
    }

    .stTextInput input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }

    /* Buton Stilleri */
    .stButton button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: background 0.2s ease;
    }
    .stButton button:hover {
        background: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* Özel Başlık Kartları */
    .t-panel-header {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-bottom: 2px solid #2563eb;
        border-radius: 8px 8px 0 0;
        padding: 8px 14px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        justify-content: space-between;
        align-items: center;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# --- FINANSAL HESAPLAMALAR VE YOK EDİLEMEZ VERİ MOTORU ---
def sanitize_symbol(symbol: str) -> str:
    """Türk hisse senetleri için otomatik .IS eklemesi yapar."""
    if not symbol:
        return "THYAO.IS"
    symbol = symbol.strip().upper()
    if not symbol.endswith(".IS") and "-" not in symbol and "=" not in symbol:
        if symbol.isalpha() and 3 <= len(symbol) <= 5:
            return f"{symbol}.IS"
    return symbol

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_data_multi_fallback(symbol):
    """
    Yahoo Finance v8/v7 + Alternatif Domainler üzerinden veri çeker.
    Tek bir sunucuya bağımlı kalmaz.
    """
    clean_symbol = sanitize_symbol(symbol)
    
    # Engelleri aşmak için rotating User-Agent ve Header listesi
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }

    endpoints = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_symbol}?range=6m&interval=1d",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{clean_symbol}?range=6m&interval=1d",
        f"https://query1.finance.yahoo.com/v7/finance/chart/{clean_symbol}?range=6m&interval=1d"
    ]

    # Eğer .IS yoksa yedek listeye ekle
    if not clean_symbol.endswith(".IS") and "-" not in clean_symbol and "=" not in clean_symbol:
        alt_sym = f"{clean_symbol}.IS"
        endpoints.append(f"https://query1.finance.yahoo.com/v8/finance/chart/{alt_sym}?range=6m&interval=1d")

    data = None
    used_symbol = clean_symbol

    for url in endpoints:
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                js = res.json()
                result = js.get('chart', {}).get('result', [])
                if result and result[0].get('timestamp'):
                    data = result[0]
                    used_symbol = result[0].get('meta', {}).get('symbol', clean_symbol)
                    break
        except Exception:
            continue

    if not data:
        return None

    try:
        timestamps = data.get('timestamp', [])
        quote = data.get('indicators', {}).get('quote', [{}])[0]

        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Open': quote.get('open'),
            'High': quote.get('high'),
            'Low': quote.get('low'),
            'Close': quote.get('close')
        }).dropna()

        if df.empty or len(df) < 5:
            return None

        df.set_index('Date', inplace=True)
        
        # İndikatörler
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)

        last_p = float(df['Close'].iloc[-1])
        prev_p = float(df['Close'].iloc[-2]) if len(df) > 1 else last_p
        pct_chg = ((last_p - prev_p) / prev_p) * 100.0 if prev_p else 0.0
        curr = 'TRY' if used_symbol.endswith('.IS') else 'USD'

        return {
            "symbol": used_symbol,
            "price": last_p,
            "change": pct_chg,
            "currency": curr,
            "df": df
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def get_quick_market_data():
    tickers = {
        "USD/TRY": "USDTRY=X",
        "EUR/TRY": "EURTRY=X",
        "ONS ALTIN": "GC=F",
        "BIST 100": "^XU100"
    }
    data = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for name, sym in tickers.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5d&interval=1d"
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                json_data = res.json()
                closes = json_data['chart']['result'][0]['indicators']['quote'][0]['close']
                valid_closes = [c for c in closes if c is not None]
                if len(valid_closes) >= 2:
                    last = float(valid_closes[-1])
                    prev = float(valid_closes[-2])
                    chg = ((last - prev) / prev) * 100
                    data[name] = (last, chg)
                elif len(valid_closes) == 1:
                    data[name] = (float(valid_closes[-1]), 0.0)
        except Exception:
            pass
    return data

def analyze_with_ai(user_prompt, market_data, history, client):
    if market_data and market_data.get('df') is not None:
        df = market_data['df']
        last_rsi = df['RSI'].iloc[-1] if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 0
        data_str = (
            f"Varlık: {market_data['symbol']} | "
            f"Son Fiyat: {market_data['price']:.2f} {market_data['currency']} | "
            f"Günlük Değişim: %{market_data['change']:+.2f} | "
            f"RSI(14): {last_rsi:.1f}"
        )
    else:
        data_str = "UYARI: İstenen sembole ait canlı piyasa verisi sunucudan çekilemedi."

    system_instruction = (
        "Sen 'T' adında profesyonel bir borsa ve finans analistisin. "
        "Mevcut Veri Durumu: " + data_str + " "
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
        return sanitize_symbol(code)
    except Exception:
        return None

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.markdown("<h3 style='color: #2563eb; font-size: 1.1rem; margin:0;'>⚡ T — TERMINAL</h3>", unsafe_allow_html=True)
    st.caption("Quantitative Trading Core")
    st.markdown("---")

    groq_api_key = st.text_input("Groq API Key:", type="password")
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; margin-bottom:5px;'>WATCHLIST</p>", unsafe_allow_html=True)
    watchlist_input = st.text_input("Semboller:", value="THYAO.IS, ASELS.IS, BTC-USD")
    if st.button("🔄 GÜNCELLE"):
        symbols = [sanitize_symbol(s) for s in watchlist_input.split(",") if s.strip()]
        for sym in symbols:
            res_data = fetch_data_multi_fallback(sym)
            if res_data:
                st.metric(label=res_data['symbol'], value=f"{res_data['price']:,.2f}", delta=f"%{res_data['change']:+.2f}")
            else:
                st.caption(f"⚠️ {sym} okunamadı.")

if not groq_api_key:
    st.info("👈 Sol menüden **Groq API Key** girerek terminali aktif edin.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- ANA TERMINAL EKRANI ---

# 1. Üst Bant (Piyasa Verileri)
market_summary = get_quick_market_data()
if market_summary:
    cols = st.columns(len(market_summary))
    for idx, (name, (val, chg)) in enumerate(market_summary.items()):
        cols[idx].metric(label=name, value=f"{val:,.2f}", delta=f"%{chg:+.2f}")

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# 2. İki Sütunlu Izgara
col_left, col_right = st.columns([1.6, 1.0], gap="small")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "⚡ **T Terminal Çevrimiçi.** Bir hisse/kripto kodu girin (Örn: `THYAO`, `BTC-USD`)."}
    ]

# SOL PANEL: Grafik Engine
with col_left:
    st.markdown("<div class='t-panel-header'><span>📊 TECHNICAL ANALYTICS & CANDLESTICK ENGINE</span><span style='color:#16a34a;'>● LIVE</span></div>", unsafe_allow_html=True)
    
    last_user_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "THYAO.IS")
    active_symbol = detect_symbol_with_ai(last_user_query, st.session_state.messages, client) or "THYAO.IS"
    
    market_data = fetch_data_multi_fallback(active_symbol)
    
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
            increasing_line_color='#16a34a', decreasing_line_color='#dc2626',
            increasing_fillcolor='rgba(22, 163, 74, 0.1)', decreasing_fillcolor='rgba(220, 38, 38, 0.1)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='#d97706', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='#2563eb', width=1.5)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#9333ea', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", opacity=0.5, row=2, col=1)

        fig.update_layout(
            template="plotly_white",
            height=560,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        )
        fig.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0")
        fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Lütfen sohbet paneline analiz etmek istediğiniz varlığı yazın (Örn: THYAO, ASELS, BTC-USD).")

# SAĞ PANEL: AI Engine & Chat
with col_right:
    st.markdown("<div class='t-panel-header'><span>🤖 AI QUANT ANALYST</span><span>MODEL: 70B</span></div>", unsafe_allow_html=True)
    
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Soru veya sembol yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Analiz ediliyor..."):
            symbol = detect_symbol_with_ai(prompt, st.session_state.messages, client)
            current_market_data = fetch_data_multi_fallback(symbol) if symbol else None
            ai_response = analyze_with_ai(prompt, current_market_data, st.session_state.messages, client)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.rerun()
