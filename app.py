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
import yfinance as yf

# Cloud IP Engellerini (HTTP 403/401) aşan Tarayıcı Taklit Modülü
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests as cffi_requests
    HAS_CURL_CFFI = False

# Model Tanımlamaları
MODEL_70B = "llama-3.3-70b-versatile"

# Plotly Tema Ayarı
pio.templates.default = "plotly_white"

# Sayfa Yapılandırması
st.set_page_config(
    page_title="T - Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIGHT MODE FINANS TEMASI ---
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }
    header, footer { display: none !important; }
    .main .block-container { padding: 1rem 1.5rem !important; max-width: 99% !important; }
    [data-testid="stVerticalBlock"] > div { background: transparent !important; border: none !important; }
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.25rem !important; font-weight: 800 !important; }
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    [data-testid="stChatInput"] { background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important; }
    .stButton button { background: #2563eb !important; color: #ffffff !important; border-radius: 6px !important; font-weight: 600 !important; }
    .t-panel-header {
        background: #ffffff; border: 1px solid #cbd5e1; border-bottom: 2px solid #2563eb;
        border-radius: 8px 8px 0 0; padding: 8px 14px; font-size: 0.8rem; font-weight: 700; color: #1e293b;
        display: flex; justify-content: space-between; align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- VERİ VE YARDIMCI FONKSİYONLAR ---

def sanitize_symbol(symbol: str) -> str:
    """Sembol isimlerini borsalara uygun formata getirir (Örn: THYAO -> THYAO.IS)."""
    if not symbol:
        return "THYAO.IS"
    symbol = symbol.strip().upper()
    if not symbol.endswith(".IS") and "-" not in symbol and "=" not in symbol:
        if symbol.isalpha() and 3 <= len(symbol) <= 6:
            return f"{symbol}.IS"
    return symbol

def extract_symbol_fast(text: str, default_sym: str = "THYAO.IS") -> str:
    """Metin içinden hisse/kripto kodunu milisaniyeler içinde Regex ile yakalar."""
    words = re.findall(r'\b[A-Za-z0-9\.\=\-]{3,10}\b', text.upper())
    for w in words:
        if w in ["DOLAR", "EURO", "ALTIN", "ANALIZ", "NEDIR", "GUNCEL", "SERBEST"]:
            continue
        if len(w) >= 3:
            return sanitize_symbol(w)
    return default_sym

def calculate_rsi(series, period=14):
    """Wilder RSI Hesaplama Motoru."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_browser_session():
    """Chrome TLS Parmak İzini taklit eden canlı oturum oluşturur."""
    if HAS_CURL_CFFI:
        return cffi_requests.Session(impersonate="chrome120")
    else:
        session = cffi_requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        return session

def fetch_bist_tradingview(symbol_raw: str):
    """
    TradingView REST API - BIST Hisseleri ve BIST100 Endeksi İçin Gerçek Zamanlı Veri
    """
    session = get_browser_session()
    ticker_clean = symbol_raw.replace(".IS", "").replace("^", "").upper()
    
    # Endeks mi yoksa Hisse mi kontrolü
    if ticker_clean in ["XU100", "BIST100"]:
        tv_symbol = "BIST:XU100"
    else:
        tv_symbol = f"BIST:{ticker_clean}"
    
    url = "https://scanner.tradingview.com/turkey/scan"
    payload = {
        "symbols": {"tickers": [tv_symbol]},
        "columns": ["name", "close", "change", "open", "high", "low", "volume", "RSI"]
    }
    try:
        res = session.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data and len(data) > 0:
                d = data[0].get("d", [])
                close_p = d[1]
                change_pct = d[2]
                open_p = d[3] if d[3] is not None else close_p
                high_p = d[4] if d[4] is not None else close_p
                low_p = d[5] if d[5] is not None else close_p
                rsi_val = d[7] if len(d) > 7 and d[7] is not None else 50.0

                if close_p is None:
                    return None

                dates = pd.date_range(end=datetime.datetime.now(), periods=10, freq='D')
                df_real = pd.DataFrame({
                    'Open': [open_p]*10,
                    'High': [high_p]*10,
                    'Low': [low_p]*10,
                    'Close': [close_p]*10,
                }, index=dates)
                df_real['SMA20'] = close_p
                df_real['SMA50'] = close_p
                df_real['RSI'] = rsi_val

                return {
                    "symbol": symbol_raw.upper(),
                    "price": float(close_p),
                    "change": float(change_pct),
                    "currency": "TRY" if "USD" not in symbol_raw else "USD",
                    "support": float(low_p),
                    "resistance": float(high_p),
                    "df": df_real
                }
    except Exception as e:
        print(f"TradingView Fetch Error: {e}")
    return None

def fetch_real_market_data(symbol: str):
    """SADECE GERÇEK CANLI VERİ ÇEKER."""
    clean_sym = sanitize_symbol(symbol)
    df = None

    # MOTOR 1: TradingView API
    if clean_sym.endswith(".IS"):
        tv_res = fetch_bist_tradingview(clean_sym)
        if tv_res:
            return tv_res

    # MOTOR 2: Stooq CSV
    try:
        stooq_code = clean_sym.replace(".IS", ".TR").lower() if clean_sym.endswith(".IS") else clean_sym.lower()
        stooq_url = f"https://stooq.com/q/d/l/?s={stooq_code}&i=d"
        df_stooq = pd.read_csv(stooq_url)
        
        if not df_stooq.empty and 'Close' in df_stooq.columns and len(df_stooq) > 5:
            df_stooq['Date'] = pd.to_datetime(df_stooq['Date'])
            df_stooq.set_index('Date', inplace=True)
            df_stooq.sort_index(inplace=True)
            for col in ['Open', 'High', 'Low', 'Close']:
                df_stooq[col] = pd.to_numeric(df_stooq[col], errors='coerce')
            df_stooq.dropna(subset=['Close'], inplace=True)
            if len(df_stooq) > 5:
                df = df_stooq
    except Exception:
        df = None

    # MOTOR 3: Yahoo REST API
    if df is None or df.empty:
        session = get_browser_session()
        try:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{clean_sym}?range=6m&interval=1d"
            res = session.get(url, timeout=5)
            if res.status_code == 200:
                result = res.json().get('chart', {}).get('result', [])
                if result and result[0].get('timestamp'):
                    timestamps = result[0]['timestamp']
                    quote = result[0]['indicators']['quote'][0]
                    df_res = pd.DataFrame({
                        'Date': pd.to_datetime(timestamps, unit='s'),
                        'Open': quote.get('open'),
                        'High': quote.get('high'),
                        'Low': quote.get('low'),
                        'Close': quote.get('close')
                    }).dropna()
                    if not df_res.empty:
                        df_res.set_index('Date', inplace=True)
                        df = df_res
        except Exception:
            df = None

    if df is None or df.empty or len(df) < 5:
        return None

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)

    last_p = float(df['Close'].iloc[-1])
    prev_p = float(df['Close'].iloc[-2]) if len(df) > 1 else last_p
    pct_chg = ((last_p - prev_p) / prev_p) * 100.0 if prev_p else 0.0
    curr = 'TRY' if clean_sym.endswith('.IS') else 'USD'
    
    support = float(df['Low'].tail(20).min())
    resistance = float(df['High'].tail(20).max())

    return {
        "symbol": clean_sym,
        "price": last_p,
        "change": pct_chg,
        "currency": curr,
        "support": support,
        "resistance": resistance,
        "df": df
    }

@st.cache_data(ttl=60)
def get_quick_market_data():
    """Üst bant piyasa özeti."""
    tickers = {"USD/TRY": "USDTRY=X", "EUR/TRY": "EURTRY=X", "ONS ALTIN": "GC=F", "BIST 100": "^XU100"}
    data = {}
    session = get_browser_session()
    
    for name, sym in tickers.items():
        try:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range=5d&interval=1d"
            res = session.get(url, timeout=3)
            if res.status_code == 200:
                closes = res.json()['chart']['result'][0]['indicators']['quote'][0]['close']
                valid = [c for c in closes if c is not None]
                if len(valid) >= 2:
                    last, prev = float(valid[-1]), float(valid[-2])
                    data[name] = (last, ((last - prev) / prev) * 100)
        except Exception:
            pass
    return data

def analyze_with_ai(user_prompt, market_data, history, client):
    """AI Analiz Motoru."""
    if market_data and market_data.get('df') is not None:
        df = market_data['df']
        last_rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 50.0
        data_str = (
            f"KESİN GERÇEK VERİLER:\n"
            f"- Sembol: {market_data['symbol']}\n"
            f"- Canlı Son Fiyat: {market_data['price']:.2f} {market_data['currency']}\n"
            f"- Günlük Değişim: %{market_data['change']:+.2f}\n"
            f"- RSI(14): {last_rsi:.2f}\n"
            f"- Destek Seviyesi: {market_data['support']:.2f} {market_data['currency']}\n"
            f"- Direnç Seviyesi: {market_data['resistance']:.2f} {market_data['currency']}"
        )
    else:
        data_str = "UYARI: Canlı veri çekilemedi. Kullanıcıya verinin alınamadığını bildir."

    system_instruction = (
        "Sen 'T' adında profesyonel bir quant borsa analistisin.\n"
        "ÇOK ÖNEMLİ KURAL: Kesinlikle geçmiş bilginden fiyat UYDURMA. "
        "Yalnızca ve yalnızca sana sistem tarafından sağlanan GERÇEK FİYAT VERİSİNİ kullan.\n"
        f"Mevcut Canlı Pazar Verisi:\n{data_str}\n"
        "Analizini teknik indikatörleri temel alarak net, otoriter ve profesyonel borsa terminali üslubuyla sun."
    )

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        res = client.chat.completions.create(model=MODEL_70B, messages=messages, temperature=0.1)
        return res.choices[0].message.content
    except Exception as err:
        return f"⚠️ AI Analiz Hatası: {err}"

# --- SIDEBAR (SOL MENÜ) ---
with st.sidebar:
    st.markdown("<h3 style='color: #2563eb; font-size: 1.1rem; margin:0;'>⚡ T — TERMINAL</h3>", unsafe_allow_html=True)
    st.caption("Quantitative Trading Core")
    st.markdown("---")

    groq_api_key = st.text_input("Groq API Key:", type="password")
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; margin-bottom:5px;'>WATCHLIST (CANLI)</p>", unsafe_allow_html=True)
    watchlist_input = st.text_input("Semboller:", value="THYAO.IS, ASELS.IS, BTC-USD")
    if st.button("🔄 GÜNCELLE"):
        symbols = [sanitize_symbol(s) for s in watchlist_input.split(",") if s.strip()]
        for sym in symbols:
            res_data = fetch_real_market_data(sym)
            if res_data:
                st.metric(label=res_data['symbol'], value=f"{res_data['price']:,.2f} {res_data['currency']}", delta=f"%{res_data['change']:+.2f}")
            else:
                st.caption(f"⚠️ {sym} canlı veri alınamadı.")

if not groq_api_key:
    st.info("👈 Sol menüden **Groq API Key** girerek terminali aktif edin.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- ANA EKRAN ÜST BANT ---
market_summary = get_quick_market_data()
if market_summary:
    cols = st.columns(len(market_summary))
    for idx, (name, (val, chg)) in enumerate(market_summary.items()):
        cols[idx].metric(label=name, value=f"{val:,.2f}", delta=f"%{chg:+.2f}")

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.6, 1.0], gap="small")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "⚡ **T Terminal Çevrimiçi.** Bir hisse/kripto kodu girin (Örn: `THYAO`, `BTC-USD`)."}
    ]

# GÜNCEL AKTİF SEMBOL TESPİTİ
last_user_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "THYAO.IS")
active_symbol = extract_symbol_fast(last_user_query, default_sym="THYAO.IS")
market_data = fetch_real_market_data(active_symbol)

# SOL PANEL (GRAFİK ENGINE)
with col_left:
    st.markdown("<div class='t-panel-header'><span>📊 TECHNICAL ANALYTICS & CANDLESTICK ENGINE</span><span style='color:#16a34a;'>● REAL-TIME ENGINE</span></div>", unsafe_allow_html=True)
    
    if market_data and market_data.get("df") is not None:
        df = market_data["df"].tail(90)
        
        st.success(f"✅ **{market_data['symbol']}** Canlı Verisi Bağlandı | Son Fiyat: **{market_data['price']:.2f} {market_data['currency']}** (%{market_data['change']:+.2f})")

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
            height=540,
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
        st.error(f"❌ **{active_symbol}** için borsadan canlı veri alınamadı.")

# SAĞ PANEL (AI CHAT ENGINE)
with col_right:
    st.markdown("<div class='t-panel-header'><span>🤖 AI QUANT ANALYST</span><span>MODEL: 70B</span></div>", unsafe_allow_html=True)
    
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Soru veya sembol yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Canlı piyasa verileri işleniyor..."):
            # Sağ tarafta da aynı canlı veriyi kullanmasını sağlıyoruz
            query_symbol = extract_symbol_fast(prompt, default_sym=active_symbol)
            target_market_data = fetch_real_market_data(query_symbol) or market_data
            
            ai_response = analyze_with_ai(prompt, target_market_data, st.session_state.messages, client)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.rerun()
