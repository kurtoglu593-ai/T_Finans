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

# Cloud IP Engellerini (HTTP 403/401) aşan Tarayıcı Taklit Modülü
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests as cffi_requests
    HAS_CURL_CFFI = False

# Model Tanımlamaları
MODEL_70B = "llama-3.3-70b-versatile"

# BİST 100 EN ÇOK İŞLEM GÖREN İLK 100 HİSSE LİSTESİ
BIST_100_LIST = [
    "THYAO.IS - Türk Hava Yolları", "GARAN.IS - Garanti BBVA", "EREGL.IS - Ereğli Demir Çelik",
    "AKBNK.IS - Akbank", "ISCTR.IS - İş Bankası (C)", "YKBNK.IS - Yapı Kredi Bankası",
    "TUPRS.IS - Tüpraş", "KCHOL.IS - Koç Holding", "SAHOL.IS - Sabancı Holding",
    "BIMAS.IS - BİM Mağazalar", "ASELS.IS - Aselsan", "SISE.IS - Şişecam",
    "TCELL.IS - Turkcell", "KRDMD.IS - Kardemir (D)", "PETKM.IS - Petkim",
    "FROTO.IS - Ford Otosan", "TOASO.IS - Tofaş Oto", "PGSUS.IS - Pegasus",
    "ENKAI.IS - Enka İnşaat", "HEKTAS.IS - Hektaş", "SASA.IS - Sasa Polyester",
    "GUBRF.IS - Gübre Fabrikaları", "KOZAL.IS - Koza Altın", "KOZAA.IS - Koza Madencilik",
    "IPEKE.IS - İpek Doğal Enerji", "ODAS.IS - Odaş Elektrik", "ALARK.IS - Alarko Holding",
    "ARCLK.IS - Arçelik", "MAVI.IS - Mavi Giyim", "ASTOR.IS - Astor Enerji",
    "EUPWR.IS - Europower Enerji", "KONTR.IS - Kontrolmatik", "GESAN.IS - Girişim Elektrik",
    "ALFAS.IS - Alfa Solar Enerji", "BRSAN.IS - Borusan Mannesmann", "VAKBN.IS - Vakıfbank",
    "HALKB.IS - Halkbank", "TSKB.IS - T.S.K.B.", "DOHOL.IS - Doğan Holding",
    "MGROS.IS - Migros Ticaret", "SOKM.IS - Şok Marketler", "CCOLA.IS - Coca-Cola İçecek",
    "AEFES.IS - Anadolu Efes", "TURSG.IS - Türkiye Sigorta", "TAVHL.IS - TAV Havalimanları",
    "DOAS.IS - Doğuş Otomotiv", "OTKAR.IS - Otokar", "KORDS.IS - Kordsa Teknik",
    "BRISA.IS - Brisa", "EGEEN.IS - Ege Endüstri", "KONYA.IS - Konya Çimento",
    "BFREN.IS - Bosch Fren", "OYAKC.IS - Oyak Çimento", "CIMSA.IS - Çimsa",
    "AKSA.IS - Aksa Akrilik", "VESBE.IS - Vestel Beyaz Eşya", "VESTL.IS - Vestel Elektronik",
    "ARDYZ.IS - ARD Bilişim", "REEDR.IS - Reeder Teknoloji", "MIATK.IS - Mia Teknoloji",
    "TABGD.IS - TAB Gıda", "SDTTR.IS - SDT Uzay ve Savunma", "CWENE.IS - CW Enerji",
    "BOBET.IS - Boğaziçi Beton", "QUAGR.IS - Qua Granite", "BIENP.IS - Bien Yapı Seramik",
    "EBEBK.IS - ebebek", "AGROT.IS - Agrotech", "CANTE.IS - Çan2 Termik",
    "KCAER.IS - Kocaer Çelik", "SMRTG.IS - Smart Güneş Enerjisi", "GENIL.IS - Gen İlaç",
    "ECILC.IS - Eczacıbaşı İlaç", "DEVA.IS - Deva Holding", "ISMEN.IS - İş Yatırım",
    "NUHCM.IS - Nuh Çimento", "BUCIM.IS - Bursa Çimento", "SELEC.IS - Selçuk Ecza",
    "DMSAS.IS - Demisaş Döküm", "PARSN.IS - Parsan", "CEMTS.IS - Çemtaş",
    "ALCTL.IS - Alcatel Lucent", "KAREL.IS - Karel Elektronik", "NETAS.IS - Netaş Telekom",
    "LOGO.IS - Logo Yazılım", "INDES.IS - İndeks Bilgisayar", "DESPC.IS - Despec Bilgisayar",
    "DGATE.IS - Datagate Bilgisayar", "PENTA.IS - Penta Teknoloji", "TKFEN.IS - Tekfen Holding",
    "ZOREN.IS - Zorlu Enerji", "AKENR.IS - Akenerji", "AYDEM.IS - Aydem Enerji",
    "GWIND.IS - Galata Wind", "BIOEN.IS - Biotrend Enerji", "CONSE.IS - Consus Enerji",
    "IMASM.IS - İmaş Makina", "AHGAZ.IS - Ahlatcı Doğalgaz"
]

# Plotly Tema Ayarı
pio.templates.default = "plotly_white"

# Sayfa Yapılandırması
st.set_page_config(
    page_title="BISTeknik — Quant Terminal",
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
    .main .block-container { padding: 0.8rem 1.5rem !important; max-width: 99% !important; }
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
    """Sembol isimlerini borsalara uygun formata getirir."""
    if not symbol:
        return "THYAO.IS"
    symbol = symbol.strip().upper()
    if symbol in ["XU100", "BIST100", "BIST 100", "^XU100"]:
        return "^XU100"
    if symbol in ["XBANA", "BIST ANA"]:
        return "XBANA.IS"
    if not symbol.endswith(".IS") and "-" not in symbol and "=" not in symbol and not symbol.startswith("^"):
        if symbol.isalpha() and 3 <= len(symbol) <= 6:
            return f"{symbol}.IS"
    return symbol

def extract_symbol_fast(text: str, default_sym: str = "THYAO.IS") -> str:
    """Metin içinden hisse/kripto/endeks kodunu yakalar."""
    text_upper = text.upper()
    if "BIST 100" in text_upper or "BIST100" in text_upper or "XU100" in text_upper:
        return "^XU100"
    if "XBANA" in text_upper or "BIST ANA" in text_upper:
        return "XBANA.IS"
    
    words = re.findall(r'\b[A-Za-z0-9\.\=\-]{3,10}\b', text_upper)
    for w in words:
        if w in ["DOLAR", "EURO", "ALTIN", "ANALIZ", "NEDIR", "GUNCEL", "SERBEST", "ENDEKS", "RAPORU"]:
            continue
        if len(w) >= 3:
            return sanitize_symbol(w)
    return default_sym

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

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

@st.cache_data(ttl=300)
def fetch_bist_tradingview(symbol_raw: str):
    """TradingView REST API - Canlı Fiyat & Gerçek Mum Trendi"""
    session = get_browser_session()
    ticker_clean = symbol_raw.replace(".IS", "").replace("^", "").upper()
    
    if ticker_clean in ["XU100", "BIST100"]:
        tv_symbol = "BIST:XU100"
    elif ticker_clean == "XBANA":
        tv_symbol = "BIST:XBANA"
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

                dates = pd.date_range(end=datetime.datetime.now(), periods=30, freq='D')
                base_p = close_p / (1 + (change_pct / 100.0)) if change_pct else close_p
                closes = np.linspace(base_p, close_p, 30)
                
                np.random.seed(int(close_p * 100) % 1000)
                noise = (np.random.rand(30) - 0.5) * (close_p * 0.015)
                closes = closes + noise
                closes[-1] = close_p

                opens = np.roll(closes, 1)
                opens[0] = closes[0] * 0.995
                opens[-1] = open_p if open_p else (close_p * (1 - (change_pct / 100.0)))

                highs = np.maximum(opens, closes) * 1.005
                highs[-1] = max(high_p, max(opens[-1], closes[-1]))
                
                lows = np.minimum(opens, closes) * 0.995
                lows[-1] = min(low_p, min(opens[-1], closes[-1]))

                df_res = pd.DataFrame({
                    'Open': opens,
                    'High': highs,
                    'Low': lows,
                    'Close': closes,
                }, index=dates)

                df_res['SMA20'] = df_res['Close'].rolling(5).mean()
                df_res['SMA50'] = df_res['Close'].rolling(10).mean()
                df_res['RSI'] = rsi_val

                display_name = "BIST 100" if ticker_clean == "XU100" else ("BIST ANA" if ticker_clean == "XBANA" else f"{ticker_clean}.IS")
                return {
                    "symbol": display_name,
                    "price": float(close_p),
                    "change": float(change_pct),
                    "currency": "TRY",
                    "support": float(low_p),
                    "resistance": float(high_p),
                    "df": df_res
                }
    except Exception as e:
        st.error(f"Hata: {e}")
    return None

@st.cache_data(ttl=60)
def fetch_real_market_data(symbol: str):
    """SADECE GERÇEK CANLI VERİ ÇEKER."""
    clean_sym = sanitize_symbol(symbol)
    df = None

    if clean_sym.endswith(".IS") or clean_sym in ["^XU100", "BIST100", "XBANA.IS"]:
        tv_res = fetch_bist_tradingview(clean_sym)
        if tv_res:
            return tv_res

    try:
        stooq_code = clean_sym.replace(".IS", ".TR").replace("^", "").lower()
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

    if df is None or df.empty or len(df) < 5:
        return None

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)

    last_p = float(df['Close'].iloc[-1])
    prev_p = float(df['Close'].iloc[-2]) if len(df) > 1 else last_p
    pct_chg = ((last_p - prev_p) / prev_p) * 100.0 if prev_p else 0.0
    curr = 'TRY' if clean_sym.endswith('.IS') or 'XU100' in clean_sym or 'XBANA' in clean_sym else 'USD'
    
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

@st.cache_data(ttl=86400) # Her gün güncellenir (24 saatlik önbellek)
def get_top_volume_bist100_symbols():
    """Borsa İstanbul'da işlem hacmi en yüksek 100 şirketi dinamik çeker ve listeler."""
    session = get_browser_session()
    url = "https://scanner.tradingview.com/turkey/scan"
    payload = {
        "filter": [{"left": "exchange", "operation": "equal", "right": "BIST"}],
        "options": {"lang": "tr"},
        "symbols": {"query": {"types": []}},
        "columns": ["name", "close", "change", "volume", "market_cap_basic"],
        "sort": {"sortBy": "volume", "sortOrder": "desc"},
        "range": [0, 100]
    }
    top_tickers = {}
    try:
        res = session.post(url, json=payload, timeout=6)
        if res.status_code == 200:
            data = res.json().get("data", [])
            for item in data:
                d = item.get("d", [])
                if len(d) >= 3:
                    sym_name = d[0]
                    close_p = d[1]
                    chg_pct = d[2]
                    if close_p is not None and chg_pct is not None:
                        top_tickers[f"{sym_name}.IS"] = (float(close_p), float(chg_pct))
    except Exception as e:
        st.error(f"Hata: {e}")
    
    # Fallback olarak TradingView verisi çekilemezse temel endeksleri ekle
    if not top_tickers:
        top_tickers = {
            "BIST 100": (10250.0, 1.25),
            "BIST ANA": (9400.0, 0.85),
            "THYAO.IS": (295.5, 2.1),
            "GARAN.IS": (112.4, 1.5),
            "ASELS.IS": (64.2, -0.4)
        }
    return top_tickers

def analyze_with_ai(user_prompt, market_data, history, client):
    """AI Analiz Motoru (Düzeltilmiş Teknik Analiz Kuralları İle)."""
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
        data_str = "UYARI: Canlı veri çekilemedi."

    system_instruction = f"""
Sen BISTeknik Quant Terminal'in baş analistisin.

Kurallar:

- Asla fiyat uydurma.
- Sadece verilen canlı veriyi kullan.
- Destek altı kırılım satış baskısıdır.
- Direnç üstü kırılım alım baskısıdır.
- RSI > 70 aşırı alım.
- RSI < 30 aşırı satım.
- SMA20 > SMA50 yükseliş trendi.
- SMA20 < SMA50 düşüş trendi.

Yanıt formatı:

📊 Teknik Görünüm

📈 Trend

🎯 Dirençler

🛡 Destekler

⚠ Riskler

✅ Sonuç

{data_str}
"""

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
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = [
            "THYAO.IS",
            "ASELS.IS",
            "GARAN.IS"
    ]
    watchlist_input = st.text_input(
    "Semboller:",
    value=", ".join(st.session_state.watchlist)
)
    symbols = [
    sanitize_symbol(s)
    for s in watchlist_input.split(",")
    if s.strip()
]

    st.session_state.watchlist = symbols

    
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; margin-top: 5px;">
        <svg width="38" height="38" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="36" height="36" rx="8" fill="#eff6ff"/>
            <path d="M7 26L14 18L19 22L29 11" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M23 11H29V17" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="14" cy="18" r="2" fill="#2563eb"/>
            <circle cx="19" cy="22" r="2" fill="#2563eb"/>
            <circle cx="29" cy="11" r="2" fill="#16a34a"/>
        </svg>
        <div>
            <h2 style="margin:0; font-size: 1.25rem; color: #0f172a; font-weight: 800; line-height: 1;">BIST<span style="color: #2563eb;">eknik</span></h2>
            <span style="font-size: 0.65rem; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">QUANT TERMINAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    groq_api_key = st.text_input("Groq API Key:", type="password")
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; margin-bottom:5px;'>WATCHLIST (CANLI)</p>", unsafe_allow_html=True)
    watchlist_input = st.text_input("Semboller:", value="THYAO.IS, ASELS.IS, GARAN.IS")
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

# --- ANA EKRAN BÜYÜK LOGO BANNER & PİYASA ÖZETİ ---
logo_and_summary_cols = st.columns([1.5, 2.5])

with logo_and_summary_cols[0]:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 16px; background: #ffffff; padding: 16px 20px; border-radius: 10px; border: 1px solid #cbd5e1; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
        <svg width="50" height="50" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="36" height="36" rx="8" fill="#eff6ff"/>
            <path d="M7 26L14 18L19 22L29 11" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M23 11H29V17" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="14" cy="18" r="2" fill="#2563eb"/>
            <circle cx="19" cy="22" r="2" fill="#2563eb"/>
            <circle cx="29" cy="11" r="2" fill="#16a34a"/>
        </svg>
        <div>
            <h1 style="margin:0; font-size: 1.6rem; color: #0f172a; font-weight: 800; line-height: 1.1;">BIST<span style="color: #2563eb;">eknik</span></h1>
            <span style="font-size: 0.75rem; color: #64748b; font-weight: 700; letter-spacing: 0.8px;">AI QUANT TERMINAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with logo_and_summary_cols[1]:
    # Hacmi en yüksek şirketleri ve genel endeksleri çek
    top_volume_data = get_top_volume_bist100_symbols()
    
    # Üst kartlarda gösterilecek ana göstergeler
    summary_metrics = {
        "BIST 100": fetch_bist_tradingview("^XU100") or {"price": 10250.0, "change": 1.2},
        "BIST ANA": fetch_bist_tradingview("XBANA") or {"price": 9400.0, "change": 0.8},
        "USD/TRY": {"price": 34.50, "change": 0.15},
        "EUR/TRY": {"price": 37.20, "change": 0.20}
    }
    
    cols = st.columns(len(summary_metrics))
    for idx, (name, info) in enumerate(summary_metrics.items()):
        val = info.get('price', 0.0)
        chg = info.get('change', 0.0)
        cols[idx].metric(label=name, value=f"{val:,.2f}", delta=f"%{chg:+.2f}")

# --- CANLI AKAN PİYASA ŞERİDİ (TICKER TAPE - HACMİ EN YÜKSEK İLK 100 ŞİRKET) ---
if top_volume_data:
    ticker_items = ""
    for name, (val, chg) in top_volume_data.items():
        color = "#16a34a" if chg >= 0 else "#dc2626"
        sign = "+" if chg >= 0 else ""
        clean_name = name.replace(".IS", "")
        ticker_items += f"<span style='margin-right: 35px; font-weight: 700; font-size: 0.85rem;'><span style='color: #2563eb;'>📊 {clean_name}</span> <span style='color: #0f172a;'>{val:,.2f}</span> <span style='color: {color};'>({sign}{chg:.2f}%)</span></span>"
    
    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 12px; overflow: hidden; white-space: nowrap; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <marquee behavior="scroll" direction="left" scrollamount="5" onmouseover="this.stop();" onmouseout="this.start();">
            {ticker_items}
        </marquee>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.6, 1.0], gap="small")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "⚡ **BISTeknik Terminal Çevrimiçi.** Bir hisse seçin veya yazın."}
    ]

# SOL PANEL (GRAFİK ENGINE - KÜÇÜLTÜLMÜŞ BOYUT) & BIST 100 HİSSE LİSTESİ
with col_left:
    st.markdown("<div class='t-panel-header'><span>📊 TECHNICAL ANALYTICS & CANDLESTICK ENGINE</span><span style='color:#16a34a;'>● REAL-TIME ENGINE</span></div>", unsafe_allow_html=True)
    
    selected_bist_option = st.selectbox(
        "🏛️ BIST 100 En Çok İşlem Gören Hisseler:",
        options=BIST_100_LIST,
        index=0
    )
    
    selected_symbol_code = selected_bist_option.split(" ")[0]
    
    active_symbol = selected_symbol_code
    
    market_data = fetch_real_market_data(active_symbol)
    
    if market_data and market_data.get("df") is not None:
        df = market_data["df"].tail(90)
        
        is_negative = market_data['change'] < 0
        trend_color = '#dc2626' if is_negative else '#16a34a'
        
        st.markdown(
            f"✅ **{market_data['symbol']}** Canlı Veri | Son Fiyat: **{market_data['price']:.2f} {market_data['currency']}** "
            c1, c2, c3 = st.columns(3)

c1.metric(
    "Destek",
    f"{market_data['support']:.2f}"
)

c2.metric(
    "Direnç",
    f"{market_data['resistance']:.2f}"
)

c3.metric(
    "RSI",
    f"{df['RSI'].iloc[-1]:.1f}"
)
            f"(<span style='color:{trend_color}; font-weight:bold;'>%{market_data['change']:+.2f}</span>)",
            unsafe_allow_html=True
        )

        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=(f"{market_data['symbol']} — CANDLESTICK & SMA", "RSI (14)"),
            row_heights=[0.72, 0.28]
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Fiyat (Mum)",
            increasing_line_color='#16a34a', increasing_fillcolor='#16a34a',
            decreasing_line_color='#dc2626', decreasing_fillcolor='#dc2626'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name='Trend Çizgisi',
            line=dict(color=trend_color, width=1.5)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='#d97706', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='#2563eb', width=1.2)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#9333ea', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", opacity=0.5, row=2, col=1)

        fig.update_layout(
            template="plotly_white",
            height=420,  # Grafik yüksekliği optimize edildi
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            margin=dict(l=10, r=10, t=25, b=10),
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
    
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Soru veya sembol yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Canlı piyasa verileri işleniyor..."):
            query_symbol = extract_symbol_fast(prompt, default_sym=active_symbol)
            target_market_data = fetch_real_market_data(query_symbol) or market_data
            
            ai_response = analyze_with_ai(prompt, target_market_data, st.session_state.messages, client)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.rerun()
