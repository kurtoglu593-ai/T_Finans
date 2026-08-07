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
import concurrent.futures
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# 📍 Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 📍 Config sınıfı
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"
    TV_SCAN_URL = "https://scanner.tradingview.com/turkey/scan"
    STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
    FX_BASE_URL = "https://api.frankfurter.app"
    
    ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
    FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")

    CACHE_TTL_SHORT = 300
    CACHE_TTL_MEDIUM = 3600
    CACHE_TTL_LONG = 86400

    RSI_PERIOD = 14
    SMA_FAST = 20
    SMA_SLOW = 50

    CHART_HEIGHT = 450
    MAX_HISTORY_DAYS = 90

# Cloud IP Engellerini aşan Tarayıcı Taklit Modülü
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests as cffi_requests
    HAS_CURL_CFFI = False

# Yahoo Finance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Alpha Vantage
try:
    from alpha_vantage.timeseries import TimeSeries
    HAS_ALPHA_VANTAGE = True if Config.ALPHA_VANTAGE_KEY else False
except ImportError:
    HAS_ALPHA_VANTAGE = False

# Finnhub
try:
    import finnhub
    HAS_FINNHUB = True if Config.FINNHUB_KEY else False
except ImportError:
    HAS_FINNHUB = False

MODEL_70B = Config.GROQ_MODEL

# BİST 100 LİSTESİ
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

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="BISTeknik — Quant Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARK TEMAS ---
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.3) !important;
    }

    header, footer { display: none !important; }

    .main .block-container {
        padding: 0.8rem 1.5rem !important;
        max-width: 99% !important;
        background: transparent !important;
    }

    [data-testid="stMetric"], .t-panel-header {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(37, 99, 235, 0.15) !important;
    }

    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        backdrop-filter: blur(10px) !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdown"] {
        color: #f1f5f9 !important;
    }

    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        position: relative !important;
        bottom: 0 !important;
        margin-top: 10px !important;
    }

    [data-testid="stChatInput"] input {
        color: #f1f5f9 !important;
        background: transparent !important;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 0.75rem !important;
        padding: 0.5rem 1rem !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4) !important;
    }

    .t-panel-header {
        background: rgba(255, 255, 255, 0.04) !important;
        border-bottom: 2px solid #2563eb !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 10px 18px !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 0 !important;
    }

    .t-panel-header span:last-child {
        color: #3b82f6 !important;
        font-size: 0.7rem !important;
        background: rgba(37, 99, 235, 0.15) !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
    }

    .stSelectbox label {
        color: rgba(255, 255, 255, 0.7) !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stSelectbox select {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    .ticker-tape {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 8px 16px !important;
        backdrop-filter: blur(10px) !important;
        margin-bottom: 12px !important;
    }

    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        border-radius: 10px;
    }

    .stSpinner > div {
        border-color: #2563eb !important;
    }

    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        color: #f1f5f9 !important;
        border-radius: 12px !important;
    }

    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        font-size: 0.8rem !important;
    }

    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    .data-source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        margin-left: 8px;
        background: rgba(37, 99, 235, 0.2);
        color: #3b82f6;
        border: 1px solid rgba(37, 99, 235, 0.3);
    }
    
    .data-source-badge.yahoo {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border-color: rgba(34, 197, 94, 0.3);
    }
    
    .data-source-badge.stooq {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border-color: rgba(245, 158, 11, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- YARDIMCI FONKSİYONLAR ---

def sanitize_symbol(symbol: str) -> str:
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

def calculate_rsi(series, period=Config.RSI_PERIOD):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.where(loss != 0, 0.00001)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).clip(0, 100)

def determine_trend(price, sma20, sma50):
    if sma20 is None or pd.isna(sma20):
        return "VERİ YOK", "❓"

    if sma50 is None or pd.isna(sma50):
        if price > sma20:
            return "YÜKSELİŞ (Kısa Vade)", "📈"
        elif price < sma20:
            return "DÜŞÜŞ (Kısa Vade)", "📉"
        else:
            return "YATAY", "➡️"

    if price > sma20 and price > sma50:
        if sma20 > sma50:
            return "GÜÇLÜ YÜKSELİŞ", "📈"
        else:
            return "YÜKSELİŞ (Dönüş)", "📈"
    elif price < sma20 and price < sma50:
        if sma20 < sma50:
            return "GÜÇLÜ DÜŞÜŞ", "📉"
        else:
            return "DÜŞÜŞ (Dönüş)", "📉"
    else:
        return "YATAY", "➡️"

def get_rsi_comment(rsi_value):
    if rsi_value > 70:
        return "Aşırı Alım", "inverse"
    elif rsi_value < 30:
        return "Aşırı Satım", "normal"
    elif 40 <= rsi_value <= 60:
        return "Nötr", "off"
    elif rsi_value > 60:
        return "Alım Bölgesi", "normal"
    else:
        return "Satım Bölgesi", "inverse"

_browser_session = None

def get_browser_session():
    global _browser_session
    if _browser_session is None:
        if HAS_CURL_CFFI:
            _browser_session = cffi_requests.Session(impersonate="chrome120")
        else:
            _browser_session = cffi_requests.Session()
            _browser_session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
    return _browser_session

# --- VERİ KAYNAKLARI ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def fetch_tv_quote(tv_symbol: str) -> Optional[Dict[str, Any]]:
    session = get_browser_session()
    payload = {
        "symbols": {"tickers": [tv_symbol]},
        "columns": ["name", "close", "change", "open", "high", "low", "volume", "RSI"]
    }
    try:
        res = session.post(Config.TV_SCAN_URL, json=payload, timeout=5)
        if res.status_code != 200:
            return None

        data = res.json().get("data", [])
        if not data:
            return None

        d = data[0].get("d", [])
        if len(d) < 6 or d[1] is None:
            return None

        close_p = float(d[1])
        return {
            "price": close_p,
            "change": float(d[2]) if d[2] is not None else 0.0,
            "open": float(d[3]) if d[3] is not None else close_p,
            "high": float(d[4]) if d[4] is not None else close_p,
            "low": float(d[5]) if d[5] is not None else close_p,
            "rsi": float(d[7]) if len(d) > 7 and d[7] is not None else None,
        }
    except Exception as e:
        logger.error(f"TradingView hatası ({tv_symbol}): {e}")
        return None

def tv_symbol_for(clean_sym: str) -> str:
    ticker_clean = clean_sym.replace(".IS", "").replace("^", "").upper()
    if ticker_clean in ["XU100", "BIST100"]:
        return "BIST:XU100"
    if ticker_clean in ["XBANA"]:
        return "BIST:XBANA"
    return f"BIST:{ticker_clean}"

def fetch_stooq_data(symbol: str):
    try:
        stooq_code = symbol.replace(".IS", ".TR").replace("^", "").lower()
        stooq_url = f"{Config.STOOQ_BASE_URL}?s={stooq_code}&i=d"
        df_stooq = pd.read_csv(stooq_url)

        if not df_stooq.empty and 'Close' in df_stooq.columns and len(df_stooq) > 5:
            df_stooq['Date'] = pd.to_datetime(df_stooq['Date'])
            df_stooq.set_index('Date', inplace=True)
            df_stooq.sort_index(inplace=True)

            for col in ['Open', 'High', 'Low', 'Close']:
                df_stooq[col] = pd.to_numeric(df_stooq[col], errors='coerce')

            df_stooq.dropna(subset=['Close'], inplace=True)

            if len(df_stooq) > 5:
                return df_stooq[['Open', 'High', 'Low', 'Close']]
    except Exception as e:
        logger.error(f"Stooq hatası ({symbol}): {e}")
    return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_yahoo_data(symbol: str) -> Optional[Dict[str, Any]]:
    if not HAS_YFINANCE:
        return None
        
    try:
        yahoo_symbol = symbol.replace(".IS", ".IS")
        ticker = yf.Ticker(yahoo_symbol)
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            return None
            
        hist = ticker.history(period="3mo")
        
        if hist.empty:
            return {
                "symbol": symbol,
                "price": float(info.get('regularMarketPrice', 0)),
                "change": float(info.get('regularMarketChangePercent', 0)),
                "currency": "TRY",
                "support": float(info.get('dayLow', 0)),
                "resistance": float(info.get('dayHigh', 0)),
                "df": None,
                "data_source": "Yahoo Finance (sadece canlı)"
            }
        
        df = hist[['Open', 'High', 'Low', 'Close']].copy()
        df.index = pd.to_datetime(df.index)
        df['SMA20'] = df['Close'].rolling(Config.SMA_FAST).mean()
        df['SMA50'] = df['Close'].rolling(Config.SMA_SLOW).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        return {
            "symbol": symbol,
            "price": float(df['Close'].iloc[-1]),
            "change": float(info.get('regularMarketChangePercent', 0)),
            "currency": "TRY",
            "support": float(df['Low'].tail(20).min()),
            "resistance": float(df['High'].tail(20).max()),
            "df": df,
            "data_source": "Yahoo Finance"
        }
    except Exception as e:
        logger.error(f"Yahoo Finance hatası ({symbol}): {e}")
        return None

def standardize_data(data: Dict, symbol: str) -> Optional[Dict]:
    try:
        if 'df' in data and isinstance(data['df'], pd.DataFrame):
            if 'currency' not in data:
                data['currency'] = 'TRY' if '.IS' in symbol else 'USD'
            if 'support' not in data and 'df' in data and not data['df'].empty:
                data['support'] = float(data['df']['Low'].tail(20).min())
            if 'resistance' not in data and 'df' in data and not data['df'].empty:
                data['resistance'] = float(data['df']['High'].tail(20).max())
            return data
        
        if 'price' in data and 'change' in data:
            if 'df' not in data:
                today = pd.Timestamp(datetime.datetime.now().date())
                df_single = pd.DataFrame({
                    'Open': [data.get('open', data['price'])],
                    'High': [data.get('high', data['price'])],
                    'Low': [data.get('low', data['price'])],
                    'Close': [data['price']]
                }, index=[today])
                data['df'] = df_single
            
            return {
                "symbol": symbol,
                "price": float(data['price']),
                "change": float(data['change']),
                "currency": "TRY" if '.IS' in symbol else "USD",
                "support": float(data.get('support', data.get('low', data['price'] * 0.95))),
                "resistance": float(data.get('resistance', data.get('high', data['price'] * 1.05))),
                "df": data['df'],
                "data_source": data.get('data_source', 'Standardized')
            }
    except Exception as e:
        logger.error(f"Standardizasyon hatası: {e}")
        return None

class MarketDataManager:
    def __init__(self):
        self.cache = {}
        self.last_success = {}
        self.data_sources = []
        
        if HAS_YFINANCE:
            self.data_sources.append(('yahoo', fetch_yahoo_data))
        self.data_sources.append(('stooq', fetch_stooq_data))
        self.data_sources.append(('tradingview', lambda x: fetch_tv_quote(tv_symbol_for(x))))
        
    def get_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        cache_key = f"{symbol}_{datetime.datetime.now().date()}"
        if not force_refresh and cache_key in self.cache:
            return self.cache[cache_key]
        
        clean_sym = sanitize_symbol(symbol)
        
        for source_name, fetch_func in self.data_sources:
            try:
                result = fetch_func(clean_sym)
                if result and isinstance(result, dict):
                    standardized = standardize_data(result, clean_sym)
                    if standardized:
                        self.cache[cache_key] = standardized
                        self.last_success[symbol] = datetime.datetime.now()
                        return standardized
            except Exception as e:
                logger.warning(f"{source_name} başarısız: {e}")
                continue
        
        if symbol in self.last_success:
            age = (datetime.datetime.now() - self.last_success[symbol]).seconds / 60
            if age < 15:
                return self.cache.get(cache_key)
        
        return None

data_manager = MarketDataManager()

def fetch_real_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    return data_manager.get_data(symbol)

def validate_market_data(data: Dict[str, Any]) -> bool:
    required_fields = ['symbol', 'price', 'change', 'df']
    if not all(field in data for field in required_fields):
        return False
    df = data['df']
    if df is None or df.empty:
        return False
    if abs(data['change']) > 20:
        return False
    if data['price'] <= 0:
        return False
    return True

def fetch_bist_ana_data():
    try:
        live = fetch_tv_quote("BIST:XBANA")
        if live:
            return {
                "symbol": "BIST ANA",
                "price": live['price'],
                "change": live['change'],
                "currency": "TRY",
                "support": live['low'],
                "resistance": live['high'],
                "df": None
            }
    except:
        pass
    
    if HAS_YFINANCE:
        try:
            yahoo_data = fetch_yahoo_data("XBANA.IS")
            if yahoo_data:
                return {
                    "symbol": "BIST ANA",
                    "price": yahoo_data['price'],
                    "change": yahoo_data['change'],
                    "currency": "TRY",
                    "support": yahoo_data.get('support', 0),
                    "resistance": yahoo_data.get('resistance', 0),
                    "df": yahoo_data.get('df')
                }
        except:
            pass
    return None

@st.cache_data(ttl=Config.CACHE_TTL_SHORT)
def fetch_fx_rate(pair_from: str, pair_to: str) -> Optional[Dict[str, float]]:
    try:
        session = get_browser_session()
        today = datetime.date.today()
        start = today - datetime.timedelta(days=6)
        url = f"{Config.FX_BASE_URL}/{start.isoformat()}..{today.isoformat()}?from={pair_from}&to={pair_to}"
        res = session.get(url, timeout=5)
        if res.status_code != 200:
            return None
        rates = res.json().get("rates", {})
        if not rates:
            return None
        dates_sorted = sorted(rates.keys())
        first_rate = rates[dates_sorted[0]].get(pair_to)
        last_rate = rates[dates_sorted[-1]].get(pair_to)
        if first_rate is None or last_rate is None:
            return None
        chg = ((last_rate - first_rate) / first_rate) * 100.0 if first_rate else 0.0
        return {"price": float(last_rate), "change": float(chg)}
    except Exception as e:
        logger.error(f"Döviz kuru hatası ({pair_from}/{pair_to}): {e}")
        return None

# --- AI ANALİZ ---
def analyze_with_ai(user_prompt: str, market_data: Optional[Dict[str, Any]], history: list, client) -> str:
    if market_data and market_data.get('df') is not None:
        df = market_data['df']

        if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]):
            last_rsi = float(df['RSI'].iloc[-1])
        else:
            last_rsi = 50.0

        sma20 = df['SMA20'].iloc[-1] if 'SMA20' in df and not pd.isna(df['SMA20'].iloc[-1]) else None
        sma50 = df['SMA50'].iloc[-1] if 'SMA50' in df and not pd.isna(df['SMA50'].iloc[-1]) else None
        last_close = float(market_data['price'])

        trend_text, trend_icon = determine_trend(last_close, sma20, sma50)
        rsi_comment, _ = get_rsi_comment(last_rsi)

        sma20_str = f"{sma20:.2f}" if sma20 is not None else "Yetersiz geçmiş veri"
        sma50_str = f"{sma50:.2f}" if sma50 is not None else "Yetersiz geçmiş veri"
        currency = market_data['currency']
        data_source = market_data.get('data_source', 'bilinmiyor')

        data_str = (
            f"📊 GERÇEK VERİLER (kaynak: {data_source}):\n"
            f"- Sembol: {market_data['symbol']}\n"
            f"- Fiyat: {market_data['price']:.2f} {currency}\n"
            f"- Değişim: %{market_data['change']:+.2f}\n"
            f"- RSI(14): {last_rsi:.1f} ({rsi_comment})\n"
            f"- SMA20: {sma20_str}\n"
            f"- SMA50: {sma50_str}\n"
            f"- Trend: {trend_icon} {trend_text}\n"
            f"- Destek: {market_data['support']:.2f}\n"
            f"- Direnç: {market_data['resistance']:.2f}"
        )
    else:
        data_str = "⚠️ Canlı veri çekilemedi. Geçerli bir sembol girin."

    system_instruction = (
        "Sen profesyonel bir borsa analistisin.\n"
        "Sadece verilen GERÇEK VERİLERE göre analiz yap.\n"
        "Cevabını 3-4 paragrafta net ve öz ver.\n"
        "Türkçe yaz, teknik terimleri doğru kullan.\n\n"
        f"Mevcut Veri:\n{data_str}"
    )

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        res = client.chat.completions.create(
            model=MODEL_70B,
            messages=messages,
            temperature=0.1,
            max_tokens=400
        )
        return res.choices[0].message.content
    except Exception as err:
        return f"⚠️ AI Analiz Hatası: {err}"

@st.cache_data(ttl=Config.CACHE_TTL_MEDIUM)
def get_top_volume_bist100_symbols():
    session = get_browser_session()
    url = Config.TV_SCAN_URL
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
        logger.error(f"Hacim sıralaması hatası: {e}")

    return top_tickers

def fetch_multiple_symbols(symbols: list) -> dict:
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(fetch_real_market_data, sym): sym 
            for sym in symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                logger.error(f"{symbol} çekilemedi: {e}")
                results[symbol] = None
    return results

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; margin-top: 5px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
        <div style="width: 42px; height: 42px; background: linear-gradient(135deg, #2563eb, #8b5cf6); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0;">
            ⚡
        </div>
        <div>
            <h2 style="margin:0; font-size: 1.2rem; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; line-height: 1.2;">BISTeknik</h2>
            <span style="font-size: 0.6rem; color: rgba(255,255,255,0.4); font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">QUANT TERMINAL v3.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key:", type="password", key="groq_key_input")

    st.markdown("---")
    
    # Veri Kaynağı Durumu
    st.markdown("<p style='font-size: 0.65rem; font-weight: 600; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>🔌 VERİ KAYNAKLARI</p>", unsafe_allow_html=True)
    
    st.caption("✅ Yahoo Finance" if HAS_YFINANCE else "❌ Yahoo Finance (kurulu değil)")
    st.caption("✅ Stooq")
    st.caption("✅ TradingView")
    
    st.markdown("---")
    
    # 💼 PORTFÖY TAKİBİ (YENİ)
    st.markdown("<p style='font-size: 0.65rem; font-weight: 600; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>💼 PORTFÖY</p>", unsafe_allow_html=True)
    
    portfolio_input = st.text_area("Hisse:Adet (örn: THYAO.IS:100)", height=80, key="portfolio_input")
    
    if st.button("📊 Hesapla", use_container_width=True):
        if portfolio_input:
            total_value = 0
            portfolio_data = []
            
            for item in portfolio_input.split(','):
                if ':' in item:
                    sym, qty = item.strip().split(':')
                    sym = sanitize_symbol(sym)
                    qty = int(qty)
                    
                    data = fetch_real_market_data(sym)
                    if data:
                        value = data['price'] * qty
                        total_value += value
                        portfolio_data.append({
                            'Sembol': sym,
                            'Adet': qty,
                            'Fiyat': data['price'],
                            'Değer': value,
                            'Değişim': data['change']
                        })
            
            if portfolio_data:
                df_portfolio = pd.DataFrame(portfolio_data)
                st.dataframe(df_portfolio, use_container_width=True)
                st.metric("💰 Toplam", f"{total_value:,.2f} TRY")

    st.markdown("---")

    st.markdown("<p style='font-size: 0.65rem; font-weight: 600; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>📋 WATCHLIST</p>", unsafe_allow_html=True)
    watchlist_input = st.text_input("Semboller (virgülle):", value="THYAO.IS, ASELS.IS, GARAN.IS", label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 GÜNCELLE", use_container_width=True):
            symbols = [sanitize_symbol(s.strip()) for s in watchlist_input.split(",") if s.strip()]
            results = fetch_multiple_symbols(symbols)

            for i, sym in enumerate(symbols):
                res_data = results.get(sym)
                if res_data:
                    st.metric(
                        label=res_data['symbol'],
                        value=f"{res_data['price']:,.2f} {res_data['currency']}",
                        delta=f"%{res_data['change']:+.2f}"
                    )
                else:
                    st.caption(f"⚠️ {sym} veri yok")

    with col2:
        if st.button("📊 ANALİZ", use_container_width=True):
            st.info("Sağ panelden soru sorun")

# API key kontrolü
if not groq_api_key:
    st.warning("⚠️ **Groq API Key bulunamadı!** Sol menüden girin.")
    st.stop()

try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"❌ Groq client hatası: {e}")
    st.stop()

# --- ANA EKRAN ---
logo_and_summary_cols = st.columns([1.2, 2.8])

with logo_and_summary_cols[0]:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 16px 20px; margin-bottom: 15px; backdrop-filter: blur(10px);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #2563eb, #8b5cf6); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; flex-shrink: 0;">
                ⚡
            </div>
            <div>
                <h1 style="margin:0; font-size: 1.4rem; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; line-height: 1.1;">BISTeknik</h1>
                <span style="font-size: 0.6rem; color: rgba(255,255,255,0.3); font-weight: 500; letter-spacing: 0.5px;">AI QUANT TERMINAL v3.0</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with logo_and_summary_cols[1]:
    top_volume_data = get_top_volume_bist100_symbols()

    summary_metrics = {}

    bist100_live = None
    try:
        bist100_live = fetch_tv_quote("BIST:XU100")
    except:
        if HAS_YFINANCE:
            try:
                yahoo_bist = fetch_yahoo_data("XU100.IS")
                if yahoo_bist:
                    bist100_live = {"price": yahoo_bist['price'], "change": yahoo_bist['change']}
            except:
                pass

    summary_metrics["BIST 100"] = (
        {"price": bist100_live['price'], "change": bist100_live['change']}
        if bist100_live else {"price": None, "change": None}
    )

    bist_ana_data = fetch_bist_ana_data()
    summary_metrics["BIST ANA"] = (
        {"price": bist_ana_data['price'], "change": bist_ana_data['change']}
        if bist_ana_data else {"price": None, "change": None}
    )

    usd_try = fetch_fx_rate("USD", "TRY")
    summary_metrics["USD/TRY"] = usd_try if usd_try else {"price": None, "change": None}

    eur_try = fetch_fx_rate("EUR", "TRY")
    summary_metrics["EUR/TRY"] = eur_try if eur_try else {"price": None, "change": None}

    cols = st.columns(len(summary_metrics))
    for idx, (name, info) in enumerate(summary_metrics.items()):
        val = info.get('price')
        chg = info.get('change')
        if val is None:
            cols[idx].metric(label=name, value="—", delta=None)
        else:
            cols[idx].metric(
                label=name,
                value=f"{val:,.2f}",
                delta=f"%{chg:+.2f}",
                delta_color="normal" if chg >= 0 else "inverse"
            )

# Ticker Tape
if top_volume_data:
    ticker_items = ""
    for name, (val, chg) in top_volume_data.items():
        color = "#22c55e" if chg >= 0 else "#ef4444"
        sign = "+" if chg >= 0 else ""
        clean_name = name.replace(".IS", "")
        ticker_items += f"<span style='margin-right: 35px; font-weight: 700; font-size: 0.8rem;'><span style='color: #3b82f6;'>📊 {clean_name}</span> <span style='color: #f1f5f9;'>{val:,.2f}</span> <span style='color: {color};'>({sign}{chg:.2f}%)</span></span>"

    st.markdown(f"""
    <div class="ticker-tape">
        <marquee behavior="scroll" direction="left" scrollamount="5" onmouseover="this.stop();" onmouseout="this.start();">
            {ticker_items}
        </marquee>
    </div>
    """, unsafe_allow_html=True)
else:
    st.caption("⚠️ Hacim sıralaması alınamıyor.")

st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.6, 1.0], gap="small")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "⚡ **BISTeknik Terminal Çevrimiçi.** Bir hisse seçin veya yazın."}
    ]

# SOL PANEL
with col_left:
    st.markdown("""
    <div class='t-panel-header'>
        <span>📊 TECHNICAL ANALYTICS</span>
        <span>● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    selected_bist_option = st.selectbox(
        "🏛️ BIST 100:",
        options=BIST_100_LIST,
        index=0,
        key="bist_selector"
    )

    selected_symbol_code = selected_bist_option.split(" ")[0]

    last_user_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), selected_symbol_code)
    active_symbol = extract_symbol_fast(last_user_query, default_sym=selected_symbol_code)

    with st.spinner(f"📊 {active_symbol} verileri çekiliyor..."):
        market_data = fetch_real_market_data(active_symbol)

    if market_data and market_data.get("df") is not None:
        df = market_data["df"].tail(Config.MAX_HISTORY_DAYS)

        is_negative = market_data['change'] < 0
        trend_color = '#ef4444' if is_negative else '#22c55e'

        if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]):
            last_rsi = float(df['RSI'].iloc[-1])
        else:
            last_rsi = 50.0

        data_source = market_data.get('data_source', 'Bilinmiyor')
        source_class = "yahoo" if "Yahoo" in data_source else "stooq" if "Stooq" in data_source else ""
        
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 8px 14px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <span style="font-weight: 700; color: #f1f5f9;">{market_data['symbol']}</span>
                <span class="data-source-badge {source_class}">{data_source}</span>
                <span style="float: right; font-weight: 700; color: #f1f5f9; font-family: 'JetBrains Mono', monospace;">
                    {market_data['price']:.2f} {market_data['currency']}
                    <span style="color: {trend_color}; margin-left: 8px;">%{market_data['change']:+.2f}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if len(df) < 5:
            st.info("ℹ️ Geçmiş veri yetersiz, sadece canlı fiyat gösteriliyor.")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f"{market_data['symbol']} — CANDLESTICK & SMA", f"RSI (14) — {last_rsi:.1f}"),
            row_heights=[0.72, 0.28]
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Mum",
            increasing_line_color='#22c55e', increasing_fillcolor='#22c55e',
            decreasing_line_color='#ef4444', decreasing_fillcolor='#ef4444'
        ), row=1, col=1)

        if 'SMA20' in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA20'], mode='lines', name='SMA 20',
                line=dict(color='#f59e0b', width=1.2)
            ), row=1, col=1)

        if 'SMA50' in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA50'], mode='lines', name='SMA 50',
                line=dict(color='#3b82f6', width=1.2)
            ), row=1, col=1)

        if 'RSI' in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['RSI'], mode='lines', name='RSI',
                line=dict(color='#8b5cf6', width=1.5)
            ), row=2, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.3, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", opacity=0.3, row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=Config.CHART_HEIGHT,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1,
                font=dict(color="rgba(255,255,255,0.6)", size=10)
            ),
            font=dict(color="rgba(255,255,255,0.7)")
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{active_symbol}_{time.time()}")

        sma20_val = df['SMA20'].iloc[-1] if 'SMA20' in df and not pd.isna(df['SMA20'].iloc[-1]) else None
        sma50_val = df['SMA50'].iloc[-1] if 'SMA50' in df and not pd.isna(df['SMA50'].iloc[-1]) else None

        rsi_label, rsi_color = get_rsi_comment(last_rsi)
        trend_text, trend_icon = determine_trend(market_data['price'], sma20_val, sma50_val)

        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        with col_metric1:
            st.metric("RSI (14)", f"{last_rsi:.1f}", delta=rsi_label, delta_color=rsi_color)
        with col_metric2:
            st.metric("SMA 20", f"{sma20_val:.2f}" if sma20_val else "—")
        with col_metric3:
            st.metric("SMA 50", f"{sma50_val:.2f}" if sma50_val else "—")
        with col_metric4:
            st.metric("Trend", f"{trend_icon} {trend_text}", delta=f"%{market_data['change']:+.2f}")

        # 📈 GELİŞMİŞ GÖSTERGELER (YENİ)
        st.markdown("---")
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: rgba(255,255,255,0.6);'>📈 GELİŞMİŞ GÖSTERGELER</p>", unsafe_allow_html=True)
        
        df_tech = df.tail(50).copy()
        
        if len(df_tech) > 26:
            # MACD
            exp1 = df_tech['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df_tech['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            sma_bb = df_tech['Close'].rolling(20).mean()
            std_bb = df_tech['Close'].rolling(20).std()
            upper_band = sma_bb + (std_bb * 2)
            lower_band = sma_bb - (std_bb * 2)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                macd_val = macd.iloc[-1] if not macd.empty else 0
                signal_val = signal.iloc[-1] if not signal.empty else 0
                st.metric("MACD", f"{macd_val:.2f}", 
                         delta="Alım" if macd_val > signal_val else "Satım",
                         delta_color="normal" if macd_val > signal_val else "inverse")
            
            with col2:
                last_price = market_data['price']
                if not lower_band.empty and not upper_band.empty:
                    bb_position = ((last_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])) * 100
                    st.metric("Bollinger", f"%{bb_position:.1f}",
                             delta="Üst Bant" if bb_position > 80 else "Alt Bant" if bb_position < 20 else "Orta")
            
            with col3:
                # ATR
                high_low = df_tech['High'] - df_tech['Low']
                high_close = abs(df_tech['High'] - df_tech['Close'].shift())
                low_close = abs(df_tech['Low'] - df_tech['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(14).mean()
                atr_val = atr.iloc[-1] if not atr.empty else 0
                st.metric("ATR (14)", f"{atr_val:.2f}",
                         delta=f"%{(atr_val / market_data['price'] * 100):.2f}" if market_data['price'] > 0 else "0")

        # 📊 KARŞILAŞTIRMA (YENİ)
        with st.expander("📊 Hisse Karşılaştır", expanded=False):
            compare_symbols = st.text_input("Hisseler (virgülle):", 
                                           value="THYAO.IS, ASELS.IS, GARAN.IS", key="compare_input")
            
            if st.button("Karşılaştır", key="compare_btn"):
                symbols = [sanitize_symbol(s.strip()) for s in compare_symbols.split(",") if s.strip()]
                
                compare_data = {}
                for sym in symbols:
                    data = fetch_real_market_data(sym)
                    if data and data.get('df') is not None:
                        compare_data[sym] = data['df']['Close'].tail(30)
                
                if compare_data:
                    fig_compare = go.Figure()
                    for sym, series in compare_data.items():
                        normalized = (series / series.iloc[0] - 1) * 100
                        fig_compare.add_trace(go.Scatter(
                            x=normalized.index,
                            y=normalized,
                            mode='lines',
                            name=sym,
                            line=dict(width=2)
                        ))
                    
                    fig_compare.update_layout(
                        title="Performans Karşılaştırması (%)",
                        template="plotly_dark",
                        height=250,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.02)",
                        yaxis_title="Değişim (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

    else:
        st.error(f"❌ **{active_symbol}** için veri alınamadı.")
        st.info("💡 Sembol kodunu kontrol edin (örn: THYAO.IS)")

# SAĞ PANEL - CHAT
with col_right:
    st.markdown("""
    <div class='t-panel-header'>
        <span>🤖 AI QUANT ANALYST</span>
        <span>70B</span>
    </div>
    """, unsafe_allow_html=True)

    # Chat container
    chat_container = st.container(height=380)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input - DÜZELTİLDİ: Mesajlar artık gözüküyor
    prompt = st.chat_input("Soru veya sembol yazın...", key="chat_input")

    if prompt:
        # Mesajı hemen göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Chat container'ı güncelle
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        progress_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("🧠 AI analiz yapıyor..."):
            progress_bar.progress(30)
            progress_text.text("📊 Piyasa verileri işleniyor...")

            default_symbol = active_symbol if 'active_symbol' in dir() else "THYAO.IS"
            query_symbol = extract_symbol_fast(prompt, default_sym=default_symbol)

            target_market_data = fetch_real_market_data(query_symbol)
            
            if not target_market_data and 'market_data' in locals():
                target_market_data = market_data

            progress_bar.progress(60)
            progress_text.text("🤖 AI modeli çalıştırılıyor...")

            ai_response = analyze_with_ai(prompt, target_market_data, st.session_state.messages, client)

            progress_bar.progress(100)
            progress_text.text("✅ Analiz tamamlandı!")
            time.sleep(0.5)

            progress_bar.empty()
            progress_text.empty()

            # AI yanıtını ekle
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            # Chat container'ı güncelle
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            
            st.rerun()
