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
from typing import Optional, Dict, Any, Tuple, List
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
    # API ayarları
    GROQ_MODEL = "llama-3.3-70b-versatile"
    TV_SCAN_URL = "https://scanner.tradingview.com/turkey/scan"
    STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
    FX_BASE_URL = "https://api.frankfurter.app"
    
    # API Key'ler
    ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
    FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")

    # Cache süreleri
    CACHE_TTL_SHORT = 300
    CACHE_TTL_MEDIUM = 3600
    CACHE_TTL_LONG = 86400

    # Teknik indikatör parametreleri
    RSI_PERIOD = 14
    SMA_FAST = 20
    SMA_SLOW = 50

    # Grafik ayarları
    CHART_HEIGHT = 450
    MAX_HISTORY_DAYS = 90

# Cloud IP Engellerini aşan Tarayıcı Taklit Modülü
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
    logger.info("curl_cffi başarıyla yüklendi")
except ImportError:
    import requests as cffi_requests
    HAS_CURL_CFFI = False
    logger.warning("curl_cffi yüklü değil, standart requests kullanılıyor")

# 📍 YENİ: Yahoo Finance Entegrasyonu
try:
    import yfinance as yf
    HAS_YFINANCE = True
    logger.info("yfinance başarıyla yüklendi")
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance yüklü değil")

# 📍 YENİ: Alpha Vantage Entegrasyonu
try:
    from alpha_vantage.timeseries import TimeSeries
    HAS_ALPHA_VANTAGE = True if Config.ALPHA_VANTAGE_KEY else False
    logger.info(f"Alpha Vantage: {'aktif' if HAS_ALPHA_VANTAGE else 'API key yok'}")
except ImportError:
    HAS_ALPHA_VANTAGE = False
    logger.warning("alpha_vantage yüklü değil")

# 📍 YENİ: Finnhub Entegrasyonu
try:
    import finnhub
    HAS_FINNHUB = True if Config.FINNHUB_KEY else False
    logger.info(f"Finnhub: {'aktif' if HAS_FINNHUB else 'API key yok'}")
except ImportError:
    HAS_FINNHUB = False
    logger.warning("finnhub-python yüklü değil")

# Model Tanımlamaları
MODEL_70B = Config.GROQ_MODEL

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

# --- PROFESYONEL DARK TEMASI ---
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

    [data-testid="stMetric"], .t-panel-header, .t-card {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stMetric"]:hover, .t-panel-header:hover {
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
    
    .data-source-badge.tv {
        background: rgba(139, 92, 246, 0.2);
        color: #8b5cf6;
        border-color: rgba(139, 92, 246, 0.3);
    }
    
    .data-source-badge.fallback {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border-color: rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- VERİ VE YARDIMCI FONKSİYONLAR ---

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

def calculate_rsi(series: pd.Series, period: int = Config.RSI_PERIOD) -> pd.Series:
    """
    Relative Strength Index (RSI) hesaplar.
    
    Args:
        series: Fiyat serisi
        period: Hesaplama periyodu (varsayılan: 14)
    
    Returns:
        RSI değerleri serisi
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.where(loss != 0, 0.00001)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).clip(0, 100)

def determine_trend(price: float, sma20: Optional[float], sma50: Optional[float]) -> Tuple[str, str]:
    """Trend yönünü belirle - Daha doğru algoritma"""
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

    elif price > sma20 and price < sma50:
        if abs(price - sma20) > abs(price - sma50):
            return "YATAY (Dirençte)", "➡️"
        else:
            return "YÜKSELİŞ (Deneme)", "↗️"

    elif price < sma20 and price > sma50:
        if abs(price - sma20) > abs(price - sma50):
            return "YATAY (Destekte)", "➡️"
        else:
            return "DÜŞÜŞ (Deneme)", "↘️"

    else:
        if abs(price - sma20) < 0.01 * price:
            return "SMA20 SEVİYESİ", "⚖️"
        elif abs(price - sma50) < 0.01 * price:
            return "SMA50 SEVİYESİ", "⚖️"
        else:
            return "YATAY", "➡️"

def get_rsi_comment(rsi_value: float) -> Tuple[str, str]:
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
        logger.info("Browser session oluşturuldu")
    return _browser_session

# ============================================================
# 📍 VERİ KAYNAKLARI (SIRALI DENEME)
# ============================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def fetch_tv_quote(tv_symbol: str) -> Optional[Dict[str, Any]]:
    """TradingView scanner API'den SADECE canlı fiyat/değişim/RSI döner."""
    session = get_browser_session()
    payload = {
        "symbols": {"tickers": [tv_symbol]},
        "columns": ["name", "close", "change", "open", "high", "low", "volume", "RSI"]
    }
    try:
        logger.info(f"TradingView canlı fiyat çekiliyor: {tv_symbol}")
        res = session.post(Config.TV_SCAN_URL, json=payload, timeout=5)
        if res.status_code != 200:
            logger.warning(f"TradingView HTTP {res.status_code} ({tv_symbol})")
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

def fetch_stooq_data(symbol: str) -> Optional[pd.DataFrame]:
    """Stooq'tan GERÇEK günlük OHLC geçmişi çeker."""
    try:
        stooq_code = symbol.replace(".IS", ".TR").replace("^", "").lower()
        stooq_url = f"{Config.STOOQ_BASE_URL}?s={stooq_code}&i=d"

        logger.info(f"Stooq verisi çekiliyor: {symbol}")
        df_stooq = pd.read_csv(stooq_url)

        if not df_stooq.empty and 'Close' in df_stooq.columns and len(df_stooq) > 5:
            df_stooq['Date'] = pd.to_datetime(df_stooq['Date'])
            df_stooq.set_index('Date', inplace=True)
            df_stooq.sort_index(inplace=True)

            for col in ['Open', 'High', 'Low', 'Close']:
                df_stooq[col] = pd.to_numeric(df_stooq[col], errors='coerce')

            df_stooq.dropna(subset=['Close'], inplace=True)

            if len(df_stooq) > 5:
                logger.info(f"Stooq verisi alındı: {symbol}")
                return df_stooq[['Open', 'High', 'Low', 'Close']]

    except Exception as e:
        logger.error(f"Stooq hatası ({symbol}): {e}")

    return None

# 📍 YENİ: Yahoo Finance Veri Kaynağı
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_yahoo_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Yahoo Finance'ten gerçek zamanlı veri çeker"""
    if not HAS_YFINANCE:
        return None
        
    try:
        # BIST hisseleri için .IS -> .IS (Yahoo'da aynı)
        yahoo_symbol = symbol.replace(".IS", ".IS")
        
        # Ticker objesi oluştur
        ticker = yf.Ticker(yahoo_symbol)
        
        # Gerçek zamanlı veri
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            logger.warning(f"Yahoo: {symbol} için veri bulunamadı")
            return None
            
        # Geçmiş veri (son 100 gün)
        hist = ticker.history(period="3mo")
        
        if hist.empty:
            logger.warning(f"Yahoo: {symbol} için geçmiş veri yok")
            # Sadece canlı veri döndür
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
        
        # DataFrame'i hazırla
        df = hist[['Open', 'High', 'Low', 'Close']].copy()
        df.index = pd.to_datetime(df.index)
        
        # Teknik indikatörler
        df['SMA20'] = df['Close'].rolling(Config.SMA_FAST).mean()
        df['SMA50'] = df['Close'].rolling(Config.SMA_SLOW).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        last_price = float(df['Close'].iloc[-1])
        change_pct = float(info.get('regularMarketChangePercent', 0))
        
        return {
            "symbol": symbol,
            "price": last_price,
            "change": change_pct,
            "currency": "TRY",
            "support": float(df['Low'].tail(20).min()),
            "resistance": float(df['High'].tail(20).max()),
            "df": df,
            "data_source": "Yahoo Finance (tam veri)"
        }
        
    except Exception as e:
        logger.error(f"Yahoo Finance hatası ({symbol}): {e}")
        return None

# 📍 YENİ: Alpha Vantage Veri Kaynağı
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def fetch_alpha_vantage_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Alpha Vantage API'den veri çeker"""
    if not HAS_ALPHA_VANTAGE or not Config.ALPHA_VANTAGE_KEY:
        return None
        
    try:
        ts = TimeSeries(key=Config.ALPHA_VANTAGE_KEY, output_format='pandas')
        
        # Hisse senedi verisi
        data, meta_data = ts.get_daily(symbol=symbol.replace('.IS', ''), outputsize='compact')
        
        if data.empty:
            return None
            
        # Veriyi yeniden adlandır
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data.index = pd.to_datetime(data.index)
        
        # Teknik indikatörler
        data['SMA20'] = data['Close'].rolling(Config.SMA_FAST).mean()
        data['SMA50'] = data['Close'].rolling(Config.SMA_SLOW).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        
        last_price = float(data['Close'].iloc[-1])
        
        return {
            "symbol": symbol,
            "price": last_price,
            "change": 0.0,  # Alpha Vantage'dan değişim gelmiyor
            "currency": "TRY",
            "support": float(data['Low'].tail(20).min()),
            "resistance": float(data['High'].tail(20).max()),
            "df": data,
            "data_source": "Alpha Vantage"
        }
        
    except Exception as e:
        logger.error(f"Alpha Vantage hatası: {e}")
        return None

# 📍 YENİ: Finnhub Veri Kaynağı
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def fetch_finnhub_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Finnhub API'den gerçek zamanlı veri çeker"""
    if not HAS_FINNHUB or not Config.FINNHUB_KEY:
        return None
        
    try:
        client = finnhub.Client(api_key=Config.FINNHUB_KEY)
        
        # Gerçek zamanlı fiyat
        quote = client.quote(symbol.replace('.IS', ''))
        
        if not quote or quote.get('c', 0) == 0:
            return None
            
        # Geçmiş veri (son 100 gün)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=100)
        
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol.replace('.IS', '')}&resolution=D&from={int(start_date.timestamp())}&to={int(end_date.timestamp())}&token={Config.FINNHUB_KEY}"
        
        response = cffi_requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('s') != 'ok':
            # Sadece canlı veri döndür
            return {
                "symbol": symbol,
                "price": float(quote.get('c', 0)),
                "change": float(quote.get('dp', 0)),
                "currency": "TRY",
                "support": float(quote.get('l', 0)),
                "resistance": float(quote.get('h', 0)),
                "df": None,
                "data_source": "Finnhub (sadece canlı)"
            }
        
        # DataFrame oluştur
        df = pd.DataFrame({
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c']
        }, index=pd.to_datetime(data['t'], unit='s'))
        
        # Teknik indikatörler
        df['SMA20'] = df['Close'].rolling(Config.SMA_FAST).mean()
        df['SMA50'] = df['Close'].rolling(Config.SMA_SLOW).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        return {
            "symbol": symbol,
            "price": float(quote.get('c', 0)),
            "change": float(quote.get('dp', 0)),
            "currency": "TRY",
            "support": float(df['Low'].tail(20).min()),
            "resistance": float(df['High'].tail(20).max()),
            "df": df,
            "data_source": "Finnhub"
        }
        
    except Exception as e:
        logger.error(f"Finnhub hatası: {e}")
        return None

# 📍 VERİ STANDARDİZASYONU
def standardize_data(data: Dict, symbol: str) -> Optional[Dict]:
    """Farklı kaynaklardan gelen verileri standartlaştırır"""
    try:
        # Eğer data zaten standart formattaysa
        if 'df' in data and isinstance(data['df'], pd.DataFrame):
            # Eksik alanları doldur
            if 'currency' not in data:
                data['currency'] = 'TRY' if '.IS' in symbol else 'USD'
            if 'support' not in data and 'df' in data and not data['df'].empty:
                data['support'] = float(data['df']['Low'].tail(20).min())
            if 'resistance' not in data and 'df' in data and not data['df'].empty:
                data['resistance'] = float(data['df']['High'].tail(20).max())
            return data
        
        # TradingView'den gelen ham veriyi standartlaştır
        if 'price' in data and 'change' in data:
            if 'df' not in data:
                # Tek satırlık veri oluştur
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

# 📍 VERİ YÖNETİCİSİ (FALLBACK MEKANİZMASI)
class MarketDataManager:
    """Veri yönetimi ve fallback mekanizması"""
    
    def __init__(self):
        self.cache = {}
        self.last_success = {}
        self.failure_count = {}
        self.data_sources = []
        
        # Veri kaynaklarını sıralı olarak ekle
        if HAS_YFINANCE:
            self.data_sources.append(('yahoo', fetch_yahoo_data))
        self.data_sources.append(('stooq', fetch_stooq_data))
        self.data_sources.append(('tradingview', lambda x: fetch_tv_quote(tv_symbol_for(x))))
        if HAS_ALPHA_VANTAGE:
            self.data_sources.append(('alpha_vantage', fetch_alpha_vantage_data))
        if HAS_FINNHUB:
            self.data_sources.append(('finnhub', fetch_finnhub_data))
        
        logger.info(f"Veri kaynakları: {[s[0] for s in self.data_sources]}")
        
    def get_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Gelişmiş veri çekme yönetimi"""
        
        # Cache kontrolü
        cache_key = f"{symbol}_{datetime.datetime.now().date()}"
        if not force_refresh and cache_key in self.cache:
            logger.info(f"Cache'den alınıyor: {symbol}")
            return self.cache[cache_key]
        
        # Veri çek
        result = self._fetch_with_fallback(symbol)
        
        if result:
            self.cache[cache_key] = result
            self.last_success[symbol] = datetime.datetime.now()
            self.failure_count[symbol] = 0
            return result
        
        # Fallback: Daha önce başarılı olan veriyi kullan
        if symbol in self.last_success:
            age = (datetime.datetime.now() - self.last_success[symbol]).seconds / 60
            if age < 15:  # 15 dakikadan eski değilse
                logger.warning(f"⚠️ {symbol} için eski veri kullanılıyor ({age:.1f} dakika)")
                return self.cache.get(cache_key)
        
        # Fallback: Sabit örnek veri (ÇOK ACİL DURUM)
        self.failure_count[symbol] = self.failure_count.get(symbol, 0) + 1
        if self.failure_count[symbol] > 3:
            logger.critical(f"🚨 {symbol} için veri alınamıyor!")
            
        return None
    
    def _fetch_with_fallback(self, symbol: str) -> Optional[Dict]:
        """Tüm veri kaynaklarını sırayla dene"""
        clean_sym = sanitize_symbol(symbol)
        logger.info(f"Veri çekiliyor: {clean_sym}")
        
        last_error = None
        for source_name, fetch_func in self.data_sources:
            try:
                logger.info(f"Deneniyor: {source_name}")
                result = fetch_func(clean_sym)
                
                if result and isinstance(result, dict):
                    # Sonuçları standartlaştır
                    standardized = standardize_data(result, clean_sym)
                    if standardized:
                        logger.info(f"✅ {source_name} başarılı!")
                        return standardized
                        
            except Exception as e:
                last_error = e
                logger.warning(f"❌ {source_name} başarısız: {e}")
                continue
        
        logger.error(f"⚠️ {clean_sym} için TÜM veri kaynakları başarısız!")
        if last_error:
            logger.error(f"Son hata: {last_error}")
        
        return None

# 📍 DATA MANAGER INSTANCE
data_manager = MarketDataManager()

# ============================================================
# 📍 ANA VERİ ÇEKME FONKSİYONU (ARTIK DATA MANAGER KULLANIYOR)
# ============================================================
def fetch_real_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Çoklu veri kaynağı ile güvenilir veri çekme"""
    return data_manager.get_data(symbol)

def validate_market_data(data: Dict[str, Any]) -> bool:
    required_fields = ['symbol', 'price', 'change', 'df']

    if not all(field in data for field in required_fields):
        logger.error(f"Eksik alan: {data.keys()}")
        return False

    df = data['df']
    if df is None or df.empty:
        logger.error("Veri boş")
        return False

    if abs(data['change']) > 20:
        logger.warning(f"Anormal değişim: {data['change']:.2f}%")
        return False
    
    # Yeni kontroller
    if data['price'] <= 0:
        logger.error(f"Geçersiz fiyat: {data['price']}")
        return False
        
    # Değişim oranının aşırı yüksek olmadığını kontrol et
    if abs(data['change']) > 30:  # %30'dan fazla değişim anormal
        logger.warning(f"Anormal değişim: {data['change']:.2f}%")
        return False

    return True

# 📍 BIST ANA
def fetch_bist_ana_data():
    try:
        live = fetch_tv_quote("BIST:XBANA")
    except Exception as e:
        logger.error(f"BIST ANA TradingView hatası: {e}")
        live = None

    if not live:
        # Yedek olarak Yahoo Finance'den dene
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

    return {
        "symbol": "BIST ANA",
        "price": live['price'],
        "change": live['change'],
        "currency": "TRY",
        "support": live['low'],
        "resistance": live['high'],
        "df": None
    }

# 📍 DÖVİZ KURLARI
@st.cache_data(ttl=Config.CACHE_TTL_SHORT)
def fetch_fx_rate(pair_from: str, pair_to: str) -> Optional[Dict[str, float]]:
    """Gerçek döviz kuru (ECB referans, Frankfurter API üzerinden)."""
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

# ============================================================
# 📍 ANALYZE WITH AI
# ============================================================
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
            f"📊 KESİN GERÇEK VERİLER (kaynak: {data_source}):\n"
            f"- Sembol: {market_data['symbol']}\n"
            f"- Canlı Son Fiyat: {market_data['price']:.2f} {market_data['currency']}\n"
            f"- Günlük Değişim: %{market_data['change']:+.2f}\n"
            f"- RSI(14): {last_rsi:.1f} ({rsi_comment})\n"
            f"- SMA20: {sma20_str} {currency if sma20 is not None else ''}\n"
            f"- SMA50: {sma50_str} {currency if sma50 is not None else ''}\n"
            f"- Trend Durumu: {trend_icon} {trend_text}\n"
            f"- Destek Seviyesi: {market_data['support']:.2f} {market_data['currency']}\n"
            f"- Direnç Seviyesi: {market_data['resistance']:.2f} {market_data['currency']}"
        )
    else:
        data_str = "⚠️ UYARI: Canlı veri çekilemedi. Lütfen geçerli bir sembol girin."

    system_instruction = (
        "Sen 'BISTeknik' adında profesyonel bir quant borsa analistisin.\n\n"
        "📌 TEMEL KURALLAR:\n"
        "1. Kesinlikle fiyat UYDURMA. Yalnızca sana verilen GERÇEK FİYAT VERİSİNİ kullan.\n"
        "2. Analizini TEKNİK İNDİKATÖRLERE dayandır.\n"
        "3. Cevabını 3-4 paragrafta, net ve öz bir şekilde ver.\n"
        "4. Türkçe yaz, ama İngilizce terimleri doğru kullan.\n"
        "5. Eğer veri yoksa veya sembol geçersizse, bunu açıkça belirt.\n\n"
        "📈 TEKNİK ANALİZ KURALLARI:\n"
        "- RSI 70 üstü = AŞIRI ALIM (satış düşünülebilir)\n"
        "- RSI 30 altı = AŞIRI SATIM (alım düşünülebilir)\n"
        "- Fiyat SMA20 üstünde = KISA VADE YÜKSELİŞ trendi\n"
        "- Fiyat SMA20 altında = KISA VADE DÜŞÜŞ trendi\n"
        "- Fiyat SMA50 üstünde = ORTA VADE YÜKSELİŞ trendi\n"
        "- Fiyat SMA50 altında = ORTA VADE DÜŞÜŞ trendi\n"
        "- DESTEK = Fiyatın düşerken tutunabileceği seviye\n"
        "- DİRENÇ = Fiyatın yükselirken karşılaşacağı seviye\n\n"
        f"Mevcut Canlı Pazar Verisi:\n{data_str}\n\n"
        "Analizini bu verilere dayanarak yap. Son cümlede özet bir görüş belirt."
    )

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        logger.info("AI analizi başlatılıyor...")
        res = client.chat.completions.create(
            model=MODEL_70B,
            messages=messages,
            temperature=0.1,
            max_tokens=400
        )
        response = res.choices[0].message.content
        logger.info("AI analizi tamamlandı")
        return response
    except Exception as err:
        logger.error(f"AI analiz hatası: {err}")
        return f"⚠️ AI Analiz Hatası: {err}"

# ============================================================
# 📍 TOP VOLUME BIST100
# ============================================================
@st.cache_data(ttl=Config.CACHE_TTL_MEDIUM)
def get_top_volume_bist100_symbols():
    """Gerçek hacim sıralaması."""
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
        logger.info("Hacim sıralaması çekiliyor...")
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

            logger.info(f"{len(top_tickers)} hisse hacim sıralaması alındı")
    except Exception as e:
        logger.error(f"Hacim sıralaması hatası: {e}")

    return top_tickers

# ============================================================
# 📍 PARALEL VERİ ÇEKME
# ============================================================
def fetch_multiple_symbols(symbols: list) -> dict:
    """Birden fazla sembolün verisini paralel olarak çeker"""
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

# --- SIDEBAR (SOL MENÜ) ---
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
        st.info("🔑 Groq API Key girin veya `.env` dosyasına ekleyin")
        groq_api_key = st.text_input("Groq API Key:", type="password", key="groq_key_input")

    st.markdown("---")
    
    # 📍 YENİ: Veri Kaynağı Durumu
    st.markdown("<p style='font-size: 0.65rem; font-weight: 600; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>🔌 VERİ KAYNAKLARI</p>", unsafe_allow_html=True)
    
    source_status = []
    if HAS_YFINANCE:
        source_status.append("✅ Yahoo Finance")
    else:
        source_status.append("❌ Yahoo Finance (kurulu değil)")
    source_status.append("✅ Stooq")
    source_status.append("✅ TradingView")
    if HAS_ALPHA_VANTAGE:
        source_status.append("✅ Alpha Vantage")
    else:
        source_status.append("❌ Alpha Vantage (API key yok)")
    if HAS_FINNHUB:
        source_status.append("✅ Finnhub")
    else:
        source_status.append("❌ Finnhub (API key yok)")
    
    for status in source_status:
        st.caption(status)
    
    st.markdown("---")

    st.markdown("<p style='font-size: 0.65rem; font-weight: 600; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>📋 CANLI WATCHLIST</p>", unsafe_allow_html=True)
    watchlist_input = st.text_input("Semboller (virgülle ayırın):", value="THYAO.IS, ASELS.IS, GARAN.IS", label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 GÜNCELLE", use_container_width=True):
            symbols = [sanitize_symbol(s.strip()) for s in watchlist_input.split(",") if s.strip()]

            progress_bar = st.progress(0)
            
            # 📍 YENİ: Paralel veri çekme
            results = fetch_multiple_symbols(symbols)

            for i, sym in enumerate(symbols):
                progress_bar.progress((i + 1) / len(symbols))
                res_data = results.get(sym)
                if res_data:
                    st.metric(
                        label=res_data['symbol'],
                        value=f"{res_data['price']:,.2f} {res_data['currency']}",
                        delta=f"%{res_data['change']:+.2f}"
                    )
                    # Veri kaynağını göster
                    st.caption(f"📡 {res_data.get('data_source', 'Bilinmiyor')}")
                else:
                    st.caption(f"⚠️ {sym} canlı veri alınamadı.")

            progress_bar.empty()

    with col2:
        if st.button("📊 ANALİZ", use_container_width=True):
            st.info("AI analizi için sağ panelde soru sorun")

# API key kontrolü
if not groq_api_key:
    st.warning("""
    ⚠️ **Groq API Key bulunamadı!**

    Lütfen:
    1. 📝 Sol menüden manuel girin
    2. 📄 `.env` dosyası oluşturun: `GROQ_API_KEY=your_key_here`
    3. 🔄 Sayfayı yenileyin
    """)
    st.stop()

try:
    client = Groq(api_key=groq_api_key)
    logger.info("Groq client başarıyla başlatıldı")
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
    except Exception as e:
        logger.error(f"BIST100 hatası: {e}")
        
        # Yedek: Yahoo Finance
        if HAS_YFINANCE:
            try:
                yahoo_bist = fetch_yahoo_data("XU100.IS")
                if yahoo_bist:
                    bist100_live = {
                        "price": yahoo_bist['price'],
                        "change": yahoo_bist['change']
                    }
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
            cols[idx].metric(label=name, value="— veri yok", delta=None)
        else:
            delta_color = "normal" if chg >= 0 else "inverse"
            cols[idx].metric(
                label=name,
                value=f"{val:,.2f}",
                delta=f"%{chg:+.2f}",
                delta_color=delta_color
            )

# --- CANLI AKAN PİYASA ŞERİDİ ---
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
    st.caption("⚠️ Hacim sıralaması şu anda alınamıyor (TradingView bağlantı sorunu).")

st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.6, 1.0], gap="small")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "⚡ **BISTeknik Terminal Çevrimiçi.** Bir hisse seçin veya yazın."}
    ]

# SOL PANEL
with col_left:
    st.markdown("""
    <div class='t-panel-header'>
        <span>📊 TECHNICAL ANALYTICS ENGINE</span>
        <span>● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    selected_bist_option = st.selectbox(
        "🏛️ BIST 100 En Çok İşlem Gören Hisseler:",
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

        # 📍 YENİ: Veri kaynağı badge'li başlık
        data_source = market_data.get('data_source', 'Bilinmiyor')
        source_class = "yahoo" if "Yahoo" in data_source else "stooq" if "Stooq" in data_source else "tv" if "TradingView" in data_source else "fallback"
        
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
            st.info("ℹ️ Bu sembol için gerçek geçmiş mum verisi bulunamadı — sadece şu anki canlı fiyat gösteriliyor. SMA/RSI için yetersiz veri.")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f"{market_data['symbol']} — CANDLESTICK & SMA", f"RSI (14) — {last_rsi:.1f}"),
            row_heights=[0.72, 0.28]
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Fiyat (Mum)",
            increasing_line_color='#22c55e', increasing_fillcolor='#22c55e',
            decreasing_line_color='#ef4444', decreasing_fillcolor='#ef4444'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name='Trend Çizgisi',
            line=dict(color=trend_color, width=1.5)
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
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="#ef4444", opacity=0.05, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="#22c55e", opacity=0.05, row=2, col=1)

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
            st.metric(
                "RSI (14)",
                f"{last_rsi:.1f}",
                delta=rsi_label,
                delta_color=rsi_color
            )
        with col_metric2:
            st.metric(
                "SMA 20",
                f"{sma20_val:.2f}" if sma20_val else "—",
                delta="Fiyat Üstü" if sma20_val and market_data['price'] > sma20_val else "Fiyat Altı" if sma20_val else "—"
            )
        with col_metric3:
            st.metric(
                "SMA 50",
                f"{sma50_val:.2f}" if sma50_val else "—",
                delta="Fiyat Üstü" if sma50_val and market_data['price'] > sma50_val else "Fiyat Altı" if sma50_val else "—"
            )
        with col_metric4:
            st.metric(
                "Trend",
                f"{trend_icon} {trend_text}",
                delta=f"%{market_data['change']:+.2f}"
            )

    else:
        st.error(f"❌ **{active_symbol}** için borsadan canlı veri alınamadı.")
        st.info("💡 Öneriler:\n- Sembol kodunu kontrol edin (örn: THYAO.IS)\n- Borsa açık mı kontrol edin\n- Geçerli bir hisse kodu girin (örn: GARAN.IS, THYAO.IS)")
        
        # 📍 YENİ: Veri kaynağı önerileri
        st.warning("""
        🔍 **Veri kaynağı sorunu mu yaşıyorsunuz?**
        
        1. 📦 `pip install yfinance` ile Yahoo Finance'i kurun
        2. 🔑 Alpha Vantage veya Finnhub API key alın
        3. 🌐 İnternet bağlantınızı kontrol edin
        """)

# SAĞ PANEL
with col_right:
    st.markdown("""
    <div class='t-panel-header'>
        <span>🤖 AI QUANT ANALYST</span>
        <span>70B</span>
    </div>
    """, unsafe_allow_html=True)

    chat_container = st.container(height=380)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Soru veya sembol yazın...", key="chat_input")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        progress_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("🧠 AI analiz yapıyor..."):
            progress_bar.progress(30)
            progress_text.text("📊 Piyasa verileri işleniyor...")

            default_symbol = active_symbol if 'active_symbol' in dir() else "THYAO.IS"
            query_symbol = extract_symbol_fast(prompt, default_sym=default_symbol)

            # Veri çek
            target_market_data = fetch_real_market_data(query_symbol)
            
            # Fallback: Mevcut market_data'yı kullan
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

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.rerun()
