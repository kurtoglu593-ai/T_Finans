import os
import ast
import shutil
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from groq import Groq

# Plotly Tema Ayarı
pio.templates.default = "plotly_dark"

# Sayfa Yapılandırması
st.set_page_config(
    page_title="T - Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FINANS TERMINALI CUSTOM CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0B0E14;
        color: #E1E6ED;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #12161F !important;
        border-right: 1px solid #1E2330 !important;
    }
    header {visibility: hidden;}
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 98%;
    }
    [data-testid="stMetric"] {
        background: #151922;
        border: 1px solid #222836;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #8B95A5 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    .stChatMessage {
        background-color: #151922 !important;
        border: 1px solid #222836 !important;
        border-radius: 8px !important;
        color: #E1E6ED !important;
    }
    .stTextInput input {
        background-color: #181D28 !important;
        color: #FFFFFF !important;
        border: 1px solid #2A3142 !important;
        border-radius: 6px !important;
    }
    .stButton button {
        background-color: #1E2433 !important;
        color: #29B6F6 !important;
        border: 1px solid #2A3142 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #29B6F6 !important;
        color: #0B0E14 !important;
        border-color: #29B6F6 !important;
    }
    hr {
        border-color: #1E2330 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SİGORTA VE KOD DÜZENLEME MOTORU ---
APP_FILE = "app.py"
BACKUP_FILE = "app_backup.py"

def backup_code():
    if os.path.exists(APP_FILE):
        shutil.copy(APP_FILE, BACKUP_FILE)

def restore_backup():
    if os.path.exists(BACKUP_FILE):
        shutil.copy(BACKUP_FILE, APP_FILE)
        return True
    return False

def validate_python_code(code_string: str) -> bool:
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        return False

def evolve_self(user_instruction: str) -> str:
    try:
        with open(APP_FILE, "r", encoding="utf-8") as f:
            current_code = f.read()

        prompt = (
            "Sen expert bir Python ve Streamlit geliştiricisisin.\n"
            "Aşağıda app.py kodları bulunmaktadır:\n"
            "```python\n" + current_code + "\n```\n"
            "Kullanıcı İsteği: \"" + user_instruction + "\"\n"
            "GÖREVİN: Koda istenen yeni özelliği hatasız ekle ve SADECE çalışan tam Python kodunu döndür."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        new_code = response.choices[0].message.content.strip()

        if new_code.startswith("```python"):
            new_code = new_code.replace("```python", "", 1)
        if new_code.endswith("```"):
            new_code = new_code[:-3]
        new_code = new_code.strip()

        if not validate_python_code(new_code):
            return "❌ Kodda sentaks hatası oluştu, işlem iptal edildi."

        backup_code()
        with open(APP_FILE, "w", encoding="utf-8") as f:
            f.write(new_code)

        return "✅ Kodum başarıyla güncellendi! Sayfa yenileniyor..."
    except Exception as e:
        return f"❌ Hata: {e}"

# --- FINANSAL HESAPLAMALAR VE VERİ ÇEKME ---
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_data(symbol: str):
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

def analyze_with_ai(user_prompt: str, market_data: dict, history: list) -> str:
    data_str = "Canlı piyasa verisi çekilemedi."
    if market_data:
        last_rsi = market_data['df']['RSI'].iloc[-1] if 'RSI' in market_data['df'] else 0
        data_str = f"Varlık: {market_data['symbol']} | Son Fiyat: {market_data['price']:.2f} {market_data['currency']} | Değişim: %{market_data['change']:+.2f} | RSI(14): {last_rsi:.1f}"

    system_instruction = (
        "Sen 'T' adında üst düzey bir borsa ve finans analistisin. "
        f"Canlı Piyasa Verisi: {data_str}. "
        "Yorumlarını teknik göstergelere (RSI, SMA20, SMA50) dayanarak, kısa, net ve borsa terminali üslubuyla ver."
    )

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        res = client.chat.completions.create(model="llama-3.3-
