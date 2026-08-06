import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
from groq import Groq
import os
import ast
import shutil

# Plotly Varsayılan Tema (Koyu Mod)
pio.templates.default = "plotly_dark"

# Sayfa Ayarları
st.set_page_config(page_title="T - Otonom Finans Asistanı", page_icon="📈", layout="wide")

# --- KENDİ KODUNU DÜZENLEME & SİGORTA MOTORU ---
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

# --- BAŞLIK ---
st.title("📈 T — Otonom Finans & Gelişim Asistanı")
st.caption("Hem finansal analiz yapan hem de kendi kodunu geliştirebilen yapay zeka altyapısı")

# --- CANLI PİYASA ÖZET BANDI ---
@st.cache_data(ttl=60)
def get_quick_market_data():
    tickers = {
        "BIST100": "^XU100",
        "Dolar/TL": "USDTRY=X",
        "Euro/TL": "EURTRY=X",
        "Altın ($)": "GC=F"
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

market_summary = get_quick_market_data()
if market_summary:
    cols = st.columns(len(market_summary))
    for idx, (name, (val, chg)) in enumerate(market_summary.items()):
        cols[idx].metric(label=name, value=f"{val:,.2f}", delta=f"%{chg:+.2f}")

st.divider()

# --- YAN MENÜ VE API KONTROLÜ ---
with st.sidebar:
    st.header("⚙️ T Kontrol Paneli")
    groq_api_key = st.text_input("Groq API Key:", type="password", help="console.groq.com adresinden aldığınız anahtar")
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.divider()
    st.subheader("⭐ Favori Takip Listesi")
    watchlist_input = st.text_input("Semboller (virgülle ayırın):", value="THYAO.IS, ASELS.IS, BTC-USD")
    if st.button("🔄 Portföyü Güncelle"):
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
                st.caption(f"⚠️ {sym} verisi alınamadı.")

    st.divider()
    st.subheader("🛡️ Güvenlik Sigortası")
    if st.button("⏪ Son Kod Değişikliğini Geri Al"):
        if restore_backup():
            st.success("Eski çalışan koda dönüldü! Yenileniyor...")
            st.rerun()
        else:
            st.warning("Yedek bulunamadı.")

if not groq_api_key:
    st.info("👈 **Başlamak için:** Sol taraftaki menüden **Groq API Key** anahtarınızı girin.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- KOD GÜNCELLEME İŞLEVİ (SELF-CODING) ---
def evolve_self(user_instruction: str) -> str:
    try:
        with open(APP_FILE, "r", encoding="utf-8") as f:
            current_code = f.read()

        prompt = f"""
        Sen expert bir Python ve Streamlit geliştiricisisin.
        Aşağıda `app.py` kodları bulunmaktadır:
        ```python
        {current_code}
        ```
        Kullanıcı İsteği: "{user_instruction}"
        GÖREVİN: Koda istenen yeni özelliği hatasız ekle ve SADECE çalışan tam Python kodunu döndür.
        """

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

# --- TEKNİK GÖSTERGELER VE FİNANSAL ANALİZ ---
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
        
        # SMA ve RSI
