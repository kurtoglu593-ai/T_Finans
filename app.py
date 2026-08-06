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

        prompt = (
            "Sen expert bir Python ve Streamlit geliştiricisisin.\n"
            f"Aşağıda app.py kodları bulunmaktadır:\n```python\n{current_code}\n```\n"
            f'Kullanıcı İsteği: "{user_instruction}"\n'
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
        
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)

        info = ticker.fast_info
        return {
            "symbol": symbol,
            "price": info.last_price,
            "change": ((info.last_price - info.previous_close) / info.previous_close) * 100,
            "currency": getattr(info, 'currency', 'TL'),
            "df": df
        }
    except Exception:
        return None

def analyze_with_ai(user_prompt: str, market_data: dict, history: list) -> str:
    data_str = "Canlı piyasa verisi yok."
    if market_data:
        last_rsi = market_data['df']['RSI'].iloc[-1] if 'RSI' in market_data['df'] else 0
        data_str = f"Varlık: {market_data['symbol']} | Fiyat: {market_data['price']:.2f} {market_data['currency']} | Değişim: %{market_data['change']:+.2f} | Son RSI(14): {last_rsi:.1f}"

    system_instruction = (
        "Sen 'T' adında uzman bir finans analistisin. "
        f"Canlı Veri: {data_str}. "
        "Teknik göstergeleri (RSI, Hareketli Ortalamalar) değerlendirerek Türkçe profesyonel yanıt ver."
    )

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.3)
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ Hata: {e}"

def detect_symbol_with_ai(user_input: str, history: list) -> str:
    prompt = f"Geçmiş: {history[-2:]}\nSon Mesaj: '{user_input}'\nBorsa/Kripto kodu nedir? (Örn: THYAO.IS, BTC-USD). Yoksa 'YOK' yaz."
    try:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0.0)
        code = res.choices[0].message.content.strip().upper()
        return None if "YOK" in code or len(code) > 12 else code
    except Exception:
        return None

# --- SOHBET VE EKRAN YÖNETİMİ ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben **T**.\n- Finansal sorular sorabilirsiniz (`THY analizi`, `Bitcoin RSI durumu` vb.)\n- Koduma özellik ekletebilirsiniz."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("T'ye bir soru sorun veya kod güncellemesi isteyin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if any(w in prompt.lower() for w in ["koduna ekle", "kendine ekle", "özellik ekle", "sayfaya ekle", "butonu ekle", "kodunu değiştir"]):
            with st.spinner("🛠️ T kendi kodunu düzenliyor..."):
                status_msg = evolve_self(prompt)
                st.markdown(status_msg)
                st.session_state.messages.append({"role": "assistant", "content": status_msg})
                if "başarıyla güncellendi" in status_msg:
                    st.rerun()
        else:
            with st.spinner("T teknik verileri inceliyor..."):
                symbol = detect_symbol_with_ai(prompt, st.session_state.messages)
                market_data = fetch_data(symbol) if symbol else None
                ai_response = analyze_with_ai(prompt, market_data, st.session_state.messages)
                st.markdown(ai_response)

                if market_data and market_data.get("df") is not None:
                    df = market_data["df"].tail(90)
                    
                    fig = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08, 
                        subplot_titles=(f"{market_data['symbol']} Fiyat & SMA", "RSI (14)"),
                        row_heights=[0.7, 0.3]
                    )

                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)

                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=1.5)), row=2, col=1)

                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    fig.update_layout(height=500, xaxis_rangeslider_visible=False, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                st.session_state.messages.append({"role": "assistant", "content": ai_response})
