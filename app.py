import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from groq import Groq
import os
import ast
import shutil

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

# --- BAŞLIK VE ANA EKRAN (Her zaman görünür) ---
st.title("📈 T — Otonom Finans & Gelişim Asistanı")
st.caption("Hem finansal analiz yapan hem de kendi kodunu geliştirebilen yapay zeka altyapısı")

# --- YAN MENÜ VE API KONTROLÜ ---
with st.sidebar:
    st.header("⚙️ T Kontrol Paneli")
    groq_api_key = st.text_input("Groq API Key:", type="password", help="console.groq.com adresinden aldığınız anahtar")
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.divider()
    st.subheader("🛡️ Güvenlik Sigortası")
    if st.button("⏪ Son Kod Değişikliğini Geri Al"):
        if restore_backup():
            st.success("Eski çalışan koda dönüldü! Yenileniyor...")
            st.rerun()
        else:
            st.warning("Yedek bulunamadı.")

# API Key Kontrolü Uyarısı
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

# --- FİNANSAL ANALİZ ---
def fetch_data(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6m")
        if df.empty:
            return None
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
        data_str = f"Varlık: {market_data['symbol']} | Fiyat: {market_data['price']:.2f} {market_data['currency']} | Değişim: %{market_data['change']:+.2f}"

    system_instruction = f"Sen 'T' adında finans uzmanısın. Canlı Veri: {data_str}. Türkçe yanıt ver."

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

# --- SOHBET VE EKRAN ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben **T**.\n- Finansal sorular sorabilirsiniz (`THY yorumu`, `Bitcoin analizi` vb.)\n- Koda özellik ekletebilirsiniz (`Kendine dolar/euro canlı kur bandı ekle`)"}
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
            with st.spinner("T verileri inceliyor..."):
                symbol = detect_symbol_with_ai(prompt, st.session_state.messages)
                market_data = fetch_data(symbol) if symbol else None
                ai_response = analyze_with_ai(prompt, market_data, st.session_state.messages)
                st.markdown(ai_response)

                if market_data and market_data.get("df") is not None:
                    df = market_data["df"].tail(60)
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                    fig.update_layout(title=f"{market_data['symbol']} Fiyat Grafiği", height=350, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                st.session_state.messages.append({"role": "assistant", "content": ai_response})