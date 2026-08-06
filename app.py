import requests

def fetch_data(symbol):
    try:
        clean_symbol = sanitize_symbol(symbol)
        
        # Yahoo Finance bot engellemesini aşmak için User-Agent oturumu oluşturma
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        ticker = yf.Ticker(clean_symbol, session=session)
        
        # 1. Yöntem: history ile veri çekme
        df = ticker.history(period="6m", auto_adjust=True)
        
        # 2. Yöntem (Yedek): Eğer history boş geldiyse yf.download dene
        if df.empty:
            df = yf.download(clean_symbol, period="6m", progress=False, session=session)
            # multi-index sütun temizliği (yf.download bazen Ticker seviyesinde sütun üretir)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        # Hâlâ veri yoksa ve .IS eksikse ekleyip son kez dene
        if df.empty and not clean_symbol.endswith(".IS"):
            clean_symbol = f"{clean_symbol}.IS"
            df = yf.download(clean_symbol, period="6m", progress=False, session=session)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.error(f"⚠️ `{clean_symbol}` için Yahoo Finance veri döndürmedi (Erişim engeli veya geçersiz sembol).")
            return None
        
        # Indikatör Hesaplamaları
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)

        # Fiyat Bilgilerini Sağlama
        last_p = float(df['Close'].iloc[-1])
        prev_p = float(df['Close'].iloc[-2]) if len(df) > 1 else last_p
        
        pct_chg = ((last_p - prev_p) / prev_p) * 100.0 if prev_p else 0.0
        curr = 'TRY' if clean_symbol.endswith('.IS') else 'USD'

        return {
            "symbol": clean_symbol,
            "price": last_p,
            "change": pct_chg,
            "currency": curr,
            "df": df
        }
    except Exception as e:
        st.error(f"Veri işleme hatası: {e}")
        return None
