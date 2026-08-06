def sanitize_symbol(symbol: str) -> str:
    """Türk hisse senetleri için otomatik .IS tamamlama yapar."""
    symbol = symbol.strip().upper()
    # Kripto (BTC-USD), Parite (USDTRY=X) veya zaten .IS olanlara dokunma
    if not symbol.endswith(".IS") and "-" not in symbol and "=" not in symbol:
        # 3 ile 5 harf arası düz kelimeleri (Örn: THYAO, ASELS) BIST kabul et
        if symbol.isalpha() and 3 <= len(symbol) <= 5:
            return f"{symbol}.IS"
    return symbol

def fetch_data(symbol):
    try:
        # 1. Sembolü düzelt (THYAO -> THYAO.IS)
        clean_symbol = sanitize_symbol(symbol)
        ticker = yf.Ticker(clean_symbol)
        df = ticker.history(period="6m")
        
        # 2. Eğer ilk deneme boş döndüyse ve .IS yoksa bir de .IS ile dene
        if df.empty and not clean_symbol.endswith(".IS"):
            clean_symbol = f"{clean_symbol}.IS"
            ticker = yf.Ticker(clean_symbol)
            df = ticker.history(period="6m")
            
        if df.empty:
            return None
        
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)

        info = ticker.fast_info
        
        # Fast_info bazen gecikmeli olabilir, dataframe'den son kapanışı yedek alalım
        last_p = float(getattr(info, 'last_price', df['Close'].iloc[-1]))
        prev_p = float(getattr(info, 'previous_close', df['Close'].iloc[-2]))
        
        pct_chg = ((last_p - prev_p) / prev_p) * 100.0 if prev_p else 0.0
        curr = getattr(info, 'currency', 'TRY' if clean_symbol.endswith('.IS') else 'USD')

        return {
            "symbol": clean_symbol,
            "price": last_p,
            "change": pct_chg,
            "currency": curr,
            "df": df
        }
    except Exception as e:
        st.error(f"Veri çekme hatası ({symbol}): {e}")
        return None
