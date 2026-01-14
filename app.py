import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF
import datetime
from streamlit_autorefresh import st_autorefresh # New Import

# 1. Page Configuration
st.set_page_config(page_title="AI Financial Terminal Pro", layout="wide")

# 2. AUTO-REFRESH: Every 60 seconds (60,000 milliseconds)
st_autorefresh(interval=60000, key="datarefresh")

# 3. Sidebar - Settings
st.sidebar.title("🛠️ Terminal Settings")
ticker_dict = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Bitcoin (BTC-USD)": "BTC-USD"
}
selected_display = st.sidebar.selectbox("Choose Asset", list(ticker_dict.keys()))
ticker = ticker_dict[selected_display]

# 4. Data Engine (Lowered TTL to 1 minute to match refresh)
@st.cache_data(ttl=60)
def load_market_data(symbol):
    try:
        # Fetching 1-minute interval data for a "Live" feel
        data = yf.download(symbol, period="1d", interval="1m")
        if data.empty: return None
        
        # Technical Indicator: RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Dummy Sentiment for model shape (1, 60, 3)
        data['Sentiment'] = 0.05 
        return data.dropna()
    except Exception:
        return None

# 5. Dashboard Logic
st.title(f"📊 {selected_display} Live Intelligence")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

df = load_market_data(ticker)

if df is not None:
    try:
        model = load_model("stock_model.h5", compile=False)
        current_price = float(df['Close'].iloc[-1])
        
        # LSTM Prediction Logic
        last_60 = df[['Close', 'RSI', 'Sentiment']].tail(60).values
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(last_60)
        prediction_scaled = model.predict(np.reshape(scaled_input, (1, 60, 3)))
        predicted_close = current_price + (prediction_scaled[0][0] - 0.5) * (current_price * 0.05)

        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Live Price", f"${current_price:.2f}")
        m2.metric("AI Predicted Close", f"${predicted_close:.2f}", delta=f"{predicted_close-current_price:.2f}")
        m3.metric("RSI (Momentum)", f"{df['RSI'].iloc[-1]:.2f}")

        # Real-time Chart
        st.subheader("Intraday Performance (1m Interval)")
        st.line_chart(df['Close'])

        # --- NEWS FEED (Improved key searching) ---
        st.divider()
        st.subheader(f"📰 {ticker} Live News")
        news_items = yf.Ticker(ticker).news[:5]
        
        for item in news_items:
            # Check multiple possible keys used by Yahoo Finance
            title = item.get('title') or item.get('text') or "Breaking Market Update"
            link = item.get('link', '#')
            with st.expander(title):
                st.write(f"Source: {item.get('publisher', 'Financial Source')}")
                st.write(f"[Read Article]({link})")

    except Exception as e:
        st.error(f"Inference Error: {e}")
else:
    st.error("Waiting for market data... (Market might be closed or API is throttled)")