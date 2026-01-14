import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Dashboard Config
st.set_page_config(page_title="Real-Time AI Stock Predictor", layout="wide")
st.title("📈 Real-Time AI Financial Sentiment & Price Predictor")

# 2. PRO DATA ENGINE: Fetch live data on the fly
@st.cache_data # This keeps the app fast by saving data for a few minutes
def get_live_data():
    ticker = "AAPL"
    # Fetch 2 years of data to ensure we have enough for RSI and 60-day window
    data = yf.download(ticker, period="2y")
    
    # Calculate RSI (Technical Indicator)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # For the live demo, we will use a neutral sentiment if news isn't available
    # In a full production app, you'd trigger your sentiment_engine here
    data['Sentiment_Score'] = 0.05 
    
    return data.dropna()

# 3. Load Data and Model
df = get_live_data()
model = load_model("stock_model.h5", compile=False) # compile=False avoids the warning you saw

# 4. Display Metrics
current_price = float(df['Close'].iloc[-1])
st.subheader(f"Current Market Data (Live: Apple Inc.)")
st.dataframe(df[['Close', 'RSI', 'Sentiment_Score']].tail(10))

# 5. Make Live Prediction
# Prepare the last 60 days of real-time data
last_60_days = df[['Close', 'RSI', 'Sentiment_Score']].tail(60).values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(last_60_days)
input_data = np.reshape(scaled_data, (1, 60, 3))

prediction_scaled = model.predict(input_data)
# Reverse the scaling logic (Simple version for demo)
predicted_change = (prediction_scaled[0][0] - 0.5) * 5 
predicted_price = current_price + predicted_change

# 6. Show Results
col1, col2 = st.columns(2)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("AI Prediction (Next Close)", f"${predicted_price:.2f}", delta=f"{predicted_price-current_price:.2f}")

st.subheader("Recent Price Trend")
st.line_chart(df['Close'].tail(30))