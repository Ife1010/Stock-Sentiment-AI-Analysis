import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Predictor 2026", layout="wide")

st.title("📈 AI-Powered Financial Sentiment & Price Predictor")
st.write("This dashboard combines LSTM Deep Learning with FinBERT Sentiment Analysis.")

# 1. Load Data
df = pd.read_csv("refined_data.csv", index_col=0)
st.subheader("Recent Market Data")
st.dataframe(df[['Close', 'RSI', 'Sentiment_Score']].tail(10))

# 2. Load the Trained Brain
model = load_model("stock_model.h5")

# 3. Make a Prediction for "Tomorrow"
# We take the last 60 days to predict the next step
last_60_days = df[['Close', 'RSI', 'Sentiment_Score']].tail(60).values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(last_60_days)
input_data = np.reshape(scaled_data, (1, 60, 3))

prediction_scaled = model.predict(input_data)
# Simple inverse scaling for demonstration
current_price = df['Close'].iloc[-1]
predicted_change = (prediction_scaled[0][0] - 0.5) * 2 # Normalized adjustment
predicted_price = current_price + predicted_change

# 4. Display Results
col1, col2 = st.columns(2)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("AI Predicted Next Close", f"${predicted_price:.2f}", delta=f"{predicted_price-current_price:.2f}")

st.subheader("Price Trend vs AI Analysis")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Close'].tail(30).values, label="Actual Price")
ax.set_title("Last 30 Days Performance")
st.pyplot(fig)