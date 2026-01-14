import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF
import datetime
# 1. Page Configuration
st.set_page_config(page_title="AI Financial Terminal Pro", layout="wide")

# 2. Sidebar - Navigation & Settings
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

# 3. Data Engine
@st.cache_data(ttl=3600)
def load_market_data(symbol):
    try:
        data = yf.download(symbol, period="2y")
        if data.empty: return None
        # Technical Indicator: RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # FIX: Add Sentiment column to match model's expected 3rd feature
        data['Sentiment'] = 0.05 
        return data.dropna()
    except Exception:
        return None

# 4. PDF Generation Logic
def generate_pdf_report(symbol, current, predicted):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, txt=f"AI Financial Intelligence Report: {symbol}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=True)
    pdf.cell(200, 10, txt=f"Current Market Price: ${current:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"AI Predicted Closing Target: ${predicted:.2f}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# 5. Main Dashboard Logic
st.title(f"📊 {selected_display} Intelligence Dashboard")
df = load_market_data(ticker)

if df is not None:
    try:
        # Load your trained model
        model = load_model("stock_model.h5", compile=False)
        current_price = float(df['Close'].iloc[-1])
        
        # --- FIX: Reshape to (1, 60, 3) to match LSTM model shape ---
        last_60 = df[['Close', 'RSI', 'Sentiment']].tail(60).values
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(last_60)
        prediction_scaled = model.predict(np.reshape(scaled_input, (1, 60, 3)))
        
        # Scaling adjustment for daily close target
        predicted_close = current_price + (prediction_scaled[0][0] - 0.5) * (current_price * 0.05)

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${current_price:.2f}")
        m2.metric("AI Predicted Close", f"${predicted_close:.2f}", delta=f"{predicted_close-current_price:.2f}")
        m3.metric("RSI (Momentum)", f"{df['RSI'].iloc[-1]:.2f}")

        # Charts
        st.subheader("Market Trend Analysis")
        st.line_chart(df['Close'].tail(50))

        # --- NEWS FEED (With Safety Fix for KeyError) ---
        st.divider()
        st.subheader(f"📰 {ticker} Market Intelligence")
        news_items = yf.Ticker(ticker).news[:5]
        
        if news_items:
            for item in news_items:
                # Safely get title and link using .get()
                title = item.get('title', 'Headline Unavailable')
                link = item.get('link', '#')
                with st.expander(title):
                    st.write(f"Source: {item.get('publisher', 'Financial Source')}")
                    st.write(f"[Read Article]({link})")

        # --- PDF DOWNLOAD ---
        pdf_bytes = generate_pdf_report(ticker, current_price, predicted_close)
        st.sidebar.divider()
        st.sidebar.download_button(
            label="📥 Download Daily Report",
            data=pdf_bytes,
            file_name=f"{ticker}_AI_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Initialization Error: {e}. Check if stock_model.h5 is in your GitHub repo.")
else:
    st.error("Market data currently unavailable.")