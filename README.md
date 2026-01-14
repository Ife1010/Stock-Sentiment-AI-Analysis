# AI-Powered Financial Sentiment & Price Predictor 📈

## 🚀 Overview
This project is an end-to-end Machine Learning pipeline that predicts stock price movements for **Apple (AAPL)**. It uniquely combines traditional technical indicators with modern Natural Language Processing (NLP).

## 🧠 Technical Features
- **Sentiment Analysis**: Utilizes the **FinBERT** transformer model to analyze financial headlines.
- **Predictive Modeling**: Employs a **Long Short-Term Memory (LSTM)** neural network built with TensorFlow.
- **Data Pipeline**: Automated data ingestion and RSI calculation via `yfinance`.
- **Interactive Dashboard**: A custom UI built with **Streamlit**.

## 🛠️ How to Run
1. Install dependencies: `pip install tensorflow pandas streamlit yfinance scikit-learn`
2. Run the data engine: `python data_engine.py`
3. Run the merger: `python data_merger.py`
4. Train the AI: `python model_trainer.py`
5. Launch the UI: `streamlit run app.py`
