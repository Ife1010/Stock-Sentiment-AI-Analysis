import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

def get_robust_stock_data(ticker):
    print(f"--- Accessing Market Data for {ticker} ---")
    
    # FIX 1: Use Ticker object with auto_adjust to ensure we get data
    stock = yf.Ticker(ticker)
    
    # We will fetch 'max' to ensure we get plenty of rows for the 2026 model
    df = stock.history(period="5y", auto_adjust=True)
    
    if df.empty:
        print(f"Error: No data found for {ticker}. Check your internet or ticker symbol.")
        return None

    # FIX 2: Standardize column names (Sometimes yfinance returns multi-index)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # --- Feature Engineering ---
    # Add RSI (Relative Strength Index)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Add SMA (Simple Moving Average)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    
    # Add MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # Drop rows that don't have enough data for the indicators (the first 20 days)
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    symbol = "AAPL" 
    processed_df = get_robust_stock_data(symbol)
    
    if processed_df is not None:
        processed_df.to_csv(f"{symbol}_financial_data.csv")
        print(f"Success! {len(processed_df)} rows saved to {symbol}_financial_data.csv")
        
        # Visualize to confirm
        plt.figure(figsize=(12,6))
        plt.plot(processed_df['Close'], label='Close Price')
        plt.plot(processed_df['SMA_20'], label='20-Day Trend', alpha=0.7)
        plt.title(f"{symbol} Price & Trend (Processed for 2026 Model)")
        plt.legend()
        plt.show()
        