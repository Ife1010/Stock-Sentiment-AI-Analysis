import pandas as pd
import numpy as np

def merge_data():
    print("--- Merging Price Data with Sentiment Scores ---")
    
    # 1. Load the data from Phase 1
    try:
        df = pd.read_csv("AAPL_financial_data.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: AAPL_financial_data.csv not found. Run Phase 1 first!")
        return

    # 2. Simulate Sentiment (since we don't have 5 years of daily news API access)
    # 1.0 = Very Positive, 0.0 = Neutral, -1.0 = Very Negative
    np.random.seed(42) # Keeps results consistent
    df['Sentiment_Score'] = np.random.uniform(-1, 1, size=len(df))
    
    # 3. Create a 'Target' column (What the AI will try to predict)
    # We want to predict the Close price of the NEXT day
    df['Target_Price'] = df['Close'].shift(-1)
    
    # 4. Drop the last row because it won't have a 'Next Day' price
    df.dropna(inplace=True)
    
    # 5. Save the final training data
    df.to_csv("refined_data.csv")
    print(f"Success! Refined dataset with {len(df)} rows saved to refined_data.csv")
    print("\nPreview of the data the AI will see:")
    print(df[['Close', 'RSI', 'Sentiment_Score', 'Target_Price']].head())

if __name__ == "__main__":
    merge_data()