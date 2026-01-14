import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def train_model():
    print("Step 1: Checking for data file...")
    if not os.path.exists("refined_data.csv"):
        print("ERROR: refined_data.csv not found! Run data_merger.py first.")
        return

    # 1. Load the refined data
    df = pd.read_csv("refined_data.csv", index_col=0)
    print(f"Step 2: Data loaded. Found {len(df)} rows.")

    # Features: Close Price, RSI, Sentiment
    features = df[['Close', 'RSI', 'Sentiment_Score']].values
    target = df['Target_Price'].values.reshape(-1, 1)

    # 2. Scale the data
    scaler_feat = MinMaxScaler()
    scaler_target = MinMaxScaler()
    scaled_features = scaler_feat.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target)

    # 3. Create Sequences
    X, y = [], []
    lookback = 60
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_target[i])
    
    X, y = np.array(X), np.array(y)
    print(f"Step 3: Sequences created. Input shape: {X.shape}")

    # 4. Build the LSTM Model
    print("Step 4: Building the Neural Network...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("--- STARTING AI TRAINING ---")
    history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    # 5. Save
    model.save("stock_model.h5")
    print("Step 5: Model saved as stock_model.h5")
    
    plt.plot(history.history['loss'])
    plt.title('Model Learning Progress (Loss)')
    plt.show()

if __name__ == "__main__":
    train_model()