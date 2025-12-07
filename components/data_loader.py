# components/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import config

def fetch_and_process_data(ticker):
    print(f"📥 Fetching data for {ticker}...")
    df = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
    
    # Fix MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    # --- Feature Engineering for Volatility Regime ---
    # Log Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Realized Volatility (21-day rolling standard deviation)
    # We multiply by sqrt(252) to annualize it. This is what the HMM will read.
    df['Volatility'] = df['Log_Ret'].rolling(window=21).std() * np.sqrt(252)
    
    # Simple Moving Average (Trend filter)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df.dropna(inplace=True)
    print(f"✅ Data ready: {df.shape}")
    return df