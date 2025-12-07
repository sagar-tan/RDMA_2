import yfinance as yf
import pandas as pd
import numpy as np
import config
from utils.logger import setup_logger

logger = setup_logger("data_loader", "data_loader.log")

def fetch_and_process_data(ticker=config.ASSET_TICKER, force_download=False):
    """
    Fetches market data and engineers the volatility features required for regime detection.
    Includes caching to prevent repeated API calls to Yahoo Finance.
    """
    
    # 1. Check Cache First
    cache_path = config.DATA_DIR / f"{ticker}_processed.csv"
    
    if cache_path.exists() and not force_download:
        logger.info(f"Loading {ticker} data from cache: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['Date'], index_col='Date')
        return df

    # 2. Download Data if not cached
    logger.info(f"Downloading {ticker} data from Yahoo Finance ({config.START_DATE} to {config.END_DATE})...")
    df = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
    
    # Clean up Yahoo Finance MultiIndex columns (e.g., 'Close', 'SPY') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Keep only what we need
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 3. Feature Engineering (The Critical Part)
    logger.info("Calculating Volatility and Regime Features...")
    
    # Log Returns: The standard input for most quant models
    # ln(P_t / P_{t-1})
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Realized Volatility (Annualized)
    # We use a 21-day window (approx 1 trading month)
    # Formula: StdDev(LogReturns) * sqrt(252)
    df['Volatility'] = df['Log_Ret'].rolling(window=21).std() * np.sqrt(252)
    
    # Simple Trend Feature (Optional but useful for strategies)
    # 200-day Simple Moving Average
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 4. Cleanup and Save
    # Drop the rows that have NaN values (the first 200 days of history)
    original_len = len(df)
    df.dropna(inplace=True)
    dropped_len = original_len - len(df)
    
    logger.info(f"Dropped {dropped_len} rows due to rolling window initialization.")
    
    # Save to cache
    df.to_csv(cache_path)
    logger.info(f"Saved processed data to {cache_path}")
    
    return df

if __name__ == "__main__":
    # Quick test to see if it works
    data = fetch_and_process_data()
    print(data.head())
    print(data.tail())