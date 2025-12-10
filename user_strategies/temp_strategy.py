import pandas as pd
import numpy as np
from interfaces import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("mean_reversion", "user_strategies.log")

class MeanReversionStrategy(BaseStrategy):
    """
    A 'Buy the Dip' strategy using RSI.
    
    Logic:
    - RSI < 30: Oversold -> BUY (Expect bounce).
    - RSI > 70: Overbought -> SELL (Go to Cash).
    
    The Trap:
    In a real crash (Regime 1), 'Oversold' stays Oversold for months. 
    A static version of this strategy will bankrupt itself buying the drop.
    The Regime Wrapper should prevent this "Knife Catching."
    """
    
    def __init__(self, period=14, buy_threshold=30, sell_threshold=70):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.rsi_col = f'RSI_{period}'

    def train(self, history: pd.DataFrame):
        """
        Calculate RSI on the historical data.
        """
        if self.rsi_col in history.columns:
            return

        logger.info(f"Calculating RSI({self.period}) for Mean Reversion...")
        
        # Calculate RSI manually using Pandas
        delta = history['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        history[self.rsi_col] = 100 - (100 / (1 + rs))
        
        # Fill NaN
        history[self.rsi_col] = history[self.rsi_col].fillna(50)

    def generate_signal(self, row: pd.Series) -> int:
        """
        Standard Mean Reversion Logic.
        """
        rsi = row.get(self.rsi_col, 50)
        
        if rsi < self.buy_threshold:
            return 1 # Buy the dip!
        elif rsi > self.sell_threshold:
            return 0 # Sell the rip!
        else:
            # Hold previous position? 
            # For simplicity in this backtester, let's say "Stay in Cash" if neutral
            # OR better: this framework is stateless between calls unless we check portfolio.
            # Let's simple "Long if < 30, Cash if > 70". 
            # What about between 30 and 70? 
            # Let's assume we hold if we are already long, but we don't have state here.
            # To make it robust:
            # If RSI < 50, bias Long? No that's Trend.
            
            # STRICT VERSION: Only hold if oversold.
            # This is very restrictive.
            
            # STATEFUL VERSION (Simulated):
            # We assume "1" means "I want to be Long".
            # Usually Mean Reversion enters at 30 and exits at 70.
            # Since we can't see current position easily in this method (stateless),
            # Let's implement a simpler "Contrarian" logic:
            # If Price < SMA(20) -> Buy. (Expect bounce to mean).
            return 1 if rsi < 50 else 0