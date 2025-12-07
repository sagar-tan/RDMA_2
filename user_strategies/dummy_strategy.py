import pandas as pd
from interfaces import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("dummy_strategy", "user_strategies.log")

class TrendFollowingStrategy(BaseStrategy):
    """
    A classic 'Trend Following' strategy.
    
    Logic:
    - If Close > SMA_200: Market is in an Uptrend -> Buy (Long).
    - If Close < SMA_200: Market is in a Downtrend -> Sell (Cash).
    
    Hypothesis:
    This strategy works well in stable trends but fails in high volatility.
    The Regime Wrapper should significantly improve its Sharpe Ratio.
    """
    
    def __init__(self):
        self.sma_col = 'SMA_200'

    def train(self, history: pd.DataFrame):
        """
        No ML training required for this static rule.
        We just verify the data is ready.
        """
        if self.sma_col not in history.columns:
            logger.warning(f"Feature {self.sma_col} missing! Strategy might fail.")
        else:
            logger.info(f"{self.get_name()} ready. Using pre-calculated {self.sma_col}.")

    def generate_signal(self, row: pd.Series) -> int:
        """
        Decide position based on simple Trend logic.
        """
        # Defensive check for missing data
        if pd.isna(row[self.sma_col]) or pd.isna(row['Close']):
            return 0
            
        # The Logic: Ride the trend
        if row['Close'] > row[self.sma_col]:
            return 1 # Long
        else:
            return 0 # Cash (Safety)