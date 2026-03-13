import pandas as pd

from interfaces import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("breakout_strategy", "user_strategies.log")


class BreakoutStrategy(BaseStrategy):
    def __init__(self, window=20):
        self.window = window
        self.high_col = f"RollingHigh{window}"

    def train(self, history: pd.DataFrame):
        if "Close" not in history.columns:
            raise KeyError("Data missing 'Close' column. Run data_loader first.")

        if self.high_col not in history.columns:
            history[self.high_col] = history["Close"].rolling(window=self.window).max().shift(1)

        logger.info("BreakoutStrategy ready using %s.", self.high_col)

    def generate_signal(self, row: pd.Series) -> int:
        rolling_high = row.get(self.high_col)
        close = row.get("Close")

        if pd.isna(close) or pd.isna(rolling_high):
            return 0

        return int(close > rolling_high)
