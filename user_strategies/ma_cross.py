import pandas as pd

from interfaces import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("ma_cross_strategy", "user_strategies.log")


class MovingAverageCrossStrategy(BaseStrategy):
    def __init__(self, fast_window=50, slow_col="SMA_200"):
        self.fast_window = fast_window
        self.fast_col = f"SMA_{fast_window}"
        self.slow_col = slow_col

    def train(self, history: pd.DataFrame):
        if "Close" not in history.columns:
            raise KeyError("Data missing 'Close' column. Run data_loader first.")
        if self.slow_col not in history.columns:
            raise KeyError(f"Data missing '{self.slow_col}' column. Run data_loader first.")

        if self.fast_col not in history.columns:
            history[self.fast_col] = history["Close"].rolling(window=self.fast_window).mean()

        logger.info("MovingAverageCrossStrategy ready using %s and %s.", self.fast_col, self.slow_col)

    def generate_signal(self, row: pd.Series) -> int:
        fast_ma = row.get(self.fast_col)
        slow_ma = row.get(self.slow_col)

        if pd.isna(fast_ma) or pd.isna(slow_ma):
            return 0

        return int(fast_ma > slow_ma)
