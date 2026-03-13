import pandas as pd

from interfaces import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("volatility_breakout_strategy", "user_strategies.log")


class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, atr_window=14):
        self.atr_window = atr_window
        self.true_range_col = "TrueRange"
        self.atr_col = f"ATR{atr_window}"
        self.breakout_level_col = f"VolatilityBreakoutLevel{atr_window}"

    def train(self, history: pd.DataFrame):
        required_cols = {"High", "Low", "Close"}
        missing_cols = required_cols.difference(history.columns)
        if missing_cols:
            raise KeyError(f"Data missing required columns: {sorted(missing_cols)}")

        if self.true_range_col not in history.columns:
            prev_close = history["Close"].shift(1)
            true_range = pd.concat(
                [
                    history["High"] - history["Low"],
                    (history["High"] - prev_close).abs(),
                    (history["Low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            history[self.true_range_col] = true_range

        if self.atr_col not in history.columns:
            history[self.atr_col] = history[self.true_range_col].rolling(window=self.atr_window).mean()

        if self.breakout_level_col not in history.columns:
            history[self.breakout_level_col] = history["Close"].shift(1) + history[self.atr_col]

        logger.info("VolatilityBreakoutStrategy ready using %s.", self.atr_col)

    def generate_signal(self, row: pd.Series) -> int:
        close = row.get("Close")
        breakout_level = row.get(self.breakout_level_col)

        if pd.isna(close) or pd.isna(breakout_level):
            return 0

        return int(close > breakout_level)
