import pandas as pd

from interfaces import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("momentum_strategy", "user_strategies.log")


class MomentumStrategy(BaseStrategy):
    def __init__(self):
        self.return_col = "Log_Ret"
        self.signal_col = "MomentumPrevSignal"

    def train(self, history: pd.DataFrame):
        if self.return_col not in history.columns:
            raise KeyError("Data missing 'Log_Ret' column. Run data_loader first.")

        if self.signal_col not in history.columns:
            history[self.signal_col] = (history[self.return_col].shift(1) > 0).astype(int)

        logger.info("MomentumStrategy ready using previous log return sign.")

    def generate_signal(self, row: pd.Series) -> int:
        return int(row.get(self.signal_col, 0))
