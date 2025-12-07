from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    The Contract: Any strategy plugged into this framework MUST inherit from this class.
    This ensures the Backtest Engine knows exactly how to talk to it.
    """

    @abstractmethod
    def generate_signal(self, row: pd.Series) -> int:
        """
        The core logic. Decides what to do for a single day.
        
        Args:
            row (pd.Series): A single row of data (Open, High, Low, Close, Volatility, etc.)
            
        Returns:
            int: The target position signal.
                 1  = Long (Buy)
                 0  = Cash (Flat)
                -1  = Short (Sell) - Optional, can be treated as Cash if unsupported
        """
        pass

    @abstractmethod
    def train(self, history: pd.DataFrame):
        """
        Training logic. 
        - For Static strategies (e.g., RSI), this can be `pass`.
        - For ML strategies (e.g., XGBoost), this is where model.fit() happens.
        
        Args:
            history (pd.DataFrame): Historical data available for training.
                                    Strictly NO look-ahead bias allowed here.
        """
        pass

    def get_name(self) -> str:
        """
        Returns the class name. Helpful for logging.
        """
        return self.__class__.__name__


class BaseRegimeDetector(ABC):
    """
    The Interface for the "Gatekeeper".
    Allows us to swap HMM for Changepoint, VIX Threshold, or LLM-based detection easily.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Train the detector (e.g., fit the Gaussian HMM).
        """
        pass

    @abstractmethod
    def detect_regime(self, row: pd.Series) -> int:
        """
        Determine the market state for a specific day.
        
        Returns:
            int: 0 = Safe/Calm (Trading Allowed)
                 1 = Risky/Volatile (Trading Vetoed)
        """
        pass