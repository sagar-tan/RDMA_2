import pandas as pd
from interfaces import BaseStrategy, BaseRegimeDetector
from utils.logger import setup_logger

logger = setup_logger("strategy_wrapper", "wrapper.log")

class RegimeAwareWrapper(BaseStrategy):
    """
    The Decorator: Wraps a user-defined strategy with regime-aware logic.
    
    Logic:
    1. Check Market Regime (via Detector).
    2. If Regime == 0 (Low Volatility/Safe):
       -> Execute User Strategy (Pass-through).
    3. If Regime == 1 (High Volatility/Risky):
       -> Force Cash (Signal 0) OR Short (if configured).
    """
    
    def __init__(self, strategy: BaseStrategy, detector: BaseRegimeDetector):
        self.strategy = strategy
        self.detector = detector
        self.name = f"RegimeAware({strategy.get_name()})"

    def train(self, history: pd.DataFrame):
        """
        Trains both the internal strategy (if ML) and the regime detector.
        """
        logger.info(f"Training Regime Detector and Inner Strategy: {self.strategy.get_name()}")
        
        # 1. Fit the Regime Detector (HMM)
        self.detector.fit(history)
        
        # 2. Train the User's Strategy (if it needs training)
        self.strategy.train(history)

    def generate_signal(self, row: pd.Series) -> int:
        """
        Generates the final signal after regime filtering.
        """
        # 1. Check the Weather (Regime)
        regime = self.detector.detect_regime(row)
        
        # 2. Get User's Opinion
        raw_signal = self.strategy.generate_signal(row)
        
        # 3. Apply The Filter
        if regime == 0:
            # Safe Regime: Trust the strategy
            return raw_signal
        else:
            # Risky Regime: Safety First (Cash)
            # Note: We return 0 (Cash). 
            # In V2, you could return -1 (Short) if you wanted "Crisis Alpha".
            return 0

    def get_name(self) -> str:
        return self.name