import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from interfaces import BaseRegimeDetector
import config
from utils.logger import setup_logger

logger = setup_logger("regime_manager", "regime_manager.log")

class VolatilityHMM(BaseRegimeDetector):
    """
    Implements the Hidden Markov Model detection logic.
    Enforces 'State 0 = Low Volatility' to ensure consistency.
    FIXED: Uses sequence pre-calculation instead of stateless single-point prediction.
    """
    # Add this method back to VolatilityHMM class
    def predict_batch(self, data: pd.DataFrame) -> np.ndarray:
        """
        Helper for plotting: Returns the full sequence of regimes for the dataset.
        """
        if self.regime_history is None:
            self.fit(data)
        
        # Align with the input data indices
        # If data matches training data exactly, return history
        if len(data) == len(self.regime_history):
            return self.regime_history.values
        else:
            # Fallback: Re-predict (fast since models are fitted)
            vol_series = data['Volatility'].values.reshape(-1, 1)
            internal_states = self.model.predict(vol_series)
            return np.array([self.state_map[s] for s in internal_states])
    
    def __init__(self, n_states=config.HMM_STATES, n_iter=100):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42
        )
        self.state_map = {} 
        self.is_fitted = False
        self.regime_history = None # Lookup table

    def fit(self, data: pd.DataFrame):
        """
        Trains the HMM and pre-calculates the regime sequence for the entire history.
        """
        if 'Volatility' not in data.columns:
            raise KeyError("Data missing 'Volatility' column. Run data_loader first.")

        # Prepare data
        vol_series = data['Volatility'].values.reshape(-1, 1)
        
        # 1. Fit the model
        self.model.fit(vol_series)
        
        # 2. Sort states (0 = Lowest Volatility)
        means = self.model.means_.flatten()
        sorted_indices = np.argsort(means)
        self.state_map = {original: new_rank for new_rank, original in enumerate(sorted_indices)}
        
        # 3. Pre-calculate the Viterbi Path (The Sequence)
        internal_states = self.model.predict(vol_series)
        sorted_states = np.array([self.state_map[s] for s in internal_states])
        
        # 4. Store as a Lookup Series indexed by Date
        self.regime_history = pd.Series(sorted_states, index=data.index)
        self.is_fitted = True
        
        logger.info(f"HMM Fitted & Regimes Pre-calculated.")
        logger.info(f"State Means: {means}")
        logger.info(f"State Map: {self.state_map}")
        
        # Log distribution for sanity check
        counts = self.regime_history.value_counts()
        logger.info(f"Regime Distribution: {counts.to_dict()}")

    def detect_regime(self, row: pd.Series) -> int:
        """
        Look up the pre-calculated regime for the specific date.
        """
        if not self.is_fitted:
            raise ValueError("Regime Detector not fitted. Call fit() first.")
        
        # Use the Date (Index) to lookup the regime
        date = row.name 
        
        if date in self.regime_history.index:
            return int(self.regime_history.loc[date])
        else:
            # Fallback for out-of-sample data (shouldn't happen in this benchmark)
            # We treat this as a single point prediction if needed
            logger.warning(f"Date {date} not in training history. Using stateless prediction.")
            val = np.array([[row['Volatility']]])
            internal = self.model.predict(val)[0]
            return self.state_map[internal]