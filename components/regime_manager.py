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
    Enforces 'State 0 = Low Volatility' to ensure consistency across experiments.
    """
    
    def __init__(self, n_states=config.HMM_STATES, n_iter=100):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42
        )
        self.state_map = {} # Maps {Internal_ID: Sorted_ID}
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        """
        Trains the HMM on the historical volatility data.
        Critically, it sorts the states so State 0 is always the lowest volatility.
        """
        if 'Volatility' not in data.columns:
            raise KeyError("Data missing 'Volatility' column. Run data_loader first.")

        # Prepare data for HMM (sklearn expects 2D array)
        vol_series = data['Volatility'].values.reshape(-1, 1)
        
        # Fit the model
        self.model.fit(vol_series)
        
        # --- THE SORTING LOGIC ---
        # 1. Get the mean variance/volatility for each hidden state
        means = self.model.means_.flatten()
        
        # 2. Sort indices: [Index of Lowest Mean, Index of Highest Mean]
        sorted_indices = np.argsort(means)
        
        # 3. Create a map: Original_ID -> New_Rank
        # Example: If State 2 has the lowest mean, it maps to 0.
        self.state_map = {original: new_rank for new_rank, original in enumerate(sorted_indices)}
        
        self.is_fitted = True
        logger.info(f"HMM Fitted. Volatility Means (Unsorted): {means}")
        logger.info(f"State Mapping Enforced (Original->Sorted): {self.state_map}")

    def detect_regime(self, row: pd.Series) -> int:
        """
        Determines the regime for a single data point.
        Returns: 0 (Safe/Low Vol), 1 (Risky/High Vol), etc.
        """
        if not self.is_fitted:
            raise ValueError("Regime Detector not fitted. Call fit() first.")
        
        # Reshape single point
        val = np.array([[row['Volatility']]])
        
        # Predict internal state
        internal_state = self.model.predict(val)[0]
        
        # Map to sorted state
        return self.state_map[internal_state]

    def predict_batch(self, data: pd.DataFrame) -> np.ndarray:
        """
        Optimized method to predict regimes for a whole dataframe at once.
        Used by the Backtest Engine for speed.
        """
        if not self.is_fitted:
            self.fit(data)
            
        vol_series = data['Volatility'].values.reshape(-1, 1)
        internal_states = self.model.predict(vol_series)
        
        # Vectorized mapping
        # Create a lookup array where index = original state, value = new state
        # e.g. lookup[2] gives the sorted rank of state 2
        lookup = np.empty(max(self.state_map.keys()) + 1, dtype=int)
        for orig, new in self.state_map.items():
            lookup[orig] = new
            
        return lookup[internal_states]