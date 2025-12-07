# components/regime_detector.py
import numpy as np
from hmmlearn.hmm import GaussianHMM

class VolatilityRegimeDetector:
    def __init__(self, n_states=2):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    
    def fit_predict(self, vol_series):
        # vol_series needs to be reshaped for sklearn
        X = vol_series.values.reshape(-1, 1)
        self.model.fit(X)
        
        # --- ENFORCE ORDER ---
        # We want State 0 = Low Vol, State 1 = High Vol
        # We sort the states by their Mean Volatility
        means = self.model.means_.flatten()
        sorted_idx = np.argsort(means)
        
        # Create a map: Old_ID -> New_Sorted_ID
        mapper = {old: new for new, old in enumerate(sorted_idx)}
        
        # Predict hidden states
        hidden_states = self.model.predict(X)
        
        # Remap them so 0 is always Low Vol
        sorted_states = np.array([mapper[s] for s in hidden_states])
        
        return sorted_states