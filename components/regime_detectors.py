# pyright: reportMissingImports=false, reportGeneralTypeIssues=false

import numpy as np
import pandas as pd
try:
    from hmmlearn.hmm import GaussianHMM
except ModuleNotFoundError:
    GaussianHMM = None

import config
from interfaces import BaseRegimeDetector
from utils.logger import setup_logger

logger = setup_logger("regime_detectors", "regime_manager.log")


class VolatilityHMM(BaseRegimeDetector):
    def __init__(self, n_states=config.HMM_STATES, n_iter=100):
        if GaussianHMM is None:
            raise ModuleNotFoundError("VolatilityHMM requires 'hmmlearn' to be installed.")

        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
        )
        self.state_map = {}
        self.is_fitted = False
        self.regime_history: pd.Series | None = None

    def predict_batch(self, data: pd.DataFrame) -> np.ndarray:
        if self.regime_history is None:
            self.fit(data)

        regime_history = self.regime_history
        if regime_history is not None and len(data) == len(regime_history):
            return regime_history.to_numpy(dtype=int)

        vol_series = data[["Volatility"]].to_numpy()
        internal_states = self.model.predict(vol_series)
        return np.array([self.state_map[s] for s in internal_states])

    def fit(self, data: pd.DataFrame):
        if "Volatility" not in data.columns:
            raise KeyError("Data missing 'Volatility' column. Run data_loader first.")

        vol_series = data[["Volatility"]].to_numpy()
        self.model.fit(vol_series)

        means = self.model.means_.flatten()
        sorted_indices = np.argsort(means)
        self.state_map = {original: new_rank for new_rank, original in enumerate(sorted_indices)}

        internal_states = self.model.predict(vol_series)
        sorted_states = np.array([self.state_map[s] for s in internal_states])

        self.regime_history = pd.Series(sorted_states, index=data.index)
        self.is_fitted = True

        logger.info("HMM fitted and regimes pre-calculated.")
        logger.info("State Means: %s", means)
        logger.info("State Map: %s", self.state_map)
        logger.info("Regime Distribution: %s", self.regime_history.value_counts().to_dict())

    def detect_regime(self, row: pd.Series) -> int:
        if not self.is_fitted:
            raise ValueError("Regime Detector not fitted. Call fit() first.")

        regime_history = self.regime_history
        if regime_history is None:
            raise ValueError("Regime history unavailable. Call fit() first.")

        date = row.name
        if date in regime_history.index:
            return int(regime_history.loc[date])

        logger.warning("Date %s not in training history. Using stateless prediction.", date)
        val = np.array([[row["Volatility"]]])
        internal = self.model.predict(val)[0]
        return int(self.state_map[internal])


class VolatilityThresholdDetector(BaseRegimeDetector):
    def __init__(self, volatility_col="Volatility"):
        self.volatility_col = volatility_col
        self.threshold: float | None = None

    def fit(self, data: pd.DataFrame):
        if self.volatility_col not in data.columns:
            raise KeyError(f"Data missing '{self.volatility_col}' column. Run data_loader first.")

        threshold = np.median(data[self.volatility_col].to_numpy(dtype=float))
        self.threshold = float(threshold)
        logger.info("VolatilityThresholdDetector fitted with threshold %.6f.", self.threshold)

    def detect_regime(self, row: pd.Series) -> int:
        if self.threshold is None:
            raise ValueError("Regime Detector not fitted. Call fit() first.")

        threshold = self.threshold
        volatility = row[self.volatility_col] if self.volatility_col in row.index else np.nan
        if pd.isna(volatility):
            return 0

        return int(float(volatility) > threshold)


class DrawdownRegimeDetector(BaseRegimeDetector):
    def __init__(self, price_col="Close", window=252, quantile=0.75):
        self.price_col = price_col
        self.window = window
        self.quantile = quantile
        self.rolling_peak_col = f"RollingPeak{window}"
        self.drawdown_col = f"Drawdown{window}"
        self.threshold: float | None = None

    def fit(self, data: pd.DataFrame):
        if self.price_col not in data.columns:
            raise KeyError(f"Data missing '{self.price_col}' column. Run data_loader first.")

        if self.rolling_peak_col not in data.columns:
            data[self.rolling_peak_col] = data[self.price_col].rolling(window=self.window, min_periods=1).max()

        if self.drawdown_col not in data.columns:
            rolling_peak = data[self.rolling_peak_col].replace(0, np.nan)
            data[self.drawdown_col] = ((rolling_peak - data[self.price_col]) / rolling_peak).fillna(0.0)

        threshold = np.quantile(data[self.drawdown_col].to_numpy(dtype=float), self.quantile)
        self.threshold = float(threshold)
        logger.info("DrawdownRegimeDetector fitted with threshold %.6f.", self.threshold)

    def detect_regime(self, row: pd.Series) -> int:
        if self.threshold is None:
            raise ValueError("Regime Detector not fitted. Call fit() first.")

        threshold = self.threshold
        drawdown = row[self.drawdown_col] if self.drawdown_col in row.index else np.nan
        if pd.isna(drawdown):
            return 0

        return int(float(drawdown) > threshold)
