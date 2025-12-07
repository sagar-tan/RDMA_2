# config.py
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_storage"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Strategy Settings ---
ASSET_TICKER = "SPY"        # Risk-On Asset
SAFE_ASSET = "IEF"          # Risk-Off Asset (7-10yr Treasury) or "CASH"
START_DATE = "2005-01-01"   # Long history covers 2008 crash
END_DATE = "2025-01-01"

# --- Regime Settings ---
HMM_STATES = 2              # 0 = Calm, 1 = Volatile
HMM_WINDOW = 252 * 2        # Train HMM on rolling 2 years
REBALANCE_FREQ = "weekly"   # Check regime weekly to save costs (New!)

# --- Costs ---
TRANSACTION_COST = 0.0005   # 5 bps