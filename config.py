from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_storage"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# ==========================================
# 2. EXPERIMENT SETTINGS
# ==========================================
# The asset to run the benchmark on
ASSET_TICKER = "SPY" 

# Date range for the study (Longer is better for regimes)
START_DATE = "2000-01-01" 
END_DATE = "2025-01-01"

# ==========================================
# 3. REGIME DETECTION SETTINGS
# ==========================================
# Number of hidden states (0=Calm, 1=Volatile)
HMM_STATES = 2

# Rolling window for HMM training (in trading days)
# 1000 days = approx 4 years. Enough to capture a full cycle.
HMM_TRAIN_WINDOW = 1000 

# Re-fit frequency (in days) to save computation time
# 20 = Re-train HMM once a month
HMM_REFIT_INTERVAL = 20

# ==========================================
# 4. TRADING FRICTION
# ==========================================
# Initial Portfolio Value
INITIAL_CAPITAL = 10000.0

# Transaction Cost per trade (as a fraction)
# 0.0005 = 5 basis points (Standard for liquid ETFs)
# 0.0010 = 10 basis points (Crypto/Small Cap)
TRANSACTION_COST = 0.0005

# ==========================================
# 5. ANALYSIS SETTINGS
# ==========================================
# Risk-Free Rate for Sharpe Ratio calculation (annualized)
RISK_FREE_RATE = 0.02