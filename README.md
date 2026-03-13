# Regime-Aware Trading Framework Extensions

## Overview

This repository is a regime-aware backtesting framework for comparing a base trading strategy against a regime-filtered version of the same strategy.

Core idea:

1. Load historical market data
2. Train a strategy
3. Train a regime detector
4. Wrap the strategy with `RegimeAwareWrapper`
5. Run the backtest with `BacktestEngine`
6. Compare returns, drawdown, Sharpe, and trade behavior

The project now includes expanded benchmarking support across multiple assets, multiple built-in strategies, and multiple regime detectors.

## Repository Layout

- `main.py` - simple script benchmark using `MeanReversionStrategy` plus `VolatilityHMM`
- `benchmark_suite.py` - expanded benchmark runner across assets, strategies, and detectors
- `app.py` - Streamlit UI for uploading and benchmarking a strategy file
- `components/` - backtest engine, data loader, regime detectors, wrapper, portfolio logic
- `user_strategies/` - built-in and user-created strategies
- `interfaces.py` - `BaseStrategy` and `BaseRegimeDetector`
- `config.py` - paths and runtime defaults
- `docs/extending.md` - extension rules and implementation guidance
- `DEVELOPER_QUICKSTART.md` - codebase orientation for developers
- `PROJECT_IMPLEMENTATION_DOC.md` - deeper implementation reference

## Installed Strategy Set

- `MeanReversionStrategy` - RSI-based contrarian strategy
- `TrendFollowingStrategy` - price vs. `SMA_200`
- `MomentumStrategy` - previous `Log_Ret` sign
- `BreakoutStrategy` - 20-day rolling high breakout
- `MovingAverageCrossStrategy` - `SMA_50` vs. `SMA_200`
- `VolatilityBreakoutStrategy` - prior close plus `ATR14`

## Installed Regime Detectors

- `VolatilityHMM` - hidden Markov model on `Volatility`
- `VolatilityThresholdDetector` - median volatility threshold
- `DrawdownRegimeDetector` - rolling drawdown threshold

Wrapper rule in the current framework:

- regime `0` = trading allowed
- regime `1` = trading blocked and forced to cash

## Requirements

Minimum Python version:

- Python 3.10+

Recommended packages:

```bash
pip install pandas numpy matplotlib yfinance streamlit hmmlearn
```

Notes:

- `yfinance` is required only when downloading fresh data
- `hmmlearn` is required only when using `VolatilityHMM`
- cached CSV data in `data_storage/` can still be used without `yfinance`

## Quick Start

### 1. Run the basic benchmark

```bash
python main.py
```

What it does:

- loads `config.ASSET_TICKER`
- runs `MeanReversionStrategy`
- runs `RegimeAware(MeanReversionStrategy)` using `VolatilityHMM`
- writes equity and trade CSVs into `output/`
- saves `output/benchmark_chart.png`

### 2. Run the expanded benchmark suite

```bash
python benchmark_suite.py
```

What it does:

- loops through assets: `SPY`, `GLD`, `BTC-USD`, `EURUSD=X`
- runs all built-in strategies
- runs baseline plus regime-filtered variants
- evaluates detector set: `None`, `VolatilityHMM`, `VolatilityThresholdDetector`, `DrawdownRegimeDetector`
- writes aggregate results to `output/benchmark_results.csv`
- saves `output/benchmark_chart.png`
- saves `output/regime_overlay_chart.png`

Important:

- fresh assets require `yfinance`
- `VolatilityHMM` runs require `hmmlearn`
- the last regime overlay written will overwrite previous `output/regime_overlay_chart.png`

### 3. Launch the Streamlit UI

```bash
streamlit run app.py
```

What it does:

- lets you choose ticker and HMM state count
- lets you upload a strategy file
- benchmarks the uploaded strategy raw vs. regime-filtered
- plots equity curves and regime shading in the browser

## How Data Loading Works

Data is handled by `components/data_loader.py`.

Workflow:

1. Check `data_storage/<ticker>_processed.csv`
2. If present, load from cache
3. Otherwise download from Yahoo Finance
4. Engineer these core columns:
   - `Log_Ret`
   - `Volatility`
   - `SMA_200`
5. Drop initial NaN rows caused by rolling windows
6. Save the processed data back to cache

The framework expects strategies and detectors to rely only on:

- columns already created by the loader
- columns they create themselves in `train()` or `fit()`

## How To Configure Runs

Edit `config.py` to change:

- `ASSET_TICKER`
- `START_DATE`
- `END_DATE`
- `HMM_STATES`
- `INITIAL_CAPITAL`
- `TRANSACTION_COST`

Example workflow:

```python
ASSET_TICKER = "GLD"
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
HMM_STATES = 2
```

Then run:

```bash
python main.py
```

## Output Files

Common outputs written under `output/`:

- `equity_<strategy>.csv`
- `trades_<strategy>.csv`
- `benchmark_chart.png`
- `regime_overlay_chart.png`
- `benchmark_results.csv`

Logs are written under `logs/`.

## Expected Signal and Regime Semantics

Strategies should return:

- `1` = long
- `0` = cash

The current framework is primarily long/cash even though the interface mentions `-1` as a possible extension.

Detectors should return:

- `0` = safe regime
- `1` = risky regime

`RegimeAwareWrapper` is the only place where regime filtering is applied.

## Creating Your Own Strategy

All user strategies should live in `user_strategies/` and inherit from `BaseStrategy`.

Minimal template:

```python
import pandas as pd

from interfaces import BaseStrategy


class MyStrategy(BaseStrategy):
    def train(self, history: pd.DataFrame):
        if "MyFeature" not in history.columns:
            history["MyFeature"] = history["Close"].rolling(10).mean()

    def generate_signal(self, row: pd.Series) -> int:
        if pd.isna(row.get("MyFeature")):
            return 0
        return int(row["Close"] > row["MyFeature"])
```

Rules:

- do not modify `BaseStrategy`
- keep the strategy inside `user_strategies/`
- do not directly modify the regime detector from a strategy
- avoid overwriting shared canonical columns
- only depend on loader columns or columns created in `train()`

## Creating Your Own Regime Detector

All detectors must inherit `BaseRegimeDetector`.

Minimal template:

```python
import pandas as pd

from interfaces import BaseRegimeDetector


class MyRegimeDetector(BaseRegimeDetector):
    def fit(self, data: pd.DataFrame):
        self.threshold = float(data["Volatility"].median())

    def detect_regime(self, row: pd.Series) -> int:
        return int(row["Volatility"] > self.threshold)
```

Rules:

- do not modify `BaseRegimeDetector`
- keep regime filtering inside `RegimeAwareWrapper`
- make sure `detect_regime()` works for the full dataset
- return deterministic `0` or `1` semantics

## Example: Custom Benchmark Script

```python
from components.backtest_engine import BacktestEngine
from components.data_loader import fetch_and_process_data
from components.regime_detectors import VolatilityThresholdDetector
from components.strategy_wrapper import RegimeAwareWrapper
from user_strategies.momentum import MomentumStrategy


data = fetch_and_process_data("SPY")
baseline = MomentumStrategy()
wrapped = RegimeAwareWrapper(MomentumStrategy(), VolatilityThresholdDetector())

engine = BacktestEngine(data)
engine.add_strategy(baseline)
engine.add_strategy(wrapped)
results = engine.run()

print(results.tail())
```

## Validation Checklist

Before trusting a new extension, confirm:

- all strategies train without column conflicts
- all detectors produce regimes for the full dataset
- baseline and wrapped strategies both execute
- `main.py` still runs
- output CSV files are created as expected

## Known Limitations

- strategies are trained on the full dataset before simulation
- the portfolio model is simplified and long/cash oriented
- signal timing is simplified relative to a production execution engine
- `app.py` executes uploaded Python code directly
- `benchmark_suite.py` currently overwrites the shared overlay chart file on each run

## Recommended Reading Order

1. `README.md`
2. `DEVELOPER_QUICKSTART.md`
3. `docs/extending.md`
4. `PROJECT_IMPLEMENTATION_DOC.md`

## Troubleshooting

### `ModuleNotFoundError: No module named 'yfinance'`

Install it:

```bash
pip install yfinance
```

### `ModuleNotFoundError: No module named 'hmmlearn'`

Install it:

```bash
pip install hmmlearn
```

### No downloaded data appears

Check:

- your internet connection
- the ticker symbol
- whether `data_storage/` already contains a cached file

### Streamlit upload does not load your strategy

Check that:

- the file contains a class inheriting `BaseStrategy`
- the class can be instantiated with no required constructor arguments
- imports inside the uploaded file are valid in the local environment
