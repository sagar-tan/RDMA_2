# Developer Quickstart

## What This Project Is

This repository is a small regime-aware backtesting framework.

Its core workflow is:

1. Load and preprocess market data
2. Train a strategy and a regime detector
3. Run a baseline strategy
4. Run a regime-filtered version of that same strategy
5. Compare results

The main source areas are:

- `main.py` for script-based benchmarking
- `app.py` for the Streamlit UI
- `components/` for core engine logic
- `user_strategies/` for trading strategies
- `interfaces.py` for extension contracts
- `config.py` for global runtime settings

## Fast Reading Order

If you are new to the codebase, read files in this order:

1. `interfaces.py`
2. `config.py`
3. `components/data_loader.py`
4. `components/regime_manager.py`
5. `components/strategy_wrapper.py`
6. `components/backtest_engine.py`
7. `components/portfolio.py`
8. `user_strategies/mean_reversion.py`
9. `main.py`
10. `app.py`

## Core Concepts

### Strategy

A strategy is any class that inherits from `BaseStrategy` and implements:

- `train(history)`
- `generate_signal(row)`

The current engine expects signals like:

- `1` for long
- `0` for cash

### Regime Detector

A regime detector is any class that inherits from `BaseRegimeDetector` and implements:

- `fit(data)`
- `detect_regime(row)`

The wrapper currently assumes:

- `0` means safe to trade
- non-zero means risky enough to block trading

### Wrapper

`RegimeAwareWrapper` composes a normal strategy with a regime detector. It lets you benchmark:

- raw strategy behavior
- regime-filtered strategy behavior

without rewriting the strategy.

## How To Run The Project

### Script run

Use:

```bash
python main.py
```

This runs the built-in benchmark using `MeanReversionStrategy` and its wrapped version.

### Streamlit app

Use:

```bash
streamlit run app.py
```

This launches the UI and lets you upload a strategy file.

## Important Files To Know

### `config.py`

Defines paths, ticker, date range, HMM states, capital, and transaction cost.

Important note:

- `app.py` mutates values in `config.py` at runtime

### `components/data_loader.py`

Loads cached or downloaded Yahoo Finance data and computes:

- `Log_Ret`
- `Volatility`
- `SMA_200`

### `components/regime_manager.py`

Defines `VolatilityHMM`, the current HMM-based regime detector.

### `components/strategy_wrapper.py`

Defines `RegimeAwareWrapper`, which blocks trades in risky regimes.

### `components/backtest_engine.py`

Runs the simulation, tracks each strategy separately, applies costs, and writes outputs.

### `components/portfolio.py`

Maintains equity history, trades, and basic stats.

## How To Add A New Strategy

Create a file in `user_strategies/`, for example `user_strategies/my_strategy.py`.

Basic template:

```python
import pandas as pd
from interfaces import BaseStrategy


class MyStrategy(BaseStrategy):
    def train(self, history: pd.DataFrame):
        pass

    def generate_signal(self, row: pd.Series) -> int:
        return 1 if row["Close"] > row["SMA_200"] else 0
```

Rules to follow:

- Keep `generate_signal()` row-based
- Use columns that already exist or create them in `train()`
- Return consistent numeric signals
- Avoid overwriting columns used by other strategies

## How To Add A New Regime Detector

Create a detector class that follows `BaseRegimeDetector`.

Basic template:

```python
import pandas as pd
from interfaces import BaseRegimeDetector


class MyDetector(BaseRegimeDetector):
    def fit(self, data: pd.DataFrame):
        pass

    def detect_regime(self, row: pd.Series) -> int:
        return 0
```

Then pass it into `RegimeAwareWrapper` instead of `VolatilityHMM`.

## Minimal Benchmark Example

```python
from components.data_loader import fetch_and_process_data
from components.backtest_engine import BacktestEngine
from components.strategy_wrapper import RegimeAwareWrapper
from components.regime_manager import VolatilityHMM
from user_strategies.mean_reversion import MeanReversionStrategy


data = fetch_and_process_data()
strategy = MeanReversionStrategy()
detector = VolatilityHMM()
wrapped = RegimeAwareWrapper(strategy, detector)

engine = BacktestEngine(data)
engine.add_strategy(strategy)
engine.add_strategy(wrapped)
results = engine.run()
```

## Outputs

The project writes outputs to:

- `output/equity_<strategy>.csv`
- `output/trades_<strategy>.csv`
- `output/benchmark_chart.png` from `main.py`

Logs are written under `logs/`.

## Important Caveats Before Extending

- Strategies are trained on the full dataset before simulation begins
- Strategies may mutate the shared dataframe during `train()`
- `app.py` executes uploaded Python code directly
- The engine's signal timing is simplified and should be reviewed before building more realistic logic
- The current framework is mostly long/cash rather than fully long/short

## Best Next Step

If you want to extend the framework seriously, read `PROJECT_IMPLEMENTATION_DOC.md` first and then `docs/extending.md` for extension-focused guidance.
