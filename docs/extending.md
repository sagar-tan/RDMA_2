# Extending the Framework

## Purpose

This document explains how to extend the current framework safely and predictably. It focuses on the two main extension axes:

- adding new trading strategies
- adding new regime detectors

It also explains the implementation constraints that matter when you start evolving the engine itself.

## The Two Main Extension Contracts

The framework is built around two abstract interfaces in `interfaces.py`:

- `BaseStrategy`
- `BaseRegimeDetector`

Everything extensible should plug into one of those contracts.

## Extending With New Strategies

### Required interface

Every strategy must inherit from `BaseStrategy` and implement:

- `train(self, history: pd.DataFrame)`
- `generate_signal(self, row: pd.Series) -> int`

### Current engine expectations

The engine currently expects strategy behavior to fit this model:

- training happens once before simulation
- signal generation happens row by row
- signals are numeric and usually `0` or `1`
- the strategy can rely on a pandas row containing all needed features

### Recommended development pattern

For a new strategy:

1. Decide what features it needs.
2. Decide whether those features belong in `components/data_loader.py` or inside the strategy's own `train()` method.
3. Keep `generate_signal()` lightweight and deterministic.
4. Make sure the strategy behaves sensibly if data is missing.
5. Make the strategy name stable so output files remain easy to inspect.

### Where to compute features

You currently have two viable patterns.

#### Pattern 1: Centralized features in `data_loader.py`

Use this when:

- many strategies may share the feature
- the feature is general-purpose market context
- the feature should be cached with the dataset

Examples:

- moving averages
- volatility estimates
- returns
- broad trend features

#### Pattern 2: Strategy-local features in `train()`

Use this when:

- the feature is strategy-specific
- it is not needed by the rest of the system
- you want to keep the central loader simple

Example:

- `MeanReversionStrategy` computes RSI during `train()`

### Important warning about shared dataframe mutation

The engine passes the same dataframe into every strategy's `train()` call. If a strategy adds or modifies columns, that affects the shared training dataset.

That means:

- adding a new column is usually safe if the name is unique
- overwriting an existing column can break other strategies
- mutating base market columns like `Close` or `Log_Ret` is dangerous

Recommended practice:

- use unique feature names
- never overwrite canonical columns produced by `data_loader.py`
- if many strategies need complex preprocessing, consider refactoring toward copied datasets or a formal feature pipeline later

### Strategy template

```python
import pandas as pd
from interfaces import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.high_col = f"RollingHigh_{lookback}"

    def train(self, history: pd.DataFrame):
        if self.high_col not in history.columns:
            history[self.high_col] = history["Close"].rolling(self.lookback).max()

    def generate_signal(self, row: pd.Series) -> int:
        if pd.isna(row.get(self.high_col)):
            return 0
        return 1 if row["Close"] >= row[self.high_col] else 0
```

### Strategy integration path

A new strategy can be used in three ways:

1. Import it directly into `main.py`
2. Add it manually to a custom script using `BacktestEngine`
3. Upload it through the Streamlit UI if it is a valid `BaseStrategy` subclass

## Extending With New Regime Detectors

### Required interface

Every detector must inherit from `BaseRegimeDetector` and implement:

- `fit(self, data: pd.DataFrame)`
- `detect_regime(self, row: pd.Series) -> int`

### Semantics matter more than the model type

The wrapper does not care whether your detector is based on:

- HMMs
- volatility thresholds
- trend state clustering
- macro labels
- VIX-like features
- ML classifiers

What it does care about is the meaning of the returned integer.

Right now, `RegimeAwareWrapper` assumes:

- `0` means safe to trade
- anything else means do not trade

So when you design a new detector, either:

- preserve that convention
- or update the wrapper to support richer policy logic

### Recommended detector design pattern

The existing `VolatilityHMM` uses a good pattern for this codebase:

1. Fit once on the dataset
2. Precompute a date-indexed regime series
3. Perform row-level lookup during simulation

This is a strong fit for the current engine because the backtest loop is simple and row-based.

### Detector template

```python
import pandas as pd
from interfaces import BaseRegimeDetector


class ThresholdVolDetector(BaseRegimeDetector):
    def __init__(self, threshold=0.25):
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        if "Volatility" not in data.columns:
            raise KeyError("Expected 'Volatility' in input data")
        self.is_fitted = True

    def detect_regime(self, row: pd.Series) -> int:
        if not self.is_fitted:
            raise ValueError("Detector not fitted")
        return 1 if row["Volatility"] >= self.threshold else 0
```

### If you need plotting support

If you want shaded chart overlays like the HMM flow uses, add a batch helper similar to `predict_batch()` in `VolatilityHMM`.

That helper is not part of the official interface, but it is useful for visualization.

## Extending The Wrapper Layer

`RegimeAwareWrapper` is intentionally minimal. Right now it supports a single policy:

- allow strategy signal in safe regime
- force cash in risky regime

If you want more expressive behavior, this is the module to evolve.

Possible wrapper extensions:

- allow shorting in risky regimes instead of going to cash
- support multiple regime-specific policies
- support partial exposure rather than binary on/off behavior
- support strategy switching by regime
- support different detectors for different asset groups

For example, future wrapper logic might look like:

- regime `0`: normal strategy
- regime `1`: reduced exposure
- regime `2`: defensive short hedge

The current interfaces are simple enough that this can be added without redesigning the entire project.

## Extending The Backtest Engine

If your goal is not just new strategies or detectors, but deeper realism, `components/backtest_engine.py` and `components/portfolio.py` are the main places to work.

### Current limitations to understand first

#### 1. Full-history training

The engine calls `train()` before simulation and gives each strategy the full dataset. This is not a walk-forward setup.

#### 2. Simplified signal timing

The engine calculates a signal from the current row and applies it immediately with the current day's return through `Portfolio.step()`. That is simple, but it may not match your intended execution model.

#### 3. Minimal portfolio model

The current portfolio model uses signal-gated return participation rather than explicit position objects, holdings, or order execution.

### High-value engine improvements

If you plan to upgrade the framework, these are the most meaningful next changes:

#### Walk-forward or rolling training

Train strategies only on data available up to each point in time.

#### Explicit execution lag

Use today's signal to set tomorrow's position instead of today's.

#### Position sizing

Support fractional allocations instead of only binary long/cash states.

#### Long/short support

Formalize how `-1` should behave in both transaction costs and portfolio PnL.

#### Better analytics

Centralize metrics such as Sharpe, drawdown, volatility, turnover, and hit rate.

#### Better output abstraction

Separate simulation results from file-writing side effects.

## Extending The Streamlit App

`app.py` is useful for experimentation, but keep these constraints in mind:

- it mutates `config.py` at runtime
- it writes uploaded strategies to `user_strategies/temp_strategy.py`
- it executes uploaded Python directly

This is fine for local development, but if the app grows, you may want to:

- isolate runtime session config from global config
- load strategies from a safer plugin mechanism
- separate UI logic from orchestration logic
- centralize benchmark creation in a reusable service function

## Practical Strategy Checklist

Before adding a new strategy, confirm:

- it inherits from `BaseStrategy`
- `train()` does not corrupt shared data
- `generate_signal()` returns stable numeric outputs
- required features exist before use
- missing data is handled safely
- the strategy name is readable in logs and output files

## Practical Detector Checklist

Before adding a new detector, confirm:

- it inherits from `BaseRegimeDetector`
- it has clear safe-vs-risky semantics
- it fits on the expected columns
- row-level calls are fast enough for simulation
- it behaves predictably when asked for unseen dates or missing inputs

## Recommended Development Approach

If you are going to extend this repository significantly, the safest order is:

1. add or improve documentation
2. add tests around current engine behavior
3. introduce one new strategy
4. introduce one new detector
5. only then refactor engine timing or portfolio behavior

That sequence keeps the system understandable while reducing the risk of silently changing benchmark semantics.

## Final Recommendation

Preserve the current architecture's core idea:

- strategy logic and regime logic are separate
- the wrapper composes them
- the engine benchmarks them side by side

That is the strongest part of the codebase and the best foundation for future development.
