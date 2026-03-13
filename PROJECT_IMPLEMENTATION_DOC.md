# Project Implementation Documentation

## Purpose of This Document

This document explains how the current system is implemented at the module, class, method, and runtime-flow level. It is intended as a developer reference for extending the project with additional trading strategies, new regime detection engines, and deeper backtesting features.

Scope of this document:

- Covers the executable Python source in the project root, `components/`, `utils/`, and `user_strategies/`
- Explains how modules interact and what role each method plays
- Highlights implementation assumptions, coupling points, and extension seams
- Excludes the `RDMA/` directory, which contains research and paper artifacts rather than runtime code

## High-Level System Summary

The project is a regime-aware strategy benchmarking framework. Its core idea is simple:

1. Load and preprocess historical market data.
2. Compute features required by strategies and regime detectors.
3. Fit a regime detector to identify calm vs. risky market states.
4. Run a baseline strategy directly on the data.
5. Run the same strategy again through a wrapper that blocks trading in risky regimes.
6. Compare both equity curves, trades, and summary outcomes.

The current implementation supports two primary usage modes:

- A script-based benchmark runner via `main.py`
- An interactive Streamlit UI via `app.py`

The architecture is intentionally lightweight. There is no installed package definition, dependency manager configuration, or formal plugin registry. Instead, the framework relies on a small set of abstract interfaces and straightforward module composition.

## Architecture Overview

At a high level, the system is organized into five layers:

### 1. Configuration Layer

`config.py` defines global settings such as asset ticker, date range, HMM state count, output directories, initial capital, and transaction cost assumptions.

### 2. Contract Layer

`interfaces.py` defines the abstract interfaces that extension modules must follow:

- `BaseStrategy`
- `BaseRegimeDetector`

These interfaces are the main extension seam of the codebase.

### 3. Core Engine Layer

Modules in `components/` perform the actual mechanics of the framework:

- `data_loader.py` fetches and preprocesses market data
- `regime_manager.py` fits and serves regime state predictions
- `strategy_wrapper.py` composes a strategy with a regime detector
- `backtest_engine.py` simulates strategies across time
- `portfolio.py` maintains per-strategy equity and trade state
- `transaction_costs.py` computes execution friction

### 4. Strategy Layer

Modules in `user_strategies/` implement trading logic that conforms to `BaseStrategy`.

### 5. Entry-Point Layer

- `main.py` orchestrates a fixed benchmark run and produces console output plus a chart
- `app.py` provides an interactive Streamlit front end and dynamic loading of user strategy files

## Execution Entry Points

## `main.py`

`main.py` is the main non-UI runner. It performs a complete benchmark using a hardcoded baseline strategy and its regime-aware wrapped version.

Its runtime flow is:

1. Read global settings from `config.py`
2. Load historical data using `fetch_and_process_data()`
3. Create a `VolatilityHMM` detector
4. Create a `MeanReversionStrategy`
5. Wrap the strategy inside `RegimeAwareWrapper`
6. Register both strategies with `BacktestEngine`
7. Run the engine
8. Compute comparative metrics
9. Plot equity curves and regime shading
10. Save the chart to `output/benchmark_chart.png`

This file is best understood as an example orchestrator rather than a reusable library module.

## `app.py`

`app.py` exposes the same conceptual workflow through Streamlit. Instead of hardcoding a strategy class, it allows a user to upload a `.py` file that contains a `BaseStrategy` subclass.

Its runtime flow is:

1. Collect runtime inputs from the Streamlit sidebar
2. Mutate values in `config.py` at runtime for ticker and HMM state count
3. Load data for the requested asset
4. Dynamically import the uploaded strategy file
5. Instantiate the first discovered `BaseStrategy` subclass
6. Create a `VolatilityHMM`
7. Wrap the uploaded strategy with `RegimeAwareWrapper`
8. Register both raw and wrapped strategies with `BacktestEngine`
9. Run the engine
10. Display return metrics and a chart inside the UI

`app.py` is important because it demonstrates the intended user-facing extension workflow: strategy logic is meant to be swappable without changing the engine.

## Core Contracts and Extension Interfaces

## `interfaces.py`

This file is the most important architectural file in the project. It defines the behavioral contracts that the rest of the system expects.

### `BaseStrategy`

`BaseStrategy` is an abstract base class for all trading strategies.

Required methods:

#### `train(self, history: pd.DataFrame)`

Purpose:

- Gives a strategy a chance to prepare itself before simulation starts
- Can be used for feature creation, indicator computation, or machine learning model fitting

Current implementation assumptions:

- The engine calls `train()` once before the simulation loop starts
- The method receives a full historical dataframe
- The strategy may mutate that dataframe in place

This last point is important: the framework currently allows strategies to add columns directly onto the shared market dataframe. That is how `MeanReversionStrategy` adds RSI values.

#### `generate_signal(self, row: pd.Series) -> int`

Purpose:

- Produces the desired position signal for a single row of market data

Signal semantics documented by the interface:

- `1` = long
- `0` = cash / flat
- `-1` = short, if supported

Current engine reality:

- The system primarily behaves as a long/cash framework
- Transaction cost and portfolio logic support numeric changes in signal, but the project is currently designed around `0` and `1`

#### `get_name(self) -> str`

Purpose:

- Returns the class name by default
- Used for logging, output filenames, portfolio labels, and result dataframe column names

This means strategy naming affects output artifacts such as `equity_<strategy>.csv`.

### `BaseRegimeDetector`

`BaseRegimeDetector` is the matching abstraction for market-state classification.

Required methods:

#### `fit(self, data: pd.DataFrame)`

Purpose:

- Train or prepare the regime detector on historical data

#### `detect_regime(self, row: pd.Series) -> int`

Purpose:

- Return the regime classification for a specific row

Documented semantics:

- `0` = safe/calm regime
- `1` = risky/volatile regime

The current implementation assumes regime `0` means trading is allowed and any non-zero regime can be treated as a veto regime by the wrapper logic.

## Configuration and Global Runtime Settings

## `config.py`

`config.py` centralizes both path setup and experiment settings. This module has side effects on import.

### Path definitions

- `BASE_DIR`
- `DATA_DIR`
- `OUTPUT_DIR`
- `LOG_DIR`

These paths are constructed relative to the project root using `pathlib.Path`.

### Import-time side effect

When `config.py` is imported, it immediately ensures the data, output, and log directories exist:

- `data_storage/`
- `output/`
- `logs/`

This means any module importing `config.py` may implicitly create directories on disk.

### Experiment settings

The file also defines:

- `ASSET_TICKER`
- `START_DATE`
- `END_DATE`
- `HMM_STATES`
- `HMM_TRAIN_WINDOW`
- `HMM_REFIT_INTERVAL`
- `INITIAL_CAPITAL`
- `TRANSACTION_COST`
- `RISK_FREE_RATE`

### Important implementation note

Although `config.py` looks static, it is not treated as immutable configuration. In `app.py`, values like `ASSET_TICKER` and `HMM_STATES` are mutated at runtime. That makes `config.py` a global shared state container rather than a strict constant-only settings module.

This design is simple, but it creates coupling between modules and can complicate concurrent or repeated runs.

## Data Ingestion and Feature Engineering

## `components/data_loader.py`

This module is responsible for obtaining historical market data and transforming it into the dataframe structure expected by the rest of the system.

### Main function: `fetch_and_process_data(ticker=config.ASSET_TICKER, force_download=False)`

This is the only public function in the module and is a foundational dependency of both entry points.

### Responsibilities

#### 1. Cache resolution

The function builds a cache path:

- `data_storage/<ticker>_processed.csv`

If that file exists and `force_download` is `False`, the module loads the preprocessed dataframe from disk and returns it immediately.

This means the function caches not raw downloaded data, but already engineered data.

#### 2. Data download

If cache is unavailable or bypassed, the function downloads price data using `yfinance.download()`.

The date range comes from `config.START_DATE` and `config.END_DATE`.

#### 3. Column normalization

Yahoo Finance can return multi-index columns in some cases. The loader checks for this and flattens them by taking the first level.

#### 4. Column selection

Only the following market columns are retained:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

These become the base market dataset used throughout the framework.

#### 5. Feature engineering

The function adds three important derived columns:

##### `Log_Ret`

Defined as:

- `ln(Close_t / Close_t-1)`

This is used by the backtest engine as the daily return input.

##### `Volatility`

Defined as:

- 21-day rolling standard deviation of `Log_Ret`
- Annualized with `sqrt(252)`

This is the critical input feature for the HMM regime detector.

##### `SMA_200`

Defined as:

- 200-day simple moving average of close price

This exists to support trend-following strategies and is used by `TrendFollowingStrategy` in `dummy_strategy.py`.

#### 6. Cleanup

The function drops rows containing `NaN` values, which occur naturally because rolling calculations need a warm-up period.

This means the final dataset starts only after both volatility and `SMA_200` are fully available.

#### 7. Save back to cache

The fully processed dataframe is saved to CSV before being returned.

### Why this module matters architecturally

This module is not just a loader. It is the project's feature-engineering layer. The rest of the code assumes that specific columns exist after this function runs. In other words, it defines the minimum data contract for the entire backtesting system.

If future strategies or detectors require additional features, this module is one place where those can be computed centrally.

## Regime Detection Implementation

## `components/regime_manager.py`

This module implements the default regime detector used by the framework.

### Main class: `VolatilityHMM(BaseRegimeDetector)`

This class uses `hmmlearn.hmm.GaussianHMM` to classify the market into hidden volatility states.

### Constructor: `__init__(self, n_states=config.HMM_STATES, n_iter=100)`

Initialization responsibilities:

- Store the requested number of hidden states
- Create the underlying `GaussianHMM`
- Initialize `state_map`
- Initialize `is_fitted`
- Initialize `regime_history`

### Internal attributes

#### `self.model`

The underlying `GaussianHMM` instance.

#### `self.state_map`

A mapping from raw HMM state IDs to normalized semantic state ranks.

This is necessary because HMM state labels are arbitrary. The raw model might assign low volatility to state `1` in one run and state `0` in another. The code solves that by sorting states by their learned means and remapping them so lower-volatility states become lower-numbered semantic states.

#### `self.regime_history`

A pandas series indexed by date containing the normalized state assignment for each row in the training data.

This is central to the implementation. Instead of making a fresh one-row prediction for each backtest step, the class precomputes the full historical regime sequence once and then performs date-based lookups.

### Method: `fit(self, data: pd.DataFrame)`

This method trains the detector and builds the regime lookup history.

Detailed behavior:

1. Verify that the dataframe contains a `Volatility` column.
2. Extract volatility into a 2D array shaped for `GaussianHMM`.
3. Fit the HMM.
4. Read the learned state means.
5. Sort state IDs by mean volatility.
6. Build `state_map` so the lowest-volatility state becomes semantic state `0`.
7. Predict the full hidden-state sequence across the dataset.
8. Convert raw states through `state_map`.
9. Store the normalized sequence in `self.regime_history` indexed by date.
10. Mark the detector as fitted.

### Method: `detect_regime(self, row: pd.Series) -> int`

This method returns the regime for a single row.

Detailed behavior:

1. Require that the detector has already been fitted.
2. Read the row index, which is assumed to be the date.
3. If the date is present in `regime_history`, return the stored regime directly.
4. If the date is missing, fall back to a one-point model prediction based on the row's `Volatility` value.

### Method: `predict_batch(self, data: pd.DataFrame) -> np.ndarray`

This is a convenience method used mainly for visualization.

Behavior:

- If the detector has not been fitted, it fits it first.
- If the supplied data length matches the cached history length, it returns `regime_history.values`.
- Otherwise it predicts a new sequence from the already fitted model.

### Architectural interpretation

The regime detector is implemented as a precomputed sequence model, not as a live online detector. That is a major design choice.

Advantages:

- Fast lookup during simulation
- Stable and repeatable regime sequence across the run
- Easy plotting and comparison

Trade-off:

- It is benchmark-oriented, not a strict online or walk-forward deployment simulation

## Strategy Composition and Regime Filtering

## `components/strategy_wrapper.py`

This module defines the key composition mechanism that gives the project its central idea.

### Main class: `RegimeAwareWrapper(BaseStrategy)`

This class wraps any normal strategy and modifies its behavior based on the detected market regime.

It behaves like a decorator around a `BaseStrategy` implementation.

### Constructor: `__init__(self, strategy: BaseStrategy, detector: BaseRegimeDetector)`

Responsibilities:

- Store the wrapped strategy
- Store the detector
- Build a derived strategy name such as `RegimeAware(MeanReversionStrategy)`

This generated name is used downstream by the backtest engine for output columns and filenames.

### Method: `train(self, history: pd.DataFrame)`

Detailed behavior:

1. Fit the regime detector on the same historical dataframe.
2. Train the wrapped strategy on that dataframe.

This sequencing matters because both objects may depend on the same shared data.

### Method: `generate_signal(self, row: pd.Series) -> int`

Detailed behavior:

1. Ask the detector for the regime of the current row.
2. Ask the wrapped strategy for its raw signal.
3. If the regime is `0`, return the raw signal unchanged.
4. Otherwise, force the output signal to `0`.

This is the core implementation of regime-aware trading in the codebase.

### Method: `get_name(self) -> str`

Returns the wrapper-specific composite name.

### Architectural significance

This module is what allows the project to benchmark a strategy against its regime-filtered variant without rewriting the strategy itself. It is the cleanest current extension seam in the project.

Any future regime detection engine can plug into this wrapper as long as it implements `BaseRegimeDetector`.

## Backtesting and Simulation Engine

## `components/backtest_engine.py`

This module is the execution hub of the framework.

### Main class: `BacktestEngine`

The engine owns:

- the historical dataframe
- the list of strategy instances
- a `Portfolio` per strategy
- a `TransactionCosts` model

### Constructor: `__init__(self, data: pd.DataFrame)`

Responsibilities:

- Store the market dataframe
- Initialize an empty strategy list
- Initialize an empty portfolio dictionary
- Create a transaction cost model using `config.TRANSACTION_COST`

### Method: `add_strategy(self, strategy_instance)`

Detailed behavior:

1. Read the strategy name from `get_name()`.
2. Append the strategy object to the internal strategy list.
3. Create a new `Portfolio` dedicated to that strategy.
4. Store the portfolio in `self.portfolios` keyed by strategy name.

This means strategies are compared in parallel on the same input data but maintain separate portfolio states.

### Method: `run(self)`

This method performs the actual benchmark.

Detailed behavior is best understood in phases.

#### Phase 1: Training

For each registered strategy, the engine calls:

- `strat.train(self.data)`

Important implication:

- Every strategy is trained on the entire dataset before the backtest loop begins.

This is acceptable for a simple benchmark framework, but it is not equivalent to true walk-forward training.

#### Phase 2: Daily simulation loop

For each row in the dataframe:

1. Read the date and row
2. Pull `Log_Ret` as the daily return value
3. For each strategy:
   - fetch its portfolio
   - compute a new signal via `generate_signal(row)`
   - compute trade cost by comparing `portfolio.prev_signal` to the new signal
   - advance the portfolio with `portfolio.step(...)`
   - record the signal in `signal_history`

#### Phase 3: Results assembly

After the loop, the engine:

1. Starts from a copy of the original data
2. Adds an equity and signal column per strategy
3. Saves each portfolio's equity history to `output/equity_<strategy>.csv`
4. Saves each portfolio's trade history to `output/trades_<strategy>.csv`
5. Returns the enriched result dataframe

### Important timing assumption

The code comments describe a “signal for tomorrow” interpretation, but the actual implementation should be read carefully.

The engine computes a signal from the current row and passes it immediately to `portfolio.step()` along with the current day's return. Inside `Portfolio.step()`, PnL is computed using the new `signal`, not the previous one.

That means the practical behavior is closer to:

- today's row generates today's exposure for today's return

not:

- yesterday's signal earns today's return

This is an important implementation detail to understand before extending or refactoring the engine, because execution timing semantics affect realism and strategy evaluation correctness.

### Architectural role

`BacktestEngine` is where all abstractions converge. It depends on every other core component except the UI. Any major future enhancement such as shorting, leverage, walk-forward training, slippage models, or position sizing will likely require changes here.

## Portfolio State and PnL Tracking

## `components/portfolio.py`

This module defines the state container used by the engine to track performance for each strategy.

### Main class: `Portfolio`

Implemented as a dataclass with mutable runtime state.

### Core fields

- `initial_equity`
- `cash_equity`
- `prev_signal`
- `trade_count`
- `equity_history`
- `trades_history`

### Method: `__post_init__(self)`

If `cash_equity` is not provided, it is initialized to `initial_equity`.

### Method: `step(self, date, signal: int, day_return: float, trade_cost: float)`

This is the central portfolio update method.

Detailed behavior:

1. Detect whether a trade occurred by comparing `signal` to `prev_signal`.
2. Increment trade count if the position changed.
3. Compute PnL as:
   - `signal * day_return - trade_cost`
4. Update equity multiplicatively:
   - `cash_equity = cash_equity * (1 + pnl)`
5. Append an equity snapshot to `equity_history`
6. If a trade occurred, append a trade record to `trades_history`
7. Set `prev_signal` to the new signal
8. Return `(pnl, equity)`

### Implementation implications

This portfolio model is deliberately minimal. It does not separately track:

- units/shares
- cash vs invested capital
- leverage
- margin
- exposure scaling beyond direct signal multiplication
- partial fills
- realistic execution timing

Instead, it assumes a fractional exposure model where signal directly gates return participation.

### Method: `to_equity_df(self)`

Converts accumulated equity snapshots into a date-indexed dataframe.

### Method: `trades_df(self)`

Converts recorded trade events into a date-indexed dataframe.

### Method: `stats(self)`

Returns a basic dictionary of summary metrics:

- final equity
- initial equity
- cumulative return
- total trades
- max drawdown

### Methods: `save_equity(self, path)` and `save_trades(self, path)`

Persist the portfolio histories to CSV.

### Architectural role

The `Portfolio` class is intentionally simple and easy to reason about. It is a good starting point for future development, but also one of the clearest candidates for enhancement if the framework evolves toward more realistic execution logic.

## Transaction Cost Modeling

## `components/transaction_costs.py`

This module isolates trading-friction calculations.

### Main class: `TransactionCosts`

### Constructor: `__init__(self, base_cost_rate=0.0005, slippage_per_trade=0.0000, min_cost=0.0)`

Stores three configuration values:

- `base_cost_rate`
- `slippage_per_trade`
- `min_cost`

### Method: `compute_trade_cost(self, prev_signal: int, new_signal: int, notional: float = 1.0) -> float`

Detailed behavior:

1. Compute position change as `abs(new_signal - prev_signal)`.
2. If change is zero, return zero cost.
3. Compute linear cost from `base_cost_rate * change`.
4. Add slippage per trade.
5. If `min_cost` is configured, enforce it using notional scaling.
6. Return the result as a fractional cost.

### Method: `compute_round_trip_cost(self, n_trades: int, notional: float = 1.0) -> float`

Provides a rough reporting helper based on repeated one-way trade cost.

### Method: `get_config(self) -> dict`

Returns the cost model settings.

### Architectural role

This module is a clean utility abstraction. It keeps the engine from hardcoding friction math and gives future work a natural place to support more advanced cost logic.

## Logging Infrastructure

## `utils/logger.py`

This module provides a shared logger factory used throughout the project.

### Function: `setup_logger(name, log_file=None, level=logging.INFO)`

Responsibilities:

1. Retrieve or create a named logger
2. Set its level
3. If handlers already exist, return immediately to avoid duplication
4. Create a formatted stdout stream handler
5. Optionally create a file handler in the `logs/` directory
6. Return the configured logger

### Important implementation detail

The log path is hardcoded as `Path("logs")` relative to the current working directory. Because the app is normally run from the project root, this works. If execution were moved elsewhere, log file placement could become inconsistent.

### Architectural role

Logging is not sophisticated, but it is consistently used. This improves observability across data loading, engine execution, regime fitting, and strategy training.

## Strategy Modules

## `user_strategies/mean_reversion.py`

This file contains the main example strategy.

### Main class: `MeanReversionStrategy(BaseStrategy)`

This is an RSI-based long/cash strategy.

### Constructor: `__init__(self, period=14, buy_threshold=30, sell_threshold=70)`

Stores RSI settings and derives the RSI column name, such as `RSI_14`.

### Method: `train(self, history: pd.DataFrame)`

Detailed behavior:

1. If the RSI column already exists, return immediately.
2. Compute close-to-close price differences.
3. Compute rolling average gains and losses.
4. Derive relative strength.
5. Convert to RSI values.
6. Store the RSI series on the shared dataframe.
7. Fill missing RSI values with `50`.

### Method: `generate_signal(self, row: pd.Series) -> int`

Detailed behavior:

1. Read the current RSI value from the row.
2. If RSI is below the buy threshold, return `1`.
3. If RSI is above the sell threshold, return `0`.
4. Otherwise use a simplified fallback rule: return `1` if RSI is below `50`, else `0`.

### Architectural interpretation

This strategy is mostly stateless at signal time because position memory is not stored inside the strategy itself. It therefore uses simplified rules for the neutral zone rather than a classic “enter at 30, exit at 70 while holding in between” implementation.

That makes it easy to integrate into the current engine, but also means it is not a perfect representation of how a fully stateful RSI system would be implemented.

## `user_strategies/dummy_strategy.py`

Despite the filename, this is a real second example strategy.

### Main class: `TrendFollowingStrategy(BaseStrategy)`

This strategy uses a simple close-versus-`SMA_200` trend filter.

### Constructor: `__init__(self)`

Stores the required feature column name.

### Method: `train(self, history: pd.DataFrame)`

This method does not calculate features itself. Instead, it validates that `SMA_200` is already present.

This reinforces an important current design pattern: some strategies expect the central data loader to prepare their features for them.

### Method: `generate_signal(self, row: pd.Series) -> int`

Detailed behavior:

1. If `SMA_200` or `Close` is missing for the row, return `0`.
2. If `Close > SMA_200`, return `1`.
3. Otherwise return `0`.

### Architectural interpretation

This is the cleanest example of a strategy that does not mutate the dataframe and simply consumes features prepared earlier in the pipeline.

## `user_strategies/ml_strategy.py`

This file is currently empty.

Its existence suggests an intended future direction for machine-learning-based strategies. When such a strategy is added, `train()` would likely become much more substantial and the current full-history training behavior in `BacktestEngine` would become more important to revisit.

## `user_strategies/temp_strategy.py`

This file is not a core source module. It is a runtime artifact created by `app.py` when a strategy file is uploaded through the Streamlit UI.

It is effectively a staging point for dynamic imports and should not be treated as the authoritative home of any strategy implementation.

## Result and Metric Computation

## `main.py` metric helper

`main.py` defines a helper function:

### `calculate_metrics(equity_curve)`

This function computes:

- total return
- annualized Sharpe ratio using daily percent changes
- max drawdown

It is used only for script reporting and is separate from `Portfolio.stats()`.

This creates a mild duplication of analytics logic in the codebase. Future refactoring could centralize metric calculation into a dedicated analysis module.

## End-to-End Runtime Interaction Flow

The full flow of a standard benchmark run is:

1. An entry point starts the process.
2. `config.py` provides shared settings and ensures core directories exist.
3. `data_loader.fetch_and_process_data()` returns a dataframe with market columns and engineered features.
4. One or more strategy objects are instantiated.
5. A regime detector is instantiated.
6. A `RegimeAwareWrapper` is built around a baseline strategy.
7. `BacktestEngine` is created with the dataframe.
8. Strategies are registered with the engine.
9. The engine calls `train()` on each strategy.
10. During wrapper training, the regime detector is fitted.
11. The engine loops through each row.
12. Each strategy emits a signal for that row.
13. The transaction cost model computes any execution friction.
14. Each strategy's portfolio is updated independently.
15. After simulation, the engine merges all results into a single dataframe.
16. Equity and trade CSV files are written to disk.
17. The entry point computes or displays summary results.

## External Frameworks and Libraries

The project currently depends on the following important libraries.

### `pandas`

Used for:

- historical market dataframes
- rolling calculations
- per-row strategy inputs
- equity/trade history export
- date indexing and alignment

### `numpy`

Used for:

- numerical calculations
- log returns
- square-root annualization
- drawdown computation
- HMM result remapping

### `matplotlib`

Used in both `main.py` and `app.py` for equity-curve visualization and regime shading.

### `streamlit`

Used only by `app.py` to build the interactive benchmark interface.

### `yfinance`

Used by `data_loader.py` as the external market-data source.

### `hmmlearn`

Used by `regime_manager.py` to provide `GaussianHMM`, the framework's current regime engine.

### Python standard library modules

Used across the codebase for:

- abstract base classes via `abc`
- data classes via `dataclasses`
- filesystem paths via `pathlib`
- logging via `logging`
- dynamic imports via `importlib.util`
- module injection via `sys.modules`

## Current Design Strengths

The current implementation has several solid qualities that are worth preserving.

### Clear extension contracts

The use of `BaseStrategy` and `BaseRegimeDetector` makes the core extension model explicit.

### Composable regime filtering

`RegimeAwareWrapper` cleanly separates strategy logic from regime logic.

### Centralized feature engineering

`data_loader.py` provides a single place to define common market features.

### Isolated transaction cost logic

`TransactionCosts` is already separated from the engine.

### Per-strategy portfolio separation

Each strategy gets its own `Portfolio`, making side-by-side comparisons straightforward.

## Current Design Constraints and Important Caveats

These are the key implementation realities a developer should understand before extending the project.

### 1. Full-history training before simulation

Strategies and regime detectors are trained on the full dataset before the backtest loop begins. This is acceptable for internal benchmarking, but it is not a realistic walk-forward setup.

### 2. Shared dataframe mutation

Strategies are allowed to modify the shared training dataframe in place. This is convenient, but it creates possible interactions between strategies when multiple strategies add or overwrite columns.

### 3. Global mutable config

`app.py` mutates module-level values in `config.py`, which creates implicit global coupling.

### 4. Signal timing semantics need tightening

The comments in the engine describe one execution timing model, while the actual portfolio update logic behaves more like same-row exposure. This should be clarified before building more advanced strategies.

### 5. Dynamic user code execution

`app.py` executes uploaded Python code directly. This is useful for experimentation but is not safe for untrusted environments.

### 6. Long/cash bias

Although the strategy interface mentions `-1` as a possible short signal, the current system is primarily built around long/cash behavior.

### 7. No formal dependency packaging

There is no `requirements.txt`, `pyproject.toml`, or other packaging metadata visible in the current implementation. Dependency management will become more important as the project grows.

## How To Add a New Strategy

To add a strategy that integrates cleanly with the current framework:

1. Create a new file in `user_strategies/`.
2. Inherit from `BaseStrategy`.
3. Implement `train(history)`.
4. Implement `generate_signal(row)`.
5. Ensure any required feature columns either already exist from `data_loader.py` or are created safely during `train()`.
6. Return numeric position signals consistently.

### Best practices for new strategies in this codebase

- Prefer deterministic, explicit feature columns
- Avoid overwriting columns used by other strategies
- Keep `generate_signal()` row-based and lightweight
- If statefulness is required, document it clearly and confirm compatibility with the engine loop
- Give the strategy a stable and readable `get_name()` if the class name is not enough

## How To Add a New Regime Detector

To add a new regime engine:

1. Create a class that inherits from `BaseRegimeDetector`.
2. Implement `fit(data)`.
3. Implement `detect_regime(row)`.
4. Decide and document your regime semantics clearly, especially which state means “safe to trade”.
5. Plug the detector into `RegimeAwareWrapper`.

### Recommended design pattern

Follow the same style as `VolatilityHMM`:

- fit once
- cache any sequence outputs needed for backtesting
- make row-level detection simple and fast

If plotting support is needed, add a batch prediction helper similar to `predict_batch()`.

## Suggested Refactor Targets for Future Development

If the framework is going to expand meaningfully, these areas are the highest-value refactor candidates.

### 1. Separate immutable config from runtime session settings

This would reduce global coupling and make app runs more predictable.

### 2. Clarify execution timing in the engine and portfolio

This is critical for strategy correctness.

### 3. Introduce a dedicated analytics module

This would unify metric computation and reduce duplication between `main.py` and `portfolio.py`.

### 4. Move feature engineering into a more formal pipeline

This would help when many strategies require many specialized features.

### 5. Support walk-forward training and out-of-sample testing

This is the biggest methodological upgrade available to the framework.

### 6. Formalize plugin discovery

Instead of dynamically scanning uploaded modules for subclasses, a more explicit registration mechanism could improve safety and clarity.

### 7. Improve portfolio realism

Potential upgrades include explicit holdings, position sizing, short exposure, execution lag, leverage, and better cost accounting.

## Practical Reading Order for New Contributors

If you are trying to understand the project quickly before modifying it, read the code in this order:

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

That order mirrors the conceptual architecture from contracts, to data, to regime logic, to execution, to concrete strategies, and finally to orchestration layers.

## Final Takeaway

The current codebase is a compact and understandable prototype for regime-aware backtesting. Its most important architectural idea is the separation of strategy logic from regime-filtering logic through abstract interfaces and the `RegimeAwareWrapper` composition pattern.

For future development, the most important things to preserve are:

- the clean strategy interface
- the clean regime detector interface
- the wrapper-based comparison model
- the simple engine-to-portfolio flow

The most important things to improve are:

- execution timing correctness
- configuration management
- safer strategy loading
- richer portfolio realism
- walk-forward and out-of-sample methodology

Those changes can be made without discarding the current architecture, which is a strong sign that the project has a good base for further expansion.
