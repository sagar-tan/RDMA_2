# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import cast

import config
from components.backtest_engine import BacktestEngine
from components.data_loader import fetch_and_process_data
from components.regime_detectors import (
    DrawdownRegimeDetector,
    VolatilityHMM,
    VolatilityThresholdDetector,
)
from components.strategy_wrapper import RegimeAwareWrapper
from user_strategies.breakout import BreakoutStrategy
from user_strategies.ma_cross import MovingAverageCrossStrategy
from user_strategies.mean_reversion import MeanReversionStrategy
from user_strategies.momentum import MomentumStrategy
from user_strategies.trend_following import TrendFollowingStrategy
from user_strategies.volatility_breakout import VolatilityBreakoutStrategy
from utils.logger import setup_logger

logger = setup_logger("benchmark_suite", "benchmark_suite.log")

ASSETS = ["SPY", "GLD", "BTC-USD", "EURUSD=X"]


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    factory: Callable[[], Any]


STRATEGIES = [
    ExperimentDefinition("MeanReversionStrategy", MeanReversionStrategy),
    ExperimentDefinition("TrendFollowingStrategy", TrendFollowingStrategy),
    ExperimentDefinition("MomentumStrategy", MomentumStrategy),
    ExperimentDefinition("BreakoutStrategy", BreakoutStrategy),
    ExperimentDefinition("MovingAverageCrossStrategy", MovingAverageCrossStrategy),
    ExperimentDefinition("VolatilityBreakoutStrategy", VolatilityBreakoutStrategy),
]

DETECTORS = [
    ExperimentDefinition("None", lambda: None),
    ExperimentDefinition("VolatilityHMM", lambda: VolatilityHMM(n_states=config.HMM_STATES)),
    ExperimentDefinition("VolatilityThresholdDetector", VolatilityThresholdDetector),
    ExperimentDefinition("DrawdownRegimeDetector", DrawdownRegimeDetector),
]


def calculate_performance_metrics(equity_curve: pd.Series) -> tuple[float, float, float]:
    returns = equity_curve.pct_change().fillna(0.0)
    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)
    sharpe = 0.0 if returns.std() == 0 else float((returns.mean() / returns.std()) * np.sqrt(252))
    drawdown = (equity_curve / equity_curve.cummax()) - 1.0
    max_drawdown = float(drawdown.min())
    return total_return, sharpe, max_drawdown


def calculate_strategy_metrics(results: pd.DataFrame, strategy_name: str) -> tuple[int, float, float]:
    signal_col = f"{strategy_name}_Signal"
    signals = results[signal_col].fillna(0).astype(int)
    signal_values = signals.to_numpy(dtype=int)
    exposure_ratio = float((signals != 0).mean())
    trade_count = int(np.count_nonzero(np.diff(signal_values) != 0) + (signal_values[0] != 0))

    active_returns = results.loc[signals != 0, "Log_Ret"]
    win_rate = float((active_returns > 0).mean()) if not active_returns.empty else 0.0
    return trade_count, exposure_ratio, win_rate


def run_experiment(asset: str, strategy_def: ExperimentDefinition, detector_def: ExperimentDefinition) -> list[dict]:
    logger.info("Running %s on %s with detector %s", strategy_def.name, asset, detector_def.name)
    data = fetch_and_process_data(ticker=asset)

    baseline_strategy = strategy_def.factory()
    engine = BacktestEngine(data.copy())
    engine.add_strategy(baseline_strategy)

    wrapped_strategy = None
    detector = None
    if detector_def.name != "None":
        detector = detector_def.factory()
        wrapped_strategy = RegimeAwareWrapper(strategy_def.factory(), detector)
        engine.add_strategy(wrapped_strategy)

    results = engine.run()

    metric_rows = []
    for strategy_instance in [baseline_strategy, wrapped_strategy]:
        if strategy_instance is None:
            continue

        strategy_name = strategy_instance.get_name()
        equity_curve = cast(pd.Series, results[f"{strategy_name}_Equity"])
        total_return, sharpe_ratio, max_drawdown = calculate_performance_metrics(equity_curve)
        trade_count, exposure_ratio, win_rate = calculate_strategy_metrics(results, strategy_name)

        metric_rows.append(
            {
                "Asset": asset,
                "BaseStrategy": strategy_def.name,
                "ExecutedStrategy": strategy_name,
                "Detector": detector_def.name,
                "IsRegimeFiltered": detector_def.name != "None" and strategy_name.startswith("RegimeAware("),
                "TotalReturn": total_return,
                "SharpeRatio": sharpe_ratio,
                "MaxDrawdown": max_drawdown,
                "TradeCount": trade_count,
                "ExposureRatio": exposure_ratio,
                "WinRate": win_rate,
            }
        )

    if detector is not None:
        save_regime_overlay_chart(data, detector, asset, strategy_def.name, detector_def.name)

    return metric_rows


def save_benchmark_chart(results_df: pd.DataFrame):
    if results_df.empty:
        return

    chart_df = results_df.copy()
    chart_df["Label"] = chart_df["Asset"] + " | " + chart_df["ExecutedStrategy"]

    plt.figure(figsize=(16, 8))
    colors = ["#1f77b4" if value else "#7f8c8d" for value in chart_df["IsRegimeFiltered"]]
    plt.bar(chart_df["Label"], chart_df["SharpeRatio"], color=colors)
    plt.xticks(rotation=90)
    plt.ylabel("Sharpe Ratio")
    plt.title("Expanded Regime-Aware Benchmark Results")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "benchmark_chart.png")
    plt.close()


def save_regime_overlay_chart(
    data: pd.DataFrame,
    detector,
    asset: str,
    strategy_name: str,
    detector_name: str,
):
    if not hasattr(detector, "predict_batch"):
        detector.fit(data)
        regimes = data.apply(detector.detect_regime, axis=1).to_numpy(dtype=int)
    else:
        regimes = detector.predict_batch(data)

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data["Close"], color="#1f77b4", linewidth=1.5, label=asset)
    y_min, y_max = plt.ylim()
    plt.fill_between(data.index, y_min, y_max, where=(regimes == 1), color="#c0392b", alpha=0.12, label=detector_name)
    plt.title(f"Regime Overlay: {asset} | {strategy_name}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "regime_overlay_chart.png")
    plt.close()


def main():
    all_rows: list[dict] = []

    for asset in ASSETS:
        for strategy_def in STRATEGIES:
            for detector_def in DETECTORS:
                all_rows.extend(run_experiment(asset, strategy_def, detector_def))

    results_df = pd.DataFrame(all_rows)
    results_path = config.OUTPUT_DIR / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    save_benchmark_chart(results_df)

    print(results_df.to_string(index=False))
    print(f"\nSaved benchmark results to {results_path}")


if __name__ == "__main__":
    main()
