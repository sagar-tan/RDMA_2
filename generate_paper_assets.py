"""
Generate statistical tests and figures for the research paper.
Runs all strategies across all assets, computes paired t-tests on daily returns
(Raw vs Regime-Aware), and generates publication-quality figures.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from components.data_loader import fetch_and_process_data
from components.backtest_engine import BacktestEngine
from components.regime_detectors import VolatilityHMM, VolatilityThresholdDetector, DrawdownRegimeDetector
from components.strategy_wrapper import RegimeAwareWrapper
from user_strategies.mean_reversion import MeanReversionStrategy
from user_strategies.dummy_strategy import TrendFollowingStrategy
from user_strategies.momentum import MomentumStrategy
from user_strategies.breakout import BreakoutStrategy
from user_strategies.ma_cross import MovingAverageCrossStrategy
from user_strategies.volatility_breakout import VolatilityBreakoutStrategy

ASSETS = ["SPY", "GLD", "BTC-USD", "EURUSD=X"]
STRATEGIES = [
    ("Mean Reversion", MeanReversionStrategy),
    ("Trend Following", TrendFollowingStrategy),
    ("Momentum", MomentumStrategy),
    ("Breakout", BreakoutStrategy),
    ("MA Cross", MovingAverageCrossStrategy),
    ("Vol Breakout", VolatilityBreakoutStrategy),
]

def run_paired_test(asset, strat_name, strat_class):
    """Run raw vs HMM regime-aware and compute paired t-test on daily returns."""
    data = fetch_and_process_data(ticker=asset)
    
    baseline = strat_class()
    detector = VolatilityHMM(n_states=2)
    wrapped = RegimeAwareWrapper(strat_class(), detector)
    
    engine = BacktestEngine(data.copy())
    engine.add_strategy(baseline)
    engine.add_strategy(wrapped)
    results = engine.run()
    
    base_name = baseline.get_name()
    wrap_name = wrapped.get_name()
    
    base_eq = results[f"{base_name}_Equity"].values
    wrap_eq = results[f"{wrap_name}_Equity"].values
    
    # Compute daily returns from equity curves
    base_daily = np.diff(base_eq) / base_eq[:-1]
    wrap_daily = np.diff(wrap_eq) / wrap_eq[:-1]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(wrap_daily, base_daily)
    
    # Sharpe ratios
    base_sharpe = (np.mean(base_daily) / (np.std(base_daily) + 1e-12)) * np.sqrt(252)
    wrap_sharpe = (np.mean(wrap_daily) / (np.std(wrap_daily) + 1e-12)) * np.sqrt(252)
    
    # Max drawdowns
    base_hwm = np.maximum.accumulate(base_eq)
    base_dd = np.min((base_eq - base_hwm) / (base_hwm + 1e-12))
    wrap_hwm = np.maximum.accumulate(wrap_eq)
    wrap_dd = np.min((wrap_eq - wrap_hwm) / (wrap_hwm + 1e-12))
    
    # Total returns
    base_ret = (base_eq[-1] / base_eq[0]) - 1
    wrap_ret = (wrap_eq[-1] / wrap_eq[0]) - 1
    
    return {
        "Strategy": strat_name,
        "Asset": asset,
        "Raw Sharpe": round(base_sharpe, 2),
        "Regime Sharpe": round(wrap_sharpe, 2),
        "Raw Return": round(base_ret * 100, 1),
        "Regime Return": round(wrap_ret * 100, 1),
        "Raw MaxDD": round(base_dd * 100, 1),
        "Regime MaxDD": round(wrap_dd * 100, 1),
        "t-stat": round(t_stat, 3),
        "p-value": round(p_value, 4),
    }


def generate_sharpe_comparison_chart(results_df):
    """Generate a grouped bar chart comparing Raw vs Regime-Aware Sharpe across strategies on SPY."""
    spy_data = results_df[results_df["Asset"] == "SPY"]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(spy_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, spy_data["Raw Sharpe"], width, label="Raw Strategy", 
                   color="#7f8c8d", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, spy_data["Regime Sharpe"], width, label="Regime-Aware (HMM)", 
                   color="#2980b9", edgecolor="white", linewidth=0.5)
    
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Risk-Adjusted Return: Raw vs Regime-Aware (SPY)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(spy_data["Strategy"], fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "sharpe_comparison_spy.png", dpi=200)
    plt.close()
    print("Saved: sharpe_comparison_spy.png")


def generate_drawdown_reduction_chart(results_df):
    """Generate a grouped bar chart showing drawdown reduction across assets for selected strategies."""
    # Pick representative strategies across assets
    selected = results_df[results_df["Strategy"].isin(["Mean Reversion", "Trend Following", "MA Cross"])]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r['Strategy']}\n({r['Asset']})" for _, r in selected.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, selected["Raw MaxDD"], width, label="Raw Strategy", 
                   color="#c0392b", alpha=0.7, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, selected["Regime MaxDD"], width, label="Regime-Aware (HMM)", 
                   color="#27ae60", alpha=0.7, edgecolor="white", linewidth=0.5)
    
    ax.set_ylabel("Maximum Drawdown (%)", fontsize=11)
    ax.set_title("Tail Risk Reduction: Raw vs Regime-Aware", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "drawdown_reduction.png", dpi=200)
    plt.close()
    print("Saved: drawdown_reduction.png")


if __name__ == "__main__":
    all_results = []
    
    for asset in ASSETS:
        for strat_name, strat_class in STRATEGIES:
            print(f"Testing: {strat_name} on {asset}...")
            row = run_paired_test(asset, strat_name, strat_class)
            all_results.append(row)
            print(f"  Raw Sharpe={row['Raw Sharpe']}, Regime Sharpe={row['Regime Sharpe']}, p={row['p-value']}")
    
    df = pd.DataFrame(all_results)
    df.to_csv(config.OUTPUT_DIR / "statistical_tests.csv", index=False)
    
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION RESULTS (Paired t-test: Regime-Aware vs Raw)")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nSaved to: {config.OUTPUT_DIR / 'statistical_tests.csv'}")
    
    # Generate figures
    generate_sharpe_comparison_chart(df)
    generate_drawdown_reduction_chart(df)
