import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from components.data_loader import fetch_and_process_data
from components.regime_manager import VolatilityHMM
from components.strategy_wrapper import RegimeAwareWrapper
from components.backtest_engine import BacktestEngine
from user_strategies.mean_reversion import MeanReversionStrategy
from utils.logger import setup_logger

logger = setup_logger("main_runner", "main.log")

def calculate_metrics(equity_curve):
    """Helper to compute key stats for the report."""
    returns = equity_curve.pct_change().fillna(0)
    
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    
    # Sharpe Ratio (Annualized)
    # Assuming Risk Free Rate = 0 for simplicity in comparison
    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return total_return, sharpe, max_dd

def main():
    logger.info("=================================================")
    logger.info("   REGIME-AWARE BENCHMARKING FRAMEWORK v1.0")
    logger.info("=================================================")

    # 1. Load Data
    logger.info(f"Target Asset: {config.ASSET_TICKER}")
    data = fetch_and_process_data()
    
    # 2. Initialize Components
    # The "Weather Station"
    detector = VolatilityHMM(n_states=config.HMM_STATES)
    
    # The "Test Subject" (Baseline)
    baseline_strat = MeanReversionStrategy()
    
    # The "Experiment" (Wrapped Strategy)
    experimental_strat = RegimeAwareWrapper(
        strategy=baseline_strat, 
        detector=detector
    )
    
    # 3. Setup The Engine
    engine = BacktestEngine(data)
    engine.add_strategy(baseline_strat)
    engine.add_strategy(experimental_strat)
    
    # 4. Run Benchmark
    results = engine.run()
    
    # 5. Analysis & Reporting
    print("\n" + "="*60)
    print(f"BENCHMARK RESULTS: {config.ASSET_TICKER} ({config.START_DATE} to {config.END_DATE})")
    print("="*60)
    print(f"{'Strategy':<25} | {'Total Ret':<10} | {'Sharpe':<8} | {'Max DD':<10}")
    print("-" * 60)
    
    # Baseline Metrics
    base_name = baseline_strat.get_name()
    base_eq = results[f"{base_name}_Equity"]
    b_ret, b_sharpe, b_dd = calculate_metrics(base_eq)
    print(f"{base_name:<25} | {b_ret*100:>8.2f}% | {b_sharpe:>8.2f} | {b_dd*100:>8.2f}%")
    
    # Experimental Metrics
    exp_name = experimental_strat.get_name()
    exp_eq = results[f"{exp_name}_Equity"]
    e_ret, e_sharpe, e_dd = calculate_metrics(exp_eq)
    print(f"{exp_name:<25} | {e_ret*100:>8.2f}% | {e_sharpe:>8.2f} | {e_dd*100:>8.2f}%")
    print("="*60)
    
    # Calculate Improvement (Delta)
    dd_improvement = b_dd - e_dd # e.g. -0.50 - (-0.20) = -0.30 (Bad) -> Wait, logic:
    # Improvement is (Old_DD - New_DD) ? No, we want to show reduction.
    # Relative improvement
    print("\n>>> REGIME IMPACT ANALYSIS")
    if e_dd > b_dd:
        print(f"✅ Max Drawdown reduced from {b_dd*100:.1f}% to {e_dd*100:.1f}%")
    else:
        print(f"❌ Max Drawdown worsened.")
        
    if e_sharpe > b_sharpe:
        print(f"✅ Risk-Adjusted Return (Sharpe) improved by {((e_sharpe/b_sharpe)-1)*100:.1f}%")
    else:
        print(f"❌ Sharpe Ratio decreased.")

    # 6. Visualization
    plt.figure(figsize=(14, 7))
    
    # Plot Equities
    plt.plot(results.index, base_eq, label=f"Baseline: {base_name}", color='gray', alpha=0.7, linestyle='--')
    plt.plot(results.index, exp_eq, label=f"Regime-Aware: {exp_name}", color='#1f77b4', linewidth=2)
    
    # Plot Regime Overlay (Background Shading)
    # We need to re-run the detector on the full data to get the plotting overlay
    # (The wrapper ran it internally, but we want the array for plotting)
    # This is quick since it's already fitted.
    regimes = detector.predict_batch(data)
    
    # Create scale for shading
    y_min, y_max = plt.ylim()
    
    # Fill "Risky" areas (Regime 1)
    plt.fill_between(results.index, y_min, y_max, 
                     where=(regimes == 1), 
                     color='red', alpha=0.1, label='High Volatility Regime')

    plt.title(f"Regime-Awareness Benchmark: {base_name}", fontsize=14)
    plt.ylabel("Portfolio Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save
    save_path = config.OUTPUT_DIR / "benchmark_chart.png"
    plt.savefig(save_path)
    logger.info(f"Chart saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()