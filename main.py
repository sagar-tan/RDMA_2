# main.py
import config
from components.data_loader import fetch_and_process_data
from components.backtester import run_backtest
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # 1. Load Data
    df = fetch_and_process_data(config.ASSET_TICKER)
    
    # 2. Run Strategy
    # The backtester now returns the DF with 'Equity' and 'Signal' columns attached
    results = run_backtest(df)
    
    # 3. Stats & Plotting
    final_strat = results['Equity'].iloc[-1]
    
    # Calculate Buy & Hold Equity manually for comparison
    results['BaH_Return'] = results['Log_Ret']
    results['BaH_Equity'] = (1 + results['BaH_Return']).cumprod()
    final_bah = results['BaH_Equity'].iloc[-1]
    
    print("\n" + "="*40)
    print(f"FINAL RESULTS ({config.ASSET_TICKER})")
    print("="*40)
    print(f"Buy & Hold Return: {(final_bah - 1)*100:.2f}%")
    print(f"Regime Strategy:   {(final_strat - 1)*100:.2f}%")
    print("="*40)
    
    # 4. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['BaH_Equity'], label='Buy & Hold', alpha=0.6, color='grey')
    plt.plot(results.index, results['Equity'], label='Vol Regime Switch', linewidth=2, color='blue')
    
    # Shade the High Volatility Regimes (Regime 1)
    y_min, y_max = plt.ylim()
    plt.fill_between(results.index, y_min, y_max, where=(results['Regime']==1), color='red', alpha=0.1, label='High Vol Regime')
    
    plt.title(f"Regime-Based Risk Control: {config.ASSET_TICKER}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = config.OUTPUT_DIR / "final_result.png"
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()