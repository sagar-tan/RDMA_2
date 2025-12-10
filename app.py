import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
import sys
from pathlib import Path

# Import your framework components
import config
from components.data_loader import fetch_and_process_data
from components.regime_manager import VolatilityHMM
from components.strategy_wrapper import RegimeAwareWrapper
from components.backtest_engine import BacktestEngine
from interfaces import BaseStrategy

# --- PAGE CONFIG ---
st.set_page_config(page_title="RegimeAlpha Lab", layout="wide")

st.title("🛡️ Regime-Aware Strategy Benchmark")
st.markdown("""
**Hypothesis:** Most strategies fail because they trade during the wrong market regime.
**Experiment:** Upload your strategy class. We will run it **Raw** vs. **Regime-Filtered**.
""")

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Asset Ticker", value="SPY")
hmm_states = st.sidebar.slider("HMM States", 2, 4, 2)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-01"))

# --- HELPER: DYNAMIC IMPORT ---
def load_strategy_from_file(uploaded_file):
    """
    Magically imports a class from an uploaded Python file.
    """
    try:
        # Save temp file
        file_path = Path("user_strategies") / "temp_strategy.py"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Dynamic Import logic
        spec = importlib.util.spec_from_file_location("temp_module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_module"] = module
        spec.loader.exec_module(module)
        
        # Find the class that inherits from BaseStrategy
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                return obj() # Return an instance
        return None
    except Exception as e:
        st.error(f"Error loading strategy: {e}")
        return None

# --- MAIN LOGIC ---
uploaded_file = st.sidebar.file_uploader("Upload Strategy (.py)", type=["py"])

if st.sidebar.button("Run Benchmark"):
    if not uploaded_file:
        st.warning("Please upload a strategy file first! (Check 'user_strategies/mean_reversion.py' for a template)")
    else:
        with st.spinner(f"Fetching Data for {ticker} and Training HMM..."):
            # 1. Update Config (Runtime override)
            config.ASSET_TICKER = ticker
            config.HMM_STATES = hmm_states
            
            # 2. Load Data
            data = fetch_and_process_data(ticker)
            data = data[data.index >= pd.to_datetime(start_date)]
            
            # 3. Load Strategy
            baseline_strat = load_strategy_from_file(uploaded_file)
            
            if baseline_strat:
                st.success(f"Loaded Strategy: {baseline_strat.get_name()}")
                
                # 4. Setup Engine
                detector = VolatilityHMM(n_states=hmm_states)
                wrapper = RegimeAwareWrapper(baseline_strat, detector)
                
                engine = BacktestEngine(data)
                engine.add_strategy(baseline_strat)
                engine.add_strategy(wrapper)
                
                # 5. Run
                results = engine.run()
                
                # 6. Visualize
                base_name = baseline_strat.get_name()
                wrap_name = wrapper.get_name()
                
                base_eq = results[f"{base_name}_Equity"]
                wrap_eq = results[f"{wrap_name}_Equity"]
                
                # Metrics
                total_ret_base = (base_eq.iloc[-1] - base_eq.iloc[0]) / base_eq.iloc[0]
                total_ret_wrap = (wrap_eq.iloc[-1] - wrap_eq.iloc[0]) / wrap_eq.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Baseline Return", f"{total_ret_base*100:.1f}%")
                col2.metric("Regime-Aware Return", f"{total_ret_wrap*100:.1f}%", 
                            delta=f"{(total_ret_wrap - total_ret_base)*100:.1f}%")
                
                # Chart
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(results.index, base_eq, label="Baseline", color="gray", alpha=0.5)
                ax.plot(results.index, wrap_eq, label="Regime-Aware", color="blue")
                
                # Shade Regimes
                # Need to run prediction for plotting explicitly
                regimes = detector.predict_batch(data)
                y_min, y_max = ax.get_ylim()
                ax.fill_between(results.index, y_min, y_max, where=(regimes==1), color='red', alpha=0.1, label="High Volatility")
                
                ax.set_title("Equity Curve Comparison")
                ax.legend()
                st.pyplot(fig)
                
                st.write("### Data View")
                st.dataframe(results.tail())
                
            else:
                st.error("Could not find a valid strategy class in the file. Make sure it inherits from BaseStrategy.")