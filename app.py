import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
import logging
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger("app_ui", "ui.log")

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
    Dynamically load a strategy from an uploaded .py file.
    Ensures fresh load by clearing sys.modules cache.
    """
    if uploaded_file is not None:
        # Save to a fixed temporary file with absolute path
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(abs_dir, "user_strategies")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_strategy.py")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Clear from sys.modules to force a fresh import
        module_name = "user_strategies.temp_strategy"
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            # Dynamic Import logic
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the class that inherits from BaseStrategy
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj != BaseStrategy:
                    return obj()
        except Exception as e:
            st.error(f"Failed to load strategy: {e}")
            logger.error(f"Strategy load error: {e}")
            
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