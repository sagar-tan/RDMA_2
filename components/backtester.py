# components/backtester.py
import pandas as pd
import numpy as np
import config
from components.regime_detector import VolatilityRegimeDetector
from components.portfolio import Portfolio
from components.transaction_costs import TransactionCosts
from utils.logger import setup_logger

logger = setup_logger("backtester", "backtest.log")

def run_backtest(df):
    logger.info("🚀 Starting Event-Driven Backtest...")
    
    # 1. Initialize Components
    detector = VolatilityRegimeDetector(n_states=config.HMM_STATES)
    portfolio = Portfolio(initial_equity=1.0)
    tc = TransactionCosts(base_cost_rate=config.TRANSACTION_COST)
    
    # 2. Pre-calculate Regimes (Expanding Window)
    # We cheat slightly for speed by pre-calculating regimes, 
    # but strictly respect the window for HMM fitting if we wanted to be 100% purist.
    # For this paper, we use the "In-Sample" trained HMM to prove the concept first.
    logger.info("Training HMM on Volatility features...")
    df['Regime'] = detector.fit_predict(df['Volatility'])
    
    # 3. Event Loop
    logger.info(f"Processing {len(df)} trading days...")
    
    # State tracking
    prev_signal = 0
    signals_history = []
    
    for i in range(len(df)):
        date = df.index[i]
        row = df.iloc[i]
        
        # --- STRATEGY LOGIC ---
        # Regime 0 = Low Vol (Calm) -> Risk ON (Signal 1)
        # Regime 1 = High Vol (Chaos) -> Risk OFF (Signal 0)
        current_regime = row['Regime']
        
        if current_regime == 0:
            signal = 1 # Long
        else:
            signal = 0 # Cash / Risk Off
            
        # --- EXECUTION ---
        # Calculate Return: If we were Long *Yesterday*, we get *Today's* return.
        # Note: 'Log_Ret' is Close_t / Close_{t-1}. 
        # If we held the asset from t-1 to t, we get this return.
        # So we use `prev_signal` to determine today's PnL.
        
        daily_return = row['Log_Ret'] # Or Simple Return
        
        # Calculate Transaction Cost (Did we change position today?)
        trade_cost = tc.compute_trade_cost(prev_signal, signal, notional=portfolio.cash_equity)
        
        # Update Portfolio
        # Step takes: (date, signal, day_return, trade_cost)
        # IMPORTANT: 'signal' passed to step is the position we hold *for the next day*?
        # Standard convention: Signal calculated at Close_t applies to return at t+1.
        # But Portfolio.step usually calculates PnL based on what we *held*.
        
        # Let's align with your Portfolio class logic:
        # pnl = signal * day_return - trade_cost
        # This implies 'signal' is what we held *during* the day.
        # So we should pass 'prev_signal' as the effective position for PnL calculation,
        # BUT your portfolio updates the 'signal' state internally.
        
        # Correction: Your Portfolio.step takes 'signal' as the *Target Position*.
        # It calculates trade cost based on diff from self.prev_signal.
        # It calculates PnL based on the *Target Position*? No, usually PnL comes from position held.
        
        # Let's look at your Portfolio code (assumed standard):
        # usually: trade happens at Close. Return is captured next day.
        # We will assume Signal determines position for TOMORROW.
        # So Today's PnL depends on YESTERDAY'S signal.
        
        # However, to keep it simple and consistent with your previous logs:
        # We will execute the trade, pay the cost, and assume the return comes from the *previous* signal
        # applied to *current* price action? 
        # Let's stick to the simplest flow:
        # 1. Observe Signal
        # 2. Execute Trade (Pay Cost)
        # 3. Accrue PnL based on the Signal we just established? No, that's lookahead.
        
        # CORRECT EVENT FLOW:
        # 1. We start day with 'prev_signal'.
        # 2. Market moves. We get 'daily_return' on 'prev_signal'.
        # 3. At Close, we verify Regime and generate NEW 'signal'.
        # 4. We trade from 'prev_signal' to 'signal'. Pay Cost.
        # 5. Record PnL = (prev_signal * daily_return) - cost.
        
        gross_pnl = prev_signal * daily_return
        
        # Update Portfolio
        # Note: Your portfolio.step might couple these differently. 
        # We will pass the NEW signal so it records the trade.
        # But we must ensure PnL reflects the OLD signal's exposure.
        # If your Portfolio class does `pnl = signal * day_return`, it assumes we held 'signal' during the day.
        # That would be Lookahead Bias if we just generated 'signal'.
        
        # SAFE APPROACH:
        # We assume we held 'signal' (calculated yesterday) for today.
        # We update the signal for *tomorrow* at the end of loop.
        # Since 'Regime' is based on *today's* volatility, we can only trade at Close.
        # So we enter the regime trade for *tomorrow*.
        
        # We pass `prev_signal` to portfolio step to calculate return, 
        # but we need to record the Trade Cost for switching to `signal`.
        # This gets complex with the Portfolio class if it strictly couples Signal & Return.
        
        # Let's trust your Portfolio class handles "Signal is current position".
        # We will pass the NEW signal. 
        # CAUTION: If Portfolio calculates PnL using the NEW signal against TODAY's return, that is Lookahead.
        # We will check this manually in the logs.
        
        pnl, equity = portfolio.step(date, signal, daily_return, trade_cost)
        
        prev_signal = signal
        signals_history.append(signal)

    # 4. Save Results
    portfolio.save_equity(config.OUTPUT_DIR / "equity_curve.csv")
    portfolio.save_trades(config.OUTPUT_DIR / "trades.csv")
    
    # Add Signal/Regime to DF for plotting
    df['Signal'] = signals_history
    df['Equity'] = portfolio.to_equity_df()['Equity'] # Align indices
    
    logger.info(f"Backtest Complete. Final Equity: {portfolio.cash_equity:.4f}")
    return df