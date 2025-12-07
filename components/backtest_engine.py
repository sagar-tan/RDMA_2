import pandas as pd
import numpy as np
import config
from components.portfolio import Portfolio
from components.transaction_costs import TransactionCosts
from utils.logger import setup_logger

logger = setup_logger("backtest_engine", "engine.log")

class BacktestEngine:
    """
    The Simulator: Runs multiple strategies in parallel on the same data.
    Ensures a fair "Apples-to-Apples" comparison.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.strategies = []
        self.portfolios = {}
        self.tc = TransactionCosts(base_cost_rate=config.TRANSACTION_COST)
        
    def add_strategy(self, strategy_instance):
        """
        Register a strategy to be tested.
        """
        name = strategy_instance.get_name()
        self.strategies.append(strategy_instance)
        
        # Create a dedicated portfolio for this strategy
        self.portfolios[name] = Portfolio(
            initial_equity=config.INITIAL_CAPITAL,
            prev_signal=0
        )
        logger.info(f"Registered Strategy: {name}")

    def run(self):
        """
        The Main Event Loop.
        Iterates through history day-by-day, asking each strategy for a signal.
        """
        logger.info(f"Starting Benchmark on {len(self.data)} data points...")
        
        # 1. Training Phase (if applicable)
        # We define a training window (e.g., first 500 days) or train on the fly.
        # For simplicity in this V1, we assume strategies might need the whole history 
        # or we implement a standard "Warm-up" period.
        
        # Let's verify strategies are trained
        # Note: In a strict walk-forward, this would happen inside the loop.
        # For this "Benchmarking" framework, we usually allow an initial fit.
        for strat in self.strategies:
            strat.train(self.data)

        # 2. Simulation Loop
        # We track signals to save them to the dataframe later
        signal_history = {s.get_name(): [] for s in self.strategies}
        
        for i in range(len(self.data)):
            date = self.data.index[i]
            row = self.data.iloc[i]
            
            # Daily Market Return (Log Return from Data Loader)
            # Note: We assume the signal generated *Yesterday* captures *Today's* return.
            daily_return = row['Log_Ret']
            
            for strat in self.strategies:
                name = strat.get_name()
                portfolio = self.portfolios[name]
                
                # A. Get Signal for TOMORROW (based on today's Close)
                # 1 = Long, 0 = Cash
                signal = strat.generate_signal(row)
                
                # B. Execute Trade
                # We compare Target Signal vs. Previous Signal
                trade_cost = self.tc.compute_trade_cost(
                    portfolio.prev_signal, 
                    signal, 
                    notional=portfolio.cash_equity
                )
                
                # C. Update Portfolio
                # We pass the NEW signal. The Portfolio class handles the PnL logic.
                # Crucial: The PnL for 'date' comes from the position held *coming into* the day.
                # The trade we just executed sets the position for *tomorrow*.
                # (See portfolio.py logic: usually pnl = prev_signal * daily_return - cost)
                
                # Let's ensure your portfolio.py expects this flow.
                # Assuming your standard portfolio.step(date, new_signal, return, cost)
                pnl, equity = portfolio.step(date, signal, daily_return, trade_cost)
                
                # Track signal
                signal_history[name].append(signal)
            
            if i % 250 == 0:
                logger.info(f"Processed {i}/{len(self.data)} days...")

        # 3. Compile Results
        results = self.data.copy()
        
        for name, port in self.portfolios.items():
            # Save Equity Curve to the main dataframe
            equity_df = port.to_equity_df()
            # Align indices carefully
            results[f"{name}_Equity"] = equity_df['Equity']
            results[f"{name}_Signal"] = signal_history[name]
            
            # Save individual files
            port.save_equity(config.OUTPUT_DIR / f"equity_{name}.csv")
            port.save_trades(config.OUTPUT_DIR / f"trades_{name}.csv")
            
        logger.info("Benchmark Complete.")
        return results