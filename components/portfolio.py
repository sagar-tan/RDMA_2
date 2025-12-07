import logging
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger("portfolio")


@dataclass
class Portfolio:
    """
    Minimal portfolio object to keep portfolio state and compute PnL.
    Designed to work with your walk_forward_engine which uses daily fractional returns.
    """
    initial_equity: float = 1.0
    cash_equity: float = field(default=None)
    prev_signal: int = 0
    trade_count: int = 0
    equity_history: list = field(default_factory=list)  # list of dicts with Date and Equity
    trades_history: list = field(default_factory=list)  # list of trades dicts

    def __post_init__(self):
        if self.cash_equity is None:
            self.cash_equity = float(self.initial_equity)
        logger.info(f"Portfolio initialized with equity {self.cash_equity}")

    def step(self, date, signal: int, day_return: float, trade_cost: float):
        """
        Advance portfolio by one day.
        date : pd.Timestamp or str
        signal : current position (0 or 1)
        day_return : asset return for the day as fraction, e.g. 0.01 for +1%
        trade_cost : transaction cost expressed as fraction of equity to be subtracted on trade
        """

        # count trade if position changed
        if signal != self.prev_signal:
            self.trade_count += 1
            trade_happened = True
        else:
            trade_happened = False

        # apply pnl: position exposure * return - trade cost when a trade occurred
        pnl = signal * day_return - trade_cost

        # update equity
        old_equity = self.cash_equity
        self.cash_equity = self.cash_equity * (1.0 + pnl)

        # record
        self.equity_history.append({
            "Date": pd.to_datetime(date),
            "Equity": float(self.cash_equity)
        })

        if trade_happened:
            self.trades_history.append({
                "Date": pd.to_datetime(date),
                "prev_signal": int(self.prev_signal),
                "new_signal": int(signal),
                "trade_cost": float(trade_cost),
                "equity_before": float(old_equity),
                "equity_after": float(self.cash_equity)
            })

        self.prev_signal = int(signal)

        logger.debug(f"Portfolio step {date}, sig={signal}, day_ret={day_return}, cost={trade_cost}, pnl={pnl}, equity={self.cash_equity}")
        return pnl, float(self.cash_equity)

    def to_equity_df(self):
        if len(self.equity_history) == 0:
            return pd.DataFrame(columns=["Date", "Equity"]).set_index("Date")
        df = pd.DataFrame(self.equity_history).set_index("Date").sort_index()
        return df

    def trades_df(self):
        if len(self.trades_history) == 0:
            return pd.DataFrame(columns=["Date", "prev_signal", "new_signal", "trade_cost", "equity_before", "equity_after"]).set_index("Date")
        df = pd.DataFrame(self.trades_history).set_index("Date").sort_index()
        return df

    def stats(self):
        """
        Basic portfolio stats useful for quick checks.
        Returns dict with final equity, total trades, cumulative return, max drawdown.
        """
        equities = np.array([e["Equity"] for e in self.equity_history], dtype=float) if self.equity_history else np.array([self.cash_equity])
        cumulative_return = (equities[-1] / equities[0]) - 1.0 if len(equities) > 0 else 0.0

        # drawdown
        hwm = np.maximum.accumulate(equities)
        drawdowns = (equities - hwm) / (hwm + 1e-12)
        max_dd = float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0

        return {
            "final_equity": float(self.cash_equity),
            "initial_equity": float(self.initial_equity),
            "cumulative_return": float(cumulative_return),
            "total_trades": int(self.trade_count),
            "max_drawdown": float(max_dd)
        }

    def save_equity(self, path):
        df = self.to_equity_df()
        df.to_csv(path, index=True)
        logger.info(f"Saved equity to {path}")

    def save_trades(self, path):
        df = self.trades_df()
        df.to_csv(path, index=True)
        logger.info(f"Saved trades to {path}")
