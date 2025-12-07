import logging
import math

logger = logging.getLogger("transaction_costs")


class TransactionCosts:
    """
    Small utility for computing trade costs, slippage and spreads.
    API is intentionally simple so it's easy to call from walk_forward_engine.
    """

    def __init__(self, base_cost_rate=0.0005, slippage_per_trade=0.0000, min_cost=0.0):
        """
        base_cost_rate : fraction charged per trade when a position changes, e.g. 0.0005 = 5 bps
        slippage_per_trade : extra slippage fraction applied on each trade (optional)
        min_cost : minimum absolute cost per trade (useful for tiny trades)
        """
        self.base_cost_rate = float(base_cost_rate)
        self.slippage_per_trade = float(slippage_per_trade)
        self.min_cost = float(min_cost)

    def compute_trade_cost(self, prev_signal: int, new_signal: int, notional: float = 1.0) -> float:
        """
        Compute cost for changing position from prev_signal to new_signal.
        Signals expected to be 0 or 1 (or any numeric position). We use absolute change.
        notional : current portfolio notional used to scale fixed-cost-like behaviour. Default 1.0 (fractional).
        Returns cost as a fraction of portfolio notional (same scale as transaction_cost in your engine).
        """
        change = abs(new_signal - prev_signal)
        if change == 0:
            return 0.0

        # base linear cost
        cost = self.base_cost_rate * change

        # add slippage per trade
        cost += self.slippage_per_trade * change

        # scale minimum cost by notional if configured
        if self.min_cost > 0:
            cost = max(cost, self.min_cost / (notional if notional > 0 else 1.0))

        logger.debug(f"Trade cost computed, prev={prev_signal}, new={new_signal}, change={change}, cost={cost}")
        return float(cost)

    def compute_round_trip_cost(self, n_trades: int, notional: float = 1.0) -> float:
        """
        Useful helper to estimate round-trip costs for reporting.
        """
        if n_trades <= 0:
            return 0.0
        return float(n_trades) * self.compute_trade_cost(0, 1, notional)

    def get_config(self) -> dict:
        return {
            "base_cost_rate": self.base_cost_rate,
            "slippage_per_trade": self.slippage_per_trade,
            "min_cost": self.min_cost,
        }
