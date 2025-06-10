# File: src/policies/moving_average_policy.py
import numpy as np
from collections import deque

class MovingAveragePolicy:
    """
    A simple baseline policy that charges when the current price
    is below the moving average over the last `window` steps,
    discharges when above, and holds otherwise.
    Unified take_action(state) API.
    """

    def __init__(self, window: int, max_rate: float = 1.0):
        """
        Args:
            window:   Number of past time-steps to include in the average.
            max_rate: Maximum magnitude of charge (+) or discharge (–) per step [kWh].
        """
        self.window = window
        self.prices = deque(maxlen=window)
        self.max_rate = max_rate

    def _take_action_scalar(
        self,
        state_of_charge: float,
        imported_energy: float,
        market_price: float,
        cost: float
    ) -> float:
        # Append to history
        self.prices.append(market_price)

        # If not enough history yet, do nothing
        if len(self.prices) < self.window:
            return 0.0

        # Compute the moving average price
        avg_price = sum(self.prices) / len(self.prices)

        # Charge if current price below average
        if market_price < avg_price:
            return +self.max_rate

        # Discharge if above average (but not more than current SOC)
        if market_price > avg_price:
            return -min(self.max_rate, state_of_charge)

        # Otherwise hold
        return 0.0

    def take_action(self, state: np.ndarray) -> float:
        """
        Unified API:
          Args:
            state: 5‐vector [soc, imported_energy, market_price, cost, demand]
          Returns:
            Q_n: positive → charge, negative → discharge
        """
        soc, imp_en, price, cost, demand = state
        return self._take_action_scalar(soc, imp_en, price, cost)
