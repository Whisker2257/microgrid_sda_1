#/Users/nashe/nested_policy_pipeline/src/policies/moving_average_policy.py
from collections import deque
import numpy as np

class MovingAveragePolicy:
    """
    A simple baseline policy that charges when the current price
    is below the moving average over the last `window` steps,
    discharges when above, and holds otherwise.

    Interface:
        __init__(window: int, max_rate: float = 1.0)
        take_action(state: np.ndarray) -> float
    """

    def __init__(self, window: int, max_rate: float = 1.0):
        """
        Args:
            window:    Number of past time‐steps to include in the average.
            max_rate:  Maximum magnitude of charge (+) or discharge (–) per step [kWh].
        """
        self.window = window
        # Rolling buffer of most recent prices
        self.prices = deque(maxlen=window)
        self.max_rate = max_rate

    def take_action(self, state: np.ndarray) -> float:
        """
        Decide on charge/discharge action based on price history.

        Args:
            state:  4‐vector [battery_level, imported_energy, market_price, cost].

        Returns:
            A float Q_n: positive → charge up to max_rate, 
                         negative → discharge up to max_rate 
        """
        # Extract current price from state
        current_price = float(state[2])
        # Add to history
        self.prices.append(current_price)

        # If not enough history yet, do nothing
        if len(self.prices) < self.window:
            return 0.0

        # Compute the moving average price
        avg_price = sum(self.prices) / len(self.prices)
        battery_level = float(state[0])

        # Charge if current price below average
        if current_price < avg_price:
            return +self.max_rate

        # Discharge if above average (but not more than current SOC)
        elif current_price > avg_price:
            # ensure we don't discharge more than we have
            return -min(self.max_rate, battery_level)

        # Otherwise hold
        return 0.0
