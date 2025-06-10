#/Users/nashe/nested_policy_pipeline/src/environment/battery_env.py
import numpy as np
from config import (
    HORIZON,
    PRICE_SERIES,
    DEMAND_SERIES,
    INITIAL_SOC,
)
from src.utils.transition import transition        # ← fixed prefix


class BatteryEnvironment:
    """
    State vector = [soc, imported_energy, market_price, cost, demand]
    """

    def __init__(self):
        self.price_series  = np.array(PRICE_SERIES,  dtype=float)
        self.demand_series = np.array(DEMAND_SERIES, dtype=float)
        self.reset()

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.step_index = 0
        self.state = np.array(
            [
                INITIAL_SOC,          # soc
                0.0,                  # imported_energy
                self.price_series[0], # current price
                0.0,                  # cumulative cost
                self.demand_series[0] # current demand
            ],
            dtype=float,
        )
        return self.state

    def step(self, action: float) -> np.ndarray:
        next_price  = self.price_series[self.step_index + 1]
        next_demand = self.demand_series[self.step_index + 1]

        self.state = transition(self.state, action, next_price, next_demand)
        self.step_index += 1
        return self.state
