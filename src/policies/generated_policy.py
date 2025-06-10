#/Users/nashe/nested_policy_pipeline/src/policies/generated_policy.py
import numpy as np

class GeneratedPolicy:
    """
    Example template for an LLM‐generated policy class.
    Uses a fixed price threshold to decide charge vs discharge.
    Unified take_action(state) API.
    """

    def __init__(self, threshold: float, max_rate: float = 1.0):
        """
        Args:
            threshold:  Price boundary [€/kWh]. 
                        Prices below → charge; above → discharge.
            max_rate:   Maximum magnitude of charge/discharge [kWh].
        """
        self.threshold = threshold
        self.max_rate = max_rate

    def _take_action_scalar(
        self,
        state_of_charge: float,
        imported_energy: float,
        market_price: float,
        cost: float
    ) -> float:
        current_price = market_price
        battery_level = state_of_charge

        # If price below threshold → charge
        if current_price < self.threshold:
            return +self.max_rate

        # If price above threshold → discharge (but not more than SOC)
        elif current_price > self.threshold:
            return -min(self.max_rate, battery_level)

        # Otherwise hold
        return 0.0

    def take_action(self, state: np.ndarray) -> float:
        """
        Unified API:
          Args:
            state: 5‐vector [soc, imported_energy, market_price, cost, demand]
        """
        soc, imp_en, price, cost, demand = state
        return self._take_action_scalar(soc, imp_en, price, cost)
