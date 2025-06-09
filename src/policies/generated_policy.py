#/Users/nashe/nested_policy_pipeline/src/policies/generated_policy.py
import numpy as np

class GeneratedPolicy:
    """
    Example template for an LLM‐generated policy class.
    Uses a fixed price threshold to decide charge vs discharge.

    Interface matches MovingAveragePolicy:
        __init__(threshold: float, max_rate: float = 1.0)
        take_action(state: np.ndarray) -> float
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

    def take_action(self, state: np.ndarray) -> float:
        """
        Decide action based on fixed threshold rule.

        Args:
            state:  4‐vector [battery_level, imported_energy, market_price, cost].

        Returns:
            A float Q_n: positive → charge up to max_rate,
                         negative → discharge up to max_rate
        """
        current_price = float(state[2])
        battery_level = float(state[0])

        # If price below threshold → charge
        if current_price < self.threshold:
            return +self.max_rate

        # If price above threshold → discharge (but not more than SOC)
        elif current_price > self.threshold:
            return -min(self.max_rate, battery_level)

        # Otherwise hold
        return 0.0
