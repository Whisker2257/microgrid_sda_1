# File: src/algorithm/nested_algorithm.py
# File: src/algorithm/nested_algorithm.py

import logging
from typing import Any, Dict, List, Sequence

from config import HORIZON, META_STEPS
from src.environment.battery_env import BatteryEnvironment
from src.policies.moving_average_policy import MovingAveragePolicy
from src.meta.meta_controller import meta_update

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_nested_algorithm() -> Dict[str, Any]:
    """
    Runs the hierarchical (meta + base) nested algorithm.

    Returns
    -------
    dict with keys:
        final_state, history, meta_params, final_policy, per_segment
    """
    env = BatteryEnvironment()
    N_current = env.reset()  # [soc, imported, price, cost, demand]

    hat_N: Dict[str, List[float]] = {
        "battery_level_record": [float(N_current[0])],
        "action_record":        [],
        "cost_per_time_record": [],
        "total_cost_record":    [float(N_current[3])],
    }

    T_current: Dict[str, Any] = {"learning_rate": 0.01, "window_size": 24}
    base_policy = MovingAveragePolicy(window=T_current["window_size"])
    segment_len = HORIZON // META_STEPS
    results: List[Dict[str, Any]] = []

    for V in range(META_STEPS):
        logger.info("=== Meta-step V=%d ===", V)

        # Meta-update
        if V > 0:
            base_policy, T_current = meta_update(base_policy, hat_N, T_current)
            logger.info(
                "Meta-update ➜ %s  T=%s",
                base_policy.__class__.__name__,
                T_current,
            )

        # Inner loop over this segment
        for _ in range(segment_len):
            # Convert state array to tuple of floats
            state: Sequence[float] = tuple(float(x) for x in N_current)

            # Call unified API
            Q_n = base_policy.take_action(state)

            # --- Validation & fallback ---
            try:
                # Catch None or non-numeric returns
                if Q_n is None:
                    raise TypeError("take_action returned None")
                Q_n = float(Q_n)
            except Exception as e:
                logger.warning(
                    "Invalid action %r from %s; defaulting to 0.0 (%s)",
                    Q_n,
                    base_policy.__class__.__name__,
                    e,
                )
                Q_n = 0.0

            # Step environment
            N_current = env.step(Q_n)

            # Record
            hat_N["battery_level_record"].append(float(N_current[0]))
            hat_N["action_record"].append(float(Q_n))
            delta_cost = float(N_current[3]) - hat_N["total_cost_record"][-1]
            hat_N["cost_per_time_record"].append(delta_cost)
            hat_N["total_cost_record"].append(float(N_current[3]))

        # Segment cost
        segment_cost = sum(hat_N["cost_per_time_record"][-segment_len:])
        logger.info("Segment %d cost = %.3f", V, segment_cost)

        results.append(
            dict(
                meta_step=V,
                end_state=N_current.copy(),
                meta_params=T_current.copy(),
                policy_name=base_policy.__class__.__name__,
                segment_cost=segment_cost,
            )
        )

    return dict(
        final_state=N_current,
        history=hat_N,
        meta_params=T_current,
        final_policy=base_policy,
        per_segment=results,
    )
