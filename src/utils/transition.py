#/Users/nashe/nested_policy_pipeline/src/utils/transition.
import numpy as np
from config import BATTERY_CAPACITY_KWH, MAX_RATE_KWH

def transition(
    state: np.ndarray,
    action: float,
    next_price: float,
    next_demand: float,
) -> np.ndarray:
    """
    Lossless battery dynamics with:
      • demand satisfaction
      • grid import cost & export revenue
      • rate limits ±MAX_RATE_KWH
    This drops any charge/discharge inefficiencies (EFF_*).
    """
    soc, imported, _, cost, _ = state

    # 0. enforce rate & capacity limits
    action = np.clip(action, -MAX_RATE_KWH, MAX_RATE_KWH)
    if action > 0:  # charging
        # lossless: all action goes into the battery
        charge_storable = min(action, BATTERY_CAPACITY_KWH - soc)
        new_soc = soc + charge_storable
        import_chg = charge_storable
        export_rev = 0.0
    else:  # discharging / export
        discharge_req_bat = min(-action, soc)
        energy_to_grid = discharge_req_bat
        new_soc = soc - discharge_req_bat
        import_chg = 0.0
        export_rev = energy_to_grid * next_price

    # 1. serve demand from remaining SOC
    discharge_for_demand = min(new_soc, next_demand)
    new_soc -= discharge_for_demand
    unmet = next_demand - discharge_for_demand

    # 2. grid import for unmet demand
    import_dmd = unmet
    import_total = import_chg + import_dmd

    # 3. cost update (imports positive, exports negative)
    new_cost = cost + import_total * next_price - export_rev

    return np.array(
        [
            new_soc,
            imported + import_total,  # cumulative grid energy
            next_price,
            new_cost,
            next_demand,
        ],
        dtype=float,
    )
