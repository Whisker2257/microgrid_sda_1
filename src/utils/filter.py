# File: src/utils/filter.py
import ast
import inspect
import numpy as np
from typing import Tuple, Dict, Any

def vartheta(wq_code: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Filter and instantiate an LLM-generated policy, then wrap its take_action
    so it always conforms to take_action(state: np.ndarray) -> float.
    """
    # 1) Parse & ban imports
    tree = ast.parse(wq_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements not allowed in generated policy code.")

    # 2) Exec in a namespace where np is available
    safe_globals: Dict[str, Any] = {"np": np}
    local_ns: Dict[str, Any] = {}
    exec(compile(tree, filename="<generated_policy>", mode="exec"), safe_globals, local_ns)

    # 3) Find exactly one policy class
    policy_classes = [
        obj for obj in local_ns.values()
        if inspect.isclass(obj) and hasattr(obj, "take_action")
    ]
    if len(policy_classes) != 1:
        raise ValueError(f"Expected exactly one policy class with take_action, found {len(policy_classes)}")
    PolicyClass = policy_classes[0]

    # 4) Inspect __init__ for defaults
    sig = inspect.signature(PolicyClass.__init__)
    init_params: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            raise ValueError(f"Parameter '{name}' in __init__ must have a default value.")
        init_params[name] = param.default

    # 5) Instantiate policy
    policy_inst = PolicyClass(**init_params)

    # 6) Wrap its take_action if it doesn't already accept a single state arg
    orig = policy_inst.take_action
    bound_sig = inspect.signature(orig)
    if len(bound_sig.parameters) != 1:
        # e.g. orig takes (state_of_charge, imported_energy, market_price, cost)
        def unified_take_action(state: np.ndarray) -> float:
            soc, imp, price, cost, *_ = state
            return orig(soc, imp, price, cost)
        policy_inst.take_action = unified_take_action  # override instance method

    return policy_inst, init_params
