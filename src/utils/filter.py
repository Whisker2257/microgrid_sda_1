# src/utils/filter.py
import ast
import inspect
from typing import Tuple, Dict, Any


def vartheta(wq_code: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Filter and instantiate an LLM-generated policy.

    This function:
      1. Parses the code via AST to enforce safety constraints.
      2. Executes the code in a restricted namespace.
      3. Finds exactly one policy class with a take_action method.
      4. Extracts its __init__ parameters and default values.
      5. Instantiates the policy and returns both the instance and params.

    Args:
      wq_code:      Multiline Python code defining a policy class.

    Returns:
      policy_inst:  An instance of the generated policy class.
      params:       A dict mapping parameter names to values used in instantiation.

    Raises:
      ValueError:   If unsafe constructs are detected or class extraction fails.
    """
    # 1) Parse code into an AST and enforce safety (no imports allowed)
    tree = ast.parse(wq_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements not allowed in generated policy code.")

    # 2) Compile and execute in restricted namespace
    local_ns: Dict[str, Any] = {}
    exec(compile(tree, filename="<generated_policy>", mode="exec"), {}, local_ns)

    # 3) Find policy classes (must define a take_action method)
    policy_classes = [obj for obj in local_ns.values()
                      if inspect.isclass(obj) and hasattr(obj, 'take_action')]
    if len(policy_classes) != 1:
        raise ValueError(
            f"Expected exactly one policy class with take_action, found {len(policy_classes)}"
        )
    PolicyClass = policy_classes[0]

    # 4) Inspect __init__ to get parameters and defaults
    sig = inspect.signature(PolicyClass.__init__)
    init_params: Dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.default is inspect.Parameter.empty:
            raise ValueError(f"Parameter '{name}' in __init__ must have a default value.")
        init_params[name] = param.default

    # 5) Instantiate policy with default parameters
    policy_inst = PolicyClass(**init_params)

    return policy_inst, init_params
