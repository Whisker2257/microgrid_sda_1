#/Users/nashe/nested_policy_pipeline/src/meta/meta_controller.py
"""
Meta-controller: Task Generator ➜ Code Generator ➜ ϑ filter
"""
from __future__ import annotations

import inspect
import logging
from typing import Dict, Any, Tuple

from src.codegen.task_generator import build_task_prompt
from src.codegen.code_generator_qwen import generate_policy_code
from src.utils.filter import vartheta

logger = logging.getLogger(__name__)


def _safe_get_source(obj) -> str:
    try:
        return inspect.getsource(obj.__class__)
    except (OSError, TypeError):
        return obj.__class__.__name__


def meta_update(
    base_policy,
    meta_history: Dict[str, list],
    meta_params: Dict[str, Any],
    *,
    max_retries: int = 3,
) -> Tuple[Any, Dict[str, Any]]:
    """Generate, filter, and instantiate a new base policy."""
    error_msg: str | None = None

    for attempt in range(1, max_retries + 1):
        logger.info("Meta-update attempt %d/%d", attempt, max_retries)

        policy_descriptor = _safe_get_source(base_policy)
        task_prompt = build_task_prompt(
            policy_descriptor,
            meta_history,
            meta_params,
            error_ctx=error_msg,
        )

        code_snippet = generate_policy_code(task_prompt)

        try:
            new_policy, new_params = vartheta(code_snippet)
            logger.info("ϑ accepted generated policy")
            return new_policy, new_params
        except ValueError as err:
            error_msg = str(err)
            logger.warning("ϑ rejected policy: %s", error_msg)

    raise RuntimeError("Meta-controller failed after all retries.")
