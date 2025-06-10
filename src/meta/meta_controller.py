# File: src/meta/meta_controller.py

"""
Meta-controller: Task Generator ➜ Code Generator ➜ ϑ filter
"""
from __future__ import annotations

import inspect
import logging
from typing import Dict, Any, Tuple

import requests
from requests.exceptions import RequestException

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
    """
    Generate, filter, and instantiate a new base policy, feeding the full
    code of the last policy (and any error context) back into the Task Generator.
    """
    error_msg: str | None = None

    # Start with the source of the current policy (fallback to class name)
    last_code_src = _safe_get_source(base_policy)

    for attempt in range(1, max_retries + 1):
        logger.info("Meta-update attempt %d/%d", attempt, max_retries)

        # 1) Build task prompt, including the full text of the last policy
        task_prompt = build_task_prompt(
            last_code_src,
            meta_history,
            meta_params,
            error_ctx=error_msg,
        )

        # 2) Call Code Generator (generate_policy_code), fallback to last_code_src on API errors
        try:
            code_snippet = generate_policy_code(task_prompt)
        except RequestException as e:
            logger.warning(
                "Code-generator API error (%s). Reusing last policy code.",
                e
            )
            code_snippet = last_code_src

        # Update last_code_src so the next prompt sees this snippet
        last_code_src = code_snippet

        # 3) Filter & instantiate via ϑ
        try:
            new_policy, new_params = vartheta(code_snippet)
            logger.info("ϑ accepted generated policy")
            return new_policy, new_params

        except ValueError as err:
            error_msg = str(err)
            logger.warning("ϑ rejected policy: %s", error_msg)

    raise RuntimeError("Meta-controller failed after all retries.")
