# File: src/codegen/task_generator.py
"""
Task Generator with retry/timeout handling via OpenRouter.
"""
from __future__ import annotations

import os
import json
import textwrap
import time
from typing import Dict, Any

import requests
from requests.exceptions import RequestException, ReadTimeout

from config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    MODEL_DEEPSEEK,
    TASK_TEMPERATURE,
    TASK_MAX_TOKENS,
)

# default 180 s; override via .env → OPENROUTER_TIMEOUT=240
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "180"))


def _post_with_retry(payload: dict, retries: int = 3) -> str:
    """
    Call OpenRouter chat endpoint with exponential back-off on ReadTimeout.
    """
    delay = 5  # seconds
    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=OPENROUTER_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except ReadTimeout:
            print(
                f"⚠️  OpenRouter read timeout "
                f"(attempt {attempt}/{retries}). Retrying in {delay}s …"
            )
            time.sleep(delay)
            delay *= 2

        except RequestException as e:
            raise RuntimeError(f"OpenRouter API error: {e}") from e

    raise RuntimeError("OpenRouter API timed out after multiple retries.")


def build_task_prompt(
    base_policy_src: str,
    meta_history: Dict[str, list],
    meta_params: Dict[str, Any],
    error_ctx: str | None = None,
    *,
    temperature: float = TASK_TEMPERATURE,
    max_tokens: int = TASK_MAX_TOKENS,
) -> str:
    """
    Build and (via OpenRouter) refine a task prompt for the Code Generator LLM.
    Ensures no import statements and default values for all __init__ parameters.
    Falls back to a stub prompt if OpenRouter keeps timing out.
    """
    sys_prompt = (
        "You are a planning agent (Task Generator). "
        "Your output must be a prompt for another LLM that writes pure Python code. "
        "Do NOT include any Python code yourself."
    )

    # User instructions: emphasize no imports, default values, and correct signature
    usr_prompt = textwrap.dedent(
        f"""
        ## Current policy implementation ##
        ```python
        {base_policy_src}
        ```

        ## Meta-history ĤN ##
        {meta_history}

        ## Meta-parameters T_Q ##
        {meta_params}

        ----
        Craft a concise *task description* asking the code-generation model to emit
        exactly one class `GeneratedPolicy` in a Python code block, with:

          • An `__init__` method whose signature includes **only** the keys in T_Q as parameters,
            each with a default literal value (e.g. `learning_rate: float = 0.01`).
          • A `take_action(self, state_of_charge: float, imported_energy: float, market_price: float, cost: float) -> float`
            method matching Appendix C.3, without any import statements (including `numpy`).

        Do NOT include any import lines in the generated code. Provide only the Python class definition.
        """
    )

    if error_ctx:
        usr_prompt += textwrap.dedent(
            f"""

            ---
            ⚠️  Previous attempt failed with:
            ```
            {error_ctx.strip()}
            ```
            Refine your instructions so the next snippet passes all safety and signature checks.
            """
        )

    payload = {
        "model": MODEL_DEEPSEEK,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": usr_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    return _post_with_retry(payload)
