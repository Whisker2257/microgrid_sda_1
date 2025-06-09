#/Users/nashe/nested_policy_pipeline/src/codegen/task_generator.py 
"""
Deepseek high-level **Task Generator** with retry/timeout handling.
"""
from __future__ import annotations

import os               # ← added import
import textwrap
import time
from typing import Dict, Any

import requests
from requests.exceptions import RequestException, ReadTimeout

from config import (
    DEESEEK_API_KEY,
    DEESEEK_BASE_URL,
    MODEL_DEEPSEEK,
    TASK_TEMPERATURE,
    TASK_MAX_TOKENS,
)

# default 180 s; override via .env → DEESEEK_TIMEOUT=240
DEESEEK_TIMEOUT = int(os.getenv("DEESEEK_TIMEOUT", "180"))


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _post_with_retry(payload: dict, retries: int = 3) -> str:
    """
    Call Deepseek chat endpoint with exponential back-off on ReadTimeout.
    """
    delay = 5  # seconds
    url = f"{DEESEEK_BASE_URL}/chat/completions"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {DEESEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=DEESEEK_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except ReadTimeout:
            print(f"⚠️  Deepseek read timeout (attempt {attempt}/{retries}). "
                  f"Retrying in {delay}s …")
            time.sleep(delay)
            delay *= 2  # exponential back-off

        except RequestException as e:
            raise RuntimeError(f"Deepseek API error: {e}") from e

    raise RuntimeError("Deepseek API timed out after multiple retries.")


# ---------------------------------------------------------------------
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
    Build and (via Deepseek) refine a task prompt for the Code-Generator LLM.
    Falls back to a stub prompt if Deepseek keeps timing out.
    """
    sys_prompt = (
        "You are a planning agent (Task Generator). "
        "Your output must be a prompt for another LLM that writes Python "
        "code. Do NOT include any Python code yourself."
    )

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
        Craft a concise *task description* asking the coding model to emit
        exactly one class `GeneratedPolicy` with:

          • __init__(…) initialising ONLY keys in T_Q
          • take_action(self, state: np.ndarray) -> float

        Include a reference signature in a code block.  No other content.
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
            Refine your instructions so the next snippet passes.
            """
        )

    payload = {
        "model": MODEL_DEEPSEEK,           # default: deepseek-reasoner
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": usr_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        return _post_with_retry(payload)

    except RuntimeError as e:
        print("❌ Deepseek call failed after retries:", e)
        print("▶ Falling back to a minimal stub prompt so the loop continues.")
        return (
            "Rewrite the MovingAveragePolicy exactly as-is "
            "(same __init__ signature, same take_action logic)."
        )
