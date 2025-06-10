# File: src/codegen/code_generator_qwen.py

from __future__ import annotations

import json
import logging
import re
import requests
from requests.exceptions import HTTPError, RequestException

from config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    MODEL_QWEN,
    CODE_TEMPERATURE,
    CODE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def generate_policy_code(
    task_prompt: str,
    *,
    temperature: float = CODE_TEMPERATURE,
    max_tokens: int = CODE_MAX_TOKENS,
) -> str:
    """Return pure Python code implementing the requested policy via OpenRouter."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a senior Python engineer. "
            "Return ONLY valid Python 3 code for the requested class."
        ),
    }
    user_msg = {"role": "user", "content": task_prompt}

    payload = {
        "model": MODEL_QWEN,
        "messages": [system_msg, user_msg],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=60,
    )

    try:
        resp.raise_for_status()
    except HTTPError as http_err:
        # log full error
        try:
            err_body = resp.json()
        except ValueError:
            err_body = resp.text
        logger.error(
            "OpenRouter API error (status=%s): %s",
            resp.status_code,
            err_body,
        )
        raise RuntimeError(
            f"OpenRouter code-generation failed with status {resp.status_code}: {err_body}"
        ) from http_err

    # parse out the raw code text
    try:
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as parse_err:
        logger.error("Unexpected response format from OpenRouter: %s", resp.text)
        raise RuntimeError(f"Failed to parse code-generation response: {resp.text}") from parse_err

    # Remove any Markdown code-fence lines anywhere in the block
    lines = raw.splitlines()
    filtered_lines = [line for line in lines if not re.match(r"^\s*```", line)]
    code = "\n".join(filtered_lines)

    # Strip leading/trailing whitespace
    return code.strip()
