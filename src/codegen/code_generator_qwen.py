#/Users/nashe/nested_policy_pipeline/src/codegen/code_generator_qwen.py 
"""
Qwen-2.5-Coder  →  **Code Generator**

Consumes a task prompt and returns raw Python.
"""
from __future__ import annotations
import requests
from config import (
    QWEN_API_KEY,
    MODEL_QWEN,
    CODE_TEMPERATURE,
    CODE_MAX_TOKENS,
)


def generate_policy_code(
    task_prompt: str,
    *,
    temperature: float = CODE_TEMPERATURE,
    max_tokens: int = CODE_MAX_TOKENS,
) -> str:
    """Return pure Python code implementing the requested policy."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a senior Python engineer. "
            "Return ONLY valid Python 3 code for the requested class."
        ),
    }
    user_msg = {"role": "user", "content": task_prompt}

    resp = requests.post(
        "https://api.qwen.com/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_QWEN,
            "messages": [system_msg, user_msg],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
