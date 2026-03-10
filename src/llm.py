"""Shared LLM client for epistemic synthesis.

Uses Anthropic Claude Sonnet for all LLM calls. Falls back to local
Qwen3-30B (OpenAI-compatible) if Anthropic key unavailable.
"""

import json
import logging
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

_SECRETS_PATH = Path.home() / ".openclaw/secrets.json"
_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        secrets = json.loads(_SECRETS_PATH.read_text())
        key = secrets["anthropic"]["default"]
        _client = anthropic.Anthropic(api_key=key)
    return _client


def call_llm(prompt: str, *, max_tokens: int = 8192, json_mode: bool = False) -> str:
    """Call Sonnet and return the text response.

    Args:
        prompt: The user message
        max_tokens: Max tokens in response
        json_mode: If True, prepend JSON instruction to system prompt

    Returns:
        Raw text content from the model
    """
    client = _get_client()
    system = "You are a precise analytical assistant."
    if json_mode:
        system += " Respond with valid JSON only, no markdown fences."

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_llm_json(prompt: str, *, max_tokens: int = 8192) -> dict:
    """Call Sonnet and parse JSON response.

    Returns:
        Parsed JSON dict
    Raises:
        json.JSONDecodeError if response isn't valid JSON
    """
    text = call_llm(prompt, max_tokens=max_tokens, json_mode=True)
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)
