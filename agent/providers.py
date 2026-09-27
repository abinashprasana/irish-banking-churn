"""Gemini support beside the existing Groq path.

Gemini is reached through its OpenAI compatible Chat Completions endpoint, so the
hand written tool loop, the four tools and the policy gate are unchanged. The
client uses only the standard library and returns plain dictionaries, which the
loop already reads through ``_get``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


GEMINI_MODEL_NAME = "gemini-3.8-flash"
GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_TIMEOUT_SECONDS = 60
# Gemini's hidden thinking tokens count against the completion cap the loop sets
# (1,024). Tool routing needs little thinking, so ask for a low effort.
GEMINI_REASONING_EFFORT = "low"
PROVIDERS = ("groq", "gemini")

_GEMINI_KEY_PLACEHOLDERS = frozenset({"", "...", "your-gemini-api-key", "paste-your-key-here"})


class GeminiAPIError(RuntimeError):
    """HTTP error from Gemini, shaped like SDK errors for the loop's retry check."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Gemini API error {status_code}: {message}")
        self.status_code = status_code


def validated_gemini_api_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if candidate.casefold() in _GEMINI_KEY_PLACEHOLDERS or any(c.isspace() for c in candidate):
        return None
    return candidate


def resolve_gemini_api_key(secrets: Any | None = None) -> str | None:
    """Prefer a valid Streamlit secret, then the environment. Never logs the value."""

    secret_value = None
    if secrets is not None:
        try:
            secret_value = secrets.get("GEMINI_API_KEY")
        except Exception:
            secret_value = None
    return validated_gemini_api_key(secret_value) or validated_gemini_api_key(
        os.environ.get("GEMINI_API_KEY")
    )


def load_env_file(path: Path | None = None) -> list[str]:
    """Load KEY=value lines from the gitignored .env into os.environ.

    Existing environment values win. Returns the names loaded, never the values.
    Only the command line entry points call this; tests never do.
    """

    env_path = path or Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return []
    loaded = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1].strip()
        if name and value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


class _GeminiCompletions:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def create(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("reasoning_effort", GEMINI_REASONING_EFFORT)
        request = urllib.request.Request(
            GEMINI_CHAT_URL,
            data=json.dumps(kwargs).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=GEMINI_TIMEOUT_SECONDS) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").replace(self._api_key, "[redacted]")
            raise GeminiAPIError(exc.code, body[:500]) from None


class _GeminiChat:
    def __init__(self, api_key: str) -> None:
        self.completions = _GeminiCompletions(api_key)


class GeminiCompatClient:
    """Minimal OpenAI shaped client: ``client.chat.completions.create(**kwargs)``."""

    def __init__(self, api_key: str) -> None:
        self.chat = _GeminiChat(api_key)
