from __future__ import annotations

import os
from collections.abc import Mapping


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def resolve_api_base_url(secrets: Mapping[str, str] | None = None) -> str:
    if secrets is not None:
        value = secrets.get("api_base_url")
        if value:
            return value.rstrip("/")
    env_value = os.getenv("MEMETALK_API_BASE_URL")
    if env_value:
        return env_value.rstrip("/")
    return DEFAULT_API_BASE_URL
