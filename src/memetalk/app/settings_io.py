from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from memetalk.config import AppSettings

DEFAULT_CONFIG_PATH = Path("data/memetalk_config.toml")

# Fields that map between AppSettings and TOML keys (same names).
_SETTINGS_FIELDS = [f.alias or name for name, f in AppSettings.model_fields.items()]


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> AppSettings:
    """Load settings from TOML file, with env vars taking priority."""
    toml_data: dict = {}
    if path.exists():
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)
    # Build kwargs from TOML, then let from_env overlay env vars
    # Strategy: read TOML first, set as env defaults where env is not set
    env_overrides: dict = {}
    env_mapping = {
        "sqlite_path": "MEMETALK_SQLITE_PATH",
        "vector_backend": "MEMETALK_VECTOR_BACKEND",
        "chroma_path": "MEMETALK_CHROMA_PATH",
        "chroma_collection": "MEMETALK_CHROMA_COLLECTION",
        "provider_backend": "MEMETALK_PROVIDER_BACKEND",
        "ocr_backend": "MEMETALK_OCR_BACKEND",
        "api_base_url": "MEMETALK_API_BASE_URL",
        "openai_base_url": "MEMETALK_OPENAI_BASE_URL",
        "openai_api_key": "MEMETALK_OPENAI_API_KEY",
        "openai_chat_model": "MEMETALK_OPENAI_CHAT_MODEL",
        "openai_vision_model": "MEMETALK_OPENAI_VISION_MODEL",
        "openai_embedding_model": "MEMETALK_OPENAI_EMBEDDING_MODEL",
        "lmstudio_base_url": "MEMETALK_LMSTUDIO_BASE_URL",
        "lmstudio_api_key": "MEMETALK_LMSTUDIO_API_KEY",
        "lmstudio_chat_model": "MEMETALK_LMSTUDIO_CHAT_MODEL",
        "lmstudio_vision_model": "MEMETALK_LMSTUDIO_VISION_MODEL",
        "lmstudio_embedding_model": "MEMETALK_LMSTUDIO_EMBEDDING_MODEL",
        "ollama_base_url": "MEMETALK_OLLAMA_BASE_URL",
        "ollama_chat_model": "MEMETALK_OLLAMA_CHAT_MODEL",
        "ollama_vision_model": "MEMETALK_OLLAMA_VISION_MODEL",
        "ollama_embedding_model": "MEMETALK_OLLAMA_EMBEDDING_MODEL",
        "llama_cpp_base_url": "MEMETALK_LLAMA_CPP_BASE_URL",
        "gemini_api_key": "MEMETALK_GEMINI_API_KEY",
        "gemini_chat_model": "MEMETALK_GEMINI_CHAT_MODEL",
        "gemini_embedding_model": "MEMETALK_GEMINI_EMBEDDING_MODEL",
        "claude_api_key": "MEMETALK_CLAUDE_API_KEY",
        "claude_chat_model": "MEMETALK_CLAUDE_CHAT_MODEL",
        "claude_vision_model": "MEMETALK_CLAUDE_VISION_MODEL",
        "claude_embedding_provider": "MEMETALK_CLAUDE_EMBEDDING_PROVIDER",
        "search_candidate_k_default": "MEMETALK_SEARCH_CANDIDATE_K",
        "search_top_n_default": "MEMETALK_SEARCH_TOP_N",
        "search_rerank_pool_size": "MEMETALK_SEARCH_RERANK_POOL_SIZE",
        "search_scoring_profile_path": "MEMETALK_SEARCH_SCORING_PROFILE_PATH",
        "meme_folder": "MEMETALK_MEME_FOLDER",
    }

    for field_name in _SETTINGS_FIELDS:
        env_var = env_mapping.get(field_name)
        env_val = os.getenv(env_var) if env_var else None
        if env_val is not None:
            env_overrides[field_name] = env_val
        elif field_name in toml_data:
            env_overrides[field_name] = toml_data[field_name]

    return AppSettings(**env_overrides)


def save_settings(settings: AppSettings, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Save settings to TOML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    for field_name in _SETTINGS_FIELDS:
        value = getattr(settings, field_name)
        if value is None:
            continue
        if isinstance(value, Path):
            data[field_name] = str(value)
        else:
            data[field_name] = value
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def merge_settings(base: AppSettings, updates: Mapping[str, Any]) -> AppSettings:
    """Return validated settings with selected fields updated and others preserved."""
    payload = base.model_dump(mode="python")
    payload.update(updates)
    return AppSettings.model_validate(payload)
