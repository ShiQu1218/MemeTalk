from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


class AppSettings(BaseModel):
    sqlite_path: Path = Field(default_factory=lambda: Path("data/memetalk.sqlite3"))
    vector_backend: Literal["chroma", "memory"] = "chroma"
    chroma_path: Path = Field(default_factory=lambda: Path("data/chroma"))
    chroma_collection: str = "memetalk_memes"
    provider_backend: Literal[
        "openai", "mock", "local", "lmstudio",
        "ollama", "llama_cpp", "gemini", "claude",
    ] = "openai"
    ocr_backend: Literal["paddleocr", "mock"] = "paddleocr"
    api_base_url: str = "http://127.0.0.1:8000"
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"
    lmstudio_api_key: str | None = None
    lmstudio_chat_model: str | None = None
    lmstudio_vision_model: str | None = None
    lmstudio_embedding_model: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model: str | None = None
    ollama_vision_model: str | None = None
    ollama_embedding_model: str | None = None
    llama_cpp_base_url: str = "http://localhost:8080/v1"
    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"
    claude_api_key: str | None = None
    claude_chat_model: str = "claude-sonnet-4-20250514"
    claude_vision_model: str = "claude-sonnet-4-20250514"
    claude_embedding_provider: Literal["openai", "gemini"] = "openai"
    search_candidate_k_default: int = 20
    search_top_n_default: int = 6
    search_rerank_pool_size: int = 16
    search_scoring_profile_path: Path = Field(default_factory=lambda: Path("data/search_scoring_profile.json"))
    meme_folder: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str | None = None

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            sqlite_path=_env_path("MEMETALK_SQLITE_PATH", "data/memetalk.sqlite3"),
            vector_backend=os.getenv("MEMETALK_VECTOR_BACKEND", "chroma"),
            chroma_path=_env_path("MEMETALK_CHROMA_PATH", "data/chroma"),
            chroma_collection=os.getenv("MEMETALK_CHROMA_COLLECTION", "memetalk_memes"),
            provider_backend=os.getenv("MEMETALK_PROVIDER_BACKEND", "openai"),
            ocr_backend=os.getenv("MEMETALK_OCR_BACKEND", "paddleocr"),
            api_base_url=os.getenv("MEMETALK_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            openai_base_url=(os.getenv("MEMETALK_OPENAI_BASE_URL") or "").rstrip("/") or None,
            openai_api_key=os.getenv("MEMETALK_OPENAI_API_KEY"),
            openai_chat_model=os.getenv("MEMETALK_OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            openai_vision_model=os.getenv("MEMETALK_OPENAI_VISION_MODEL", "gpt-4.1-mini"),
            openai_embedding_model=os.getenv("MEMETALK_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            lmstudio_base_url=os.getenv("MEMETALK_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/"),
            lmstudio_api_key=os.getenv("MEMETALK_LMSTUDIO_API_KEY"),
            lmstudio_chat_model=os.getenv("MEMETALK_LMSTUDIO_CHAT_MODEL"),
            lmstudio_vision_model=os.getenv("MEMETALK_LMSTUDIO_VISION_MODEL"),
            lmstudio_embedding_model=os.getenv("MEMETALK_LMSTUDIO_EMBEDDING_MODEL"),
            ollama_base_url=os.getenv("MEMETALK_OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
            ollama_chat_model=os.getenv("MEMETALK_OLLAMA_CHAT_MODEL"),
            ollama_vision_model=os.getenv("MEMETALK_OLLAMA_VISION_MODEL"),
            ollama_embedding_model=os.getenv("MEMETALK_OLLAMA_EMBEDDING_MODEL"),
            llama_cpp_base_url=os.getenv("MEMETALK_LLAMA_CPP_BASE_URL", "http://localhost:8080/v1").rstrip("/"),
            gemini_api_key=os.getenv("MEMETALK_GEMINI_API_KEY"),
            gemini_chat_model=os.getenv("MEMETALK_GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
            gemini_embedding_model=os.getenv("MEMETALK_GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
            claude_api_key=os.getenv("MEMETALK_CLAUDE_API_KEY"),
            claude_chat_model=os.getenv("MEMETALK_CLAUDE_CHAT_MODEL", "claude-sonnet-4-20250514"),
            claude_vision_model=os.getenv("MEMETALK_CLAUDE_VISION_MODEL", "claude-sonnet-4-20250514"),
            claude_embedding_provider=os.getenv("MEMETALK_CLAUDE_EMBEDDING_PROVIDER", "openai"),
            search_candidate_k_default=_env_int("MEMETALK_SEARCH_CANDIDATE_K", 20),
            search_top_n_default=_env_int("MEMETALK_SEARCH_TOP_N", 6),
            search_rerank_pool_size=_env_int("MEMETALK_SEARCH_RERANK_POOL_SIZE", 16),
            search_scoring_profile_path=_env_path(
                "MEMETALK_SEARCH_SCORING_PROFILE_PATH",
                "data/search_scoring_profile.json",
            ),
            meme_folder=os.getenv("MEMETALK_MEME_FOLDER", ""),
            telegram_enabled=_env_bool("MEMETALK_TELEGRAM_ENABLED", False),
            telegram_bot_token=os.getenv("MEMETALK_TELEGRAM_BOT_TOKEN"),
        )

    def ensure_runtime_dirs(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
