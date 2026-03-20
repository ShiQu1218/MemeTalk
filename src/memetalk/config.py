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


class AppSettings(BaseModel):
    sqlite_path: Path = Field(default_factory=lambda: Path("data/memetalk.sqlite3"))
    vector_backend: Literal["chroma", "memory"] = "chroma"
    chroma_path: Path = Field(default_factory=lambda: Path("data/chroma"))
    chroma_collection: str = "memetalk_memes"
    provider_backend: Literal["openai", "mock", "local", "lmstudio"] = "openai"
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
    search_candidate_k_default: int = 8
    search_top_n_default: int = 3

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
            search_candidate_k_default=_env_int("MEMETALK_SEARCH_CANDIDATE_K", 8),
            search_top_n_default=_env_int("MEMETALK_SEARCH_TOP_N", 3),
        )

    def ensure_runtime_dirs(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
