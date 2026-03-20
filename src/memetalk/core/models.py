from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def _clean_tags(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        tag = value.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        result.append(tag)
    return result


class OCRExtraction(BaseModel):
    text: str = ""
    has_text: bool = False
    raw_lines: list[str] = Field(default_factory=list)


class MemeMetadata(BaseModel):
    has_text: bool = False
    ocr_text: str = ""
    template_name: str | None = None
    scene_description: str
    meme_usage: str
    emotion_tags: list[str] = Field(default_factory=list)
    intent_tags: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    embedding_text: str = ""

    @field_validator("emotion_tags", "intent_tags", "style_tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        return _clean_tags(values)

    @field_validator("ocr_text", mode="before")
    @classmethod
    def normalize_ocr_text(cls, value: str | None) -> str:
        return (value or "").strip()


def compose_embedding_text(metadata: MemeMetadata) -> str:
    parts = []
    if metadata.ocr_text:
        parts.append(f"核心文字：{metadata.ocr_text}")
    parts.extend([
        f"模板資訊：{metadata.template_name or '未知模板'}",
        f"畫面描述：{metadata.scene_description}",
        f"常見用途：{metadata.meme_usage}",
        f"OCR 文字：{metadata.ocr_text or '無文字'}",
        "情緒標籤：" + ("、".join(metadata.emotion_tags) if metadata.emotion_tags else "無"),
        "意圖標籤：" + ("、".join(metadata.intent_tags) if metadata.intent_tags else "無"),
        "風格標籤：" + ("、".join(metadata.style_tags) if metadata.style_tags else "無"),
    ])
    return "\n".join(parts)


class MemeAsset(BaseModel):
    image_id: str
    file_path: str
    file_sha256: str
    metadata: MemeMetadata
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EmbeddingDocument(BaseModel):
    document_id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def vector_must_not_be_empty(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("Embedding vectors must not be empty.")
        return value


class QueryAnalysis(BaseModel):
    original_query: str
    situation: str
    emotions: list[str] = Field(default_factory=list)
    tone: str
    reply_intent: str
    query_embedding_text: str

    @field_validator("emotions")
    @classmethod
    def normalize_emotions(cls, values: list[str]) -> list[str]:
        return _clean_tags(values)


class SearchMatch(BaseModel):
    image_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankCandidate(BaseModel):
    image_id: str
    vector_score: float
    file_path: str
    metadata: MemeMetadata


class RerankResult(BaseModel):
    image_id: str
    score: float
    reason: str


class SearchResult(BaseModel):
    image_id: str
    image_url: str
    reason: str
    score: float
    template_name: str | None = None
    emotion_tags: list[str] = Field(default_factory=list)
    intent_tags: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query_analysis: QueryAnalysis
    results: list[SearchResult]
    provider_trace: dict[str, str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_n: int = Field(default=3, ge=1, le=5)
    candidate_k: int = Field(default=8, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Query must not be blank.")
        return trimmed


class IndexErrorRecord(BaseModel):
    file_path: str
    error: str


class IndexRunSummary(BaseModel):
    run_id: str
    source_dir: str
    reindex: bool
    status: str
    processed_count: int = 0
    indexed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    errors: list[IndexErrorRecord] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
