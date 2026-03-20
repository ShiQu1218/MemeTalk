from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    REPLY = "reply"


class OCRStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


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
    status: OCRStatus = OCRStatus.EMPTY
    confidence: float | None = None
    error: str | None = None


class MemeMetadata(BaseModel):
    has_text: bool = False
    ocr_text: str = ""
    ocr_status: OCRStatus = OCRStatus.EMPTY
    ocr_confidence: float | None = None
    ocr_lines: list[str] = Field(default_factory=list)
    template_name: str | None = None
    template_canonical_id: str = ""
    template_aliases: list[str] = Field(default_factory=list)
    template_family: str = ""
    scene_description: str
    meme_usage: str
    emotion_tags: list[str] = Field(default_factory=list)
    intent_tags: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    embedding_text: str = ""

    @field_validator("emotion_tags", "intent_tags", "style_tags", "template_aliases")
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


def compose_reply_embedding_text(metadata: MemeMetadata) -> str:
    if metadata.ocr_text:
        parts = [
            f"梗圖文字：{metadata.ocr_text}",
            f"梗圖文字（重複強調）：{metadata.ocr_text}",
            f"語氣用途：{metadata.meme_usage}",
            "回覆意圖：" + ("、".join(metadata.intent_tags) if metadata.intent_tags else "一般回覆"),
        ]
        return "\n".join(parts)
    parts = [
        f"回覆用途：{metadata.meme_usage}",
        f"模板資訊：{metadata.template_name or '未知模板'}",
        "情緒標籤：" + ("、".join(metadata.emotion_tags) if metadata.emotion_tags else "無"),
        "回覆意圖：" + ("、".join(metadata.intent_tags) if metadata.intent_tags else "一般回覆"),
        "OCR 狀態：無可用台詞",
    ]
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


class RetrievalWeights(BaseModel):
    semantic: float = 1.0
    reply_text: float = 1.0
    keyword: float = 1.0
    template: float = 1.0


class QueryAnalysis(BaseModel):
    original_query: str
    situation: str
    emotions: list[str] = Field(default_factory=list)
    tone: str
    reply_intent: str
    query_embedding_text: str
    query_terms: list[str] = Field(default_factory=list)
    template_hints: list[str] = Field(default_factory=list)
    retrieval_weights: RetrievalWeights = Field(default_factory=RetrievalWeights)

    @field_validator("emotions", "query_terms", "template_hints", mode="before")
    @classmethod
    def normalize_list_values(cls, values: list[str] | str) -> list[str]:
        if isinstance(values, str):
            delimiter = "、" if "、" in values else ","
            values = [v.strip() for v in values.split(delimiter) if v.strip()]
        return _clean_tags(values)


class SearchMatch(BaseModel):
    image_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    channel: str | None = None
    index_version: str | None = None


class RerankCandidate(BaseModel):
    image_id: str
    vector_score: float
    file_path: str
    metadata: MemeMetadata
    candidate_sources: list[str] = Field(default_factory=list)
    feature_scores: dict[str, float] = Field(default_factory=dict)
    degradation_flags: list[str] = Field(default_factory=list)
    deterministic_score: float = 0.0


class RerankResult(BaseModel):
    image_id: str
    score: float
    reason: str


class SearchResultDebug(BaseModel):
    candidate_sources: list[str] = Field(default_factory=list)
    degradation_flags: list[str] = Field(default_factory=list)
    feature_scores: dict[str, float] = Field(default_factory=dict)


class SearchResult(BaseModel):
    image_id: str
    image_url: str
    reason: str
    score: float
    template_name: str | None = None
    emotion_tags: list[str] = Field(default_factory=list)
    intent_tags: list[str] = Field(default_factory=list)
    debug: SearchResultDebug = Field(default_factory=SearchResultDebug)


class SearchTrace(BaseModel):
    routes_used: list[str] = Field(default_factory=list)
    degraded_routes: list[str] = Field(default_factory=list)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    rerank_strategy: str = ""
    index_versions: dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query_analysis: QueryAnalysis
    results: list[SearchResult]
    provider_trace: dict[str, str]
    search_trace: SearchTrace


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_n: int = Field(default=3, ge=1, le=5)
    candidate_k: int = Field(default=15, ge=1, le=30)
    mode: SearchMode = SearchMode.REPLY

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


class IndexWarningRecord(BaseModel):
    file_path: str
    warning: str
    stage: str


class IndexRunSummary(BaseModel):
    run_id: str
    source_dir: str
    reindex: bool
    status: str
    processed_count: int = 0
    indexed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    errors: list[IndexErrorRecord] = Field(default_factory=list)
    warnings: list[IndexWarningRecord] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
