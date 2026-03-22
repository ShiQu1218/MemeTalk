from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from memetalk.core.models import OCRExtraction, QueryAnalysis, RerankCandidate, RerankResult, SearchMode


class OCRProvider(ABC):
    name: str

    @abstractmethod
    def extract_text(self, image_path: Path) -> OCRExtraction:
        raise NotImplementedError


class MetadataProvider(ABC):
    name: str

    @abstractmethod
    def analyze_image(self, image_path: Path, ocr_hint: OCRExtraction | None = None):
        raise NotImplementedError


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def index_identity(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embedding_dimensions(self) -> int | None:
        raise NotImplementedError


class QueryAnalyzer(ABC):
    name: str

    @abstractmethod
    def analyze_query(
        self,
        query: str,
        mode: SearchMode = SearchMode.REPLY,
        preferred_tone: str | None = None,
    ) -> QueryAnalysis:
        raise NotImplementedError


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> list[RerankResult]:
        raise NotImplementedError


@dataclass(slots=True)
class ProviderBundle:
    ocr_provider: OCRProvider
    metadata_provider: MetadataProvider
    embedding_provider: EmbeddingProvider
    query_analyzer: QueryAnalyzer
    reranker: Reranker

    def trace(self) -> dict[str, str]:
        return {
            "ocr": self.ocr_provider.name,
            "metadata": self.metadata_provider.name,
            "embedding": self.embedding_provider.name,
            "query_analyzer": self.query_analyzer.name,
            "reranker": self.reranker.name,
        }
