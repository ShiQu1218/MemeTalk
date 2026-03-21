from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path

from memetalk.core.models import EmbeddingDocument, SearchMatch

logger = logging.getLogger(__name__)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / left_norm / right_norm


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, documents: list[EmbeddingDocument]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        search_mode: str | None = None,
        channel: str | None = None,
        index_version: str | None = None,
    ) -> list[SearchMatch]:
        raise NotImplementedError


def _extract_image_id(document_id: str) -> str:
    return document_id.split(":")[0]


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._documents: dict[str, EmbeddingDocument] = {}

    def upsert(self, documents: list[EmbeddingDocument]) -> None:
        for document in documents:
            self._documents[document.document_id] = document

    def query(
        self,
        vector: list[float],
        top_k: int,
        search_mode: str | None = None,
        channel: str | None = None,
        index_version: str | None = None,
    ) -> list[SearchMatch]:
        scored = [
            SearchMatch(
                image_id=_extract_image_id(document.document_id),
                score=_cosine_similarity(vector, document.vector),
                metadata=document.metadata,
                channel=document.metadata.get("channel"),
                index_version=document.metadata.get("index_version"),
            )
            for document in self._documents.values()
            if (search_mode is None or document.metadata.get("search_mode") == search_mode)
            and (channel is None or document.metadata.get("channel") == channel)
            and (index_version is None or document.metadata.get("index_version") == index_version)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._collections: dict[str, object] = {}

    def _collection_key(self, index_version: str) -> str:
        digest = hashlib.sha256(index_version.encode("utf-8")).hexdigest()[:12]
        return f"{self.collection_name}__{digest}"

    def _get_collection(self, index_version: str):
        collection_key = self._collection_key(index_version)
        if collection_key in self._collections:
            return self._collections[collection_key]
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("The chroma extra is not installed. Install with `pip install -e .[chroma]`.") from exc
        try:
            client = chromadb.PersistentClient(path=str(self.persist_directory))
        except (AttributeError, ValueError) as exc:
            raise RuntimeError(
                f"ChromaDB failed to initialise (chromadb {chromadb.__version__}, "
                f"Python {__import__('sys').version.split()[0]}). "
                "This is a known issue with Python 3.14 — consider using Python <=3.13."
            ) from exc
        collection = client.get_or_create_collection(
            name=collection_key,
            metadata={"hnsw:space": "cosine", "index_version": index_version},
        )
        self._collections[collection_key] = collection
        return collection

    def upsert(self, documents: list[EmbeddingDocument]) -> None:
        grouped: dict[str, list[EmbeddingDocument]] = {}
        for document in documents:
            index_version = str(document.metadata.get("index_version", "default"))
            grouped.setdefault(index_version, []).append(document)
        for index_version, grouped_documents in grouped.items():
            collection = self._get_collection(index_version)
            collection.upsert(
                ids=[document.document_id for document in grouped_documents],
                documents=[document.text for document in grouped_documents],
                embeddings=[document.vector for document in grouped_documents],
                metadatas=[document.metadata for document in grouped_documents],
            )

    def query(
        self,
        vector: list[float],
        top_k: int,
        search_mode: str | None = None,
        channel: str | None = None,
        index_version: str | None = None,
    ) -> list[SearchMatch]:
        if index_version is None:
            raise ValueError("index_version is required for Chroma queries.")
        collection = self._get_collection(index_version)
        query_kwargs: dict[str, object] = {
            "query_embeddings": [vector],
            "n_results": top_k,
            "include": ["distances", "metadatas"],
        }
        filters: dict[str, object] = {}
        if search_mode is not None:
            filters["search_mode"] = search_mode
        if channel is not None:
            filters["channel"] = channel
        if filters:
            query_kwargs["where"] = filters
        try:
            response = collection.query(**query_kwargs)
        except Exception as exc:
            error_message = str(exc)
            if "hnsw" in error_message.lower() or "Nothing found on disk" in error_message:
                logger.warning(
                    "ChromaDB HNSW index corrupted or missing for collection %s "
                    "(index_version=%s): %s. Returning empty results. "
                    "Re-index to rebuild the vector store.",
                    self._collection_key(index_version),
                    index_version,
                    error_message,
                )
                return []
            raise
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        matches: list[SearchMatch] = []
        for doc_id, distance, metadata in zip(ids, distances, metadatas, strict=False):
            score = 1.0 - float(distance or 0.0)
            metadata_dict = metadata or {}
            matches.append(
                SearchMatch(
                    image_id=_extract_image_id(doc_id),
                    score=score,
                    metadata=metadata_dict,
                    channel=metadata_dict.get("channel"),
                    index_version=metadata_dict.get("index_version"),
                )
            )
        return matches
