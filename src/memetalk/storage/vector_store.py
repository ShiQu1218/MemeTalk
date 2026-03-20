from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

from memetalk.core.models import EmbeddingDocument, SearchMatch


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, documents: list[EmbeddingDocument]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, vector: list[float], top_k: int) -> list[SearchMatch]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._documents: dict[str, EmbeddingDocument] = {}

    def upsert(self, documents: list[EmbeddingDocument]) -> None:
        for document in documents:
            self._documents[document.document_id] = document

    def query(self, vector: list[float], top_k: int) -> list[SearchMatch]:
        scored = [
            SearchMatch(
                image_id=document.document_id,
                score=_cosine_similarity(vector, document.vector),
                metadata=document.metadata,
            )
            for document in self._documents.values()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("The chroma extra is not installed. Install with `pip install -e .[chroma]`.") from exc
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        return self._collection

    def upsert(self, documents: list[EmbeddingDocument]) -> None:
        collection = self._get_collection()
        collection.upsert(
            ids=[document.document_id for document in documents],
            documents=[document.text for document in documents],
            embeddings=[document.vector for document in documents],
            metadatas=[document.metadata for document in documents],
        )

    def query(self, vector: list[float], top_k: int) -> list[SearchMatch]:
        collection = self._get_collection()
        response = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["distances", "metadatas"],
        )
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        matches: list[SearchMatch] = []
        for image_id, distance, metadata in zip(ids, distances, metadatas, strict=False):
            score = 1.0 - float(distance or 0.0)
            matches.append(SearchMatch(image_id=image_id, score=score, metadata=metadata or {}))
        return matches
