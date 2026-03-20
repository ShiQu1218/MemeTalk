from pathlib import Path

import pytest

from memetalk.core.models import EmbeddingDocument
from memetalk.storage.vector_store import ChromaVectorStore, InMemoryVectorStore


chromadb = pytest.importorskip("chromadb")


def test_chroma_vector_store_upsert_and_query(tmp_path: Path) -> None:
    assert chromadb is not None
    store = ChromaVectorStore(tmp_path / "chroma", "test_collection")
    store.upsert(
        [
            EmbeddingDocument(
                document_id="asset-1:semantic",
                text="hello world",
                vector=[0.1, 0.2, 0.3],
                metadata={
                    "file_path": "x",
                    "template_name": "y",
                    "emotion_tags": "a",
                    "intent_tags": "b",
                    "channel": "semantic",
                    "index_version": "mock|dim=3|channel=semantic",
                },
            )
        ]
    )

    matches = store.query(
        [0.1, 0.2, 0.3],
        top_k=1,
        channel="semantic",
        index_version="mock|dim=3|channel=semantic",
    )

    assert len(matches) == 1
    assert matches[0].image_id == "asset-1"


def test_in_memory_vector_store_isolates_index_versions() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            EmbeddingDocument(
                document_id="asset-1:semantic",
                text="alpha",
                vector=[1.0, 0.0],
                metadata={"channel": "semantic", "index_version": "version-a"},
            ),
            EmbeddingDocument(
                document_id="asset-2:semantic",
                text="beta",
                vector=[0.0, 1.0],
                metadata={"channel": "semantic", "index_version": "version-b"},
            ),
        ]
    )

    matches_a = store.query([1.0, 0.0], top_k=5, channel="semantic", index_version="version-a")
    matches_b = store.query([0.0, 1.0], top_k=5, channel="semantic", index_version="version-b")

    assert [match.image_id for match in matches_a] == ["asset-1"]
    assert [match.image_id for match in matches_b] == ["asset-2"]
