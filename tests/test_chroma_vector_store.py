from pathlib import Path

import pytest

from memetalk.core.models import EmbeddingDocument
from memetalk.storage.vector_store import ChromaVectorStore


chromadb = pytest.importorskip("chromadb")


def test_chroma_vector_store_upsert_and_query(tmp_path: Path) -> None:
    assert chromadb is not None
    store = ChromaVectorStore(tmp_path / "chroma", "test_collection")
    store.upsert(
        [
            EmbeddingDocument(
                document_id="asset-1",
                text="hello world",
                vector=[0.1, 0.2, 0.3],
                metadata={
                    "file_path": "x",
                    "template_name": "y",
                    "emotion_tags": "a",
                    "intent_tags": "b",
                },
            )
        ]
    )

    matches = store.query([0.1, 0.2, 0.3], top_k=1)

    assert len(matches) == 1
    assert matches[0].image_id == "asset-1"
