from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from memetalk.core.models import (
    EmbeddingDocument,
    IndexErrorRecord,
    IndexRunSummary,
    MemeAsset,
    OCRExtraction,
    compose_embedding_text,
    utc_now,
)
from memetalk.core.providers import ProviderBundle
from memetalk.storage.sqlite_store import SQLiteMemeRepository
from memetalk.storage.vector_store import VectorStore

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IndexingService:
    def __init__(
        self,
        repository: SQLiteMemeRepository,
        vector_store: VectorStore,
        providers: ProviderBundle,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.providers = providers

    def build_index(self, source_dir: Path, reindex: bool = False) -> IndexRunSummary:
        run = IndexRunSummary(
            run_id=str(uuid4()),
            source_dir=str(source_dir.resolve()),
            reindex=reindex,
            status="running",
        )
        image_paths = sorted(
            path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        for image_path in image_paths:
            run.processed_count += 1
            file_sha256 = _sha256_file(image_path)
            if not reindex and self.repository.get_asset_by_sha256(file_sha256):
                run.skipped_count += 1
                continue
            try:
                ocr_result = self._safe_extract_text(image_path)
                metadata = self.providers.metadata_provider.analyze_image(image_path, ocr_result)
                metadata.has_text = ocr_result.has_text
                metadata.ocr_text = ocr_result.text
                metadata.embedding_text = compose_embedding_text(metadata)
                embedding = self.providers.embedding_provider.embed_texts([metadata.embedding_text])[0]
                asset = MemeAsset(
                    image_id=file_sha256[:16],
                    file_path=str(image_path.resolve()),
                    file_sha256=file_sha256,
                    metadata=metadata,
                )
                self.repository.upsert_asset(asset)
                self.vector_store.upsert(
                    [
                        EmbeddingDocument(
                            document_id=asset.image_id,
                            text=metadata.embedding_text,
                            vector=embedding,
                            metadata={
                                "file_path": asset.file_path,
                                "template_name": metadata.template_name or "",
                                "emotion_tags": ",".join(metadata.emotion_tags),
                                "intent_tags": ",".join(metadata.intent_tags),
                            },
                        )
                    ]
                )
                run.indexed_count += 1
            except Exception as exc:
                run.failed_count += 1
                run.errors.append(IndexErrorRecord(file_path=str(image_path.resolve()), error=str(exc)))
        run.completed_at = utc_now()
        run.status = "completed_with_errors" if run.failed_count else "completed"
        self.repository.save_index_run(run)
        return run

    def _safe_extract_text(self, image_path: Path) -> OCRExtraction:
        try:
            return self.providers.ocr_provider.extract_text(image_path)
        except Exception:
            return OCRExtraction()
