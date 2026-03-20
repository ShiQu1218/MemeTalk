from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from memetalk.core.models import (
    EmbeddingDocument,
    IndexErrorRecord,
    IndexRunSummary,
    IndexWarningRecord,
    MemeAsset,
    OCRExtraction,
    OCRStatus,
    compose_embedding_text,
    compose_reply_embedding_text,
    utc_now,
)
from memetalk.core.providers import ProviderBundle
from memetalk.core.retrieval import build_index_version, normalize_template_fields
from memetalk.storage.sqlite_store import SQLiteMemeRepository
from memetalk.storage.vector_store import VectorStore

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(slots=True)
class IndexProgress:
    total: int
    current: int
    file_name: str
    step: str  # "scan", "sha256", "skip", "ocr", "metadata", "embedding", "store", "done", "error"
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    warnings: int = 0
    error_records: list[IndexErrorRecord] = field(default_factory=list)
    warning_records: list[IndexWarningRecord] = field(default_factory=list)


ProgressCallback = Callable[[IndexProgress], None]


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

    def build_index(
        self,
        source_dir: Path,
        reindex: bool = False,
        on_progress: ProgressCallback | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> IndexRunSummary:
        run = IndexRunSummary(
            run_id=str(uuid4()),
            source_dir=str(source_dir.resolve()),
            reindex=reindex,
            status="running",
        )
        image_paths = sorted(
            path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        total = len(image_paths)

        def _report(file_name: str, step: str) -> None:
            if on_progress is not None:
                on_progress(
                    IndexProgress(
                        total=total,
                        current=run.processed_count,
                        file_name=file_name,
                        step=step,
                        indexed=run.indexed_count,
                        skipped=run.skipped_count,
                        failed=run.failed_count,
                        warnings=run.warning_count,
                        error_records=run.errors,
                        warning_records=run.warnings,
                    )
                )

        _report("", "scan")

        for image_path in image_paths:
            if cancel_check is not None and cancel_check():
                run.status = "cancelled"
                run.completed_at = utc_now()
                self.repository.save_index_run(run)
                return run

            run.processed_count += 1
            name = image_path.name

            _report(name, "sha256")
            file_sha256 = _sha256_file(image_path)

            if not reindex and self.repository.get_asset_by_sha256(file_sha256):
                run.skipped_count += 1
                _report(name, "skip")
                continue
            try:
                _report(name, "ocr")
                ocr_result = self._extract_text(image_path, run)

                _report(name, "metadata")
                metadata = self.providers.metadata_provider.analyze_image(image_path, ocr_result)
                metadata.has_text = ocr_result.has_text
                metadata.ocr_text = ocr_result.text
                metadata.ocr_status = ocr_result.status
                metadata.ocr_confidence = ocr_result.confidence
                metadata.ocr_lines = ocr_result.raw_lines
                (
                    metadata.template_canonical_id,
                    metadata.template_aliases,
                    metadata.template_family,
                ) = normalize_template_fields(metadata.template_name)

                _report(name, "embedding")
                metadata.embedding_text = compose_embedding_text(metadata)
                reply_embedding_text = compose_reply_embedding_text(metadata)
                semantic_embedding, reply_embedding = self.providers.embedding_provider.embed_texts(
                    [metadata.embedding_text, reply_embedding_text]
                )

                semantic_index_version = build_index_version(
                    self.providers.embedding_provider.index_identity(),
                    len(semantic_embedding),
                    "semantic",
                )
                reply_index_version = build_index_version(
                    self.providers.embedding_provider.index_identity(),
                    len(reply_embedding),
                    "reply_text",
                )

                _report(name, "store")
                asset = MemeAsset(
                    image_id=file_sha256[:16],
                    file_path=str(image_path.resolve()),
                    file_sha256=file_sha256,
                    metadata=metadata,
                )
                self.repository.upsert_asset(asset)
                base_metadata = {
                    "file_path": asset.file_path,
                    "template_name": metadata.template_name or "",
                    "template_canonical_id": metadata.template_canonical_id,
                    "emotion_tags": ",".join(metadata.emotion_tags),
                    "intent_tags": ",".join(metadata.intent_tags),
                }
                self.vector_store.upsert(
                    [
                        EmbeddingDocument(
                            document_id=f"{asset.image_id}:semantic",
                            text=metadata.embedding_text,
                            vector=semantic_embedding,
                            metadata={
                                **base_metadata,
                                "search_mode": "semantic",
                                "channel": "semantic",
                                "index_version": semantic_index_version,
                            },
                        ),
                        EmbeddingDocument(
                            document_id=f"{asset.image_id}:reply_text",
                            text=reply_embedding_text,
                            vector=reply_embedding,
                            metadata={
                                **base_metadata,
                                "search_mode": "reply",
                                "channel": "reply_text",
                                "index_version": reply_index_version,
                            },
                        ),
                    ]
                )
                run.indexed_count += 1
                _report(name, "done")
            except Exception as exc:
                run.failed_count += 1
                run.errors.append(IndexErrorRecord(file_path=str(image_path.resolve()), error=str(exc)))
                _report(name, "error")
        run.completed_at = utc_now()
        if run.failed_count:
            run.status = "completed_with_errors"
        elif run.warning_count:
            run.status = "completed_with_warnings"
        else:
            run.status = "completed"
        self.repository.save_index_run(run)
        return run

    def _extract_text(self, image_path: Path, run: IndexRunSummary) -> OCRExtraction:
        try:
            result = self.providers.ocr_provider.extract_text(image_path)
        except Exception as exc:
            run.warning_count += 1
            run.warnings.append(
                IndexWarningRecord(
                    file_path=str(image_path.resolve()),
                    warning=str(exc),
                    stage="ocr",
                )
            )
            return OCRExtraction(status=OCRStatus.FAILED, error=str(exc))
        if result.status == OCRStatus.FAILED:
            run.warning_count += 1
            run.warnings.append(
                IndexWarningRecord(
                    file_path=str(image_path.resolve()),
                    warning=result.error or "OCR returned failed status.",
                    stage="ocr",
                )
            )
            return result
        return result
