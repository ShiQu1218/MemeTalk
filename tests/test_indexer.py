from pathlib import Path

from PIL import Image

from memetalk.app.indexer import IndexingService, _sha256_file
from memetalk.app.container import build_container
from memetalk.config import AppSettings
from memetalk.core.models import EmbeddingDocument, MemeAsset, compose_embedding_text
from memetalk.core.providers import ProviderBundle
from memetalk.core.retrieval import build_index_version, normalize_template_fields
from memetalk.providers.mock import (
    MockEmbeddingProvider,
    MockMetadataProvider,
    MockOCRProvider,
    MockQueryAnalyzer,
    MockReranker,
)
from memetalk.storage.sqlite_store import SQLiteMemeRepository
from memetalk.storage.vector_store import InMemoryVectorStore


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


def _build_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        sqlite_path=tmp_path / "memetalk.sqlite3",
        vector_backend="memory",
        provider_backend="mock",
        ocr_backend="mock",
        api_base_url="http://testserver",
    )


def _build_provider_bundle(dimensions: int = 24) -> ProviderBundle:
    return ProviderBundle(
        ocr_provider=MockOCRProvider(),
        metadata_provider=MockMetadataProvider(),
        embedding_provider=MockEmbeddingProvider(dimensions=dimensions),
        query_analyzer=MockQueryAnalyzer(),
        reranker=MockReranker(),
    )


class FlakyInMemoryVectorStore(InMemoryVectorStore):
    def __init__(self, failures_remaining: int = 1) -> None:
        super().__init__()
        self.failures_remaining = failures_remaining

    def upsert(self, documents: list[EmbeddingDocument]) -> None:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("Vector store forced failure.")
        super().upsert(documents)


def test_index_build_tracks_ocr_success_empty_and_failed_statuses(tmp_path: Path) -> None:
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    _create_image(source_dir / "friend_text.png", (255, 0, 0))
    _create_image(source_dir / "reaction_face.png", (0, 255, 0))
    _create_image(source_dir / "caption_ocrfail.png", (0, 0, 255))

    container = build_container(_build_settings(tmp_path))
    summary = container.indexing_service.build_index(source_dir)

    success_asset = container.repository.get_asset_by_sha256(_sha256_file(source_dir / "friend_text.png"))
    empty_asset = container.repository.get_asset_by_sha256(_sha256_file(source_dir / "reaction_face.png"))
    failed_asset = container.repository.get_asset_by_sha256(_sha256_file(source_dir / "caption_ocrfail.png"))

    assert summary.indexed_count == 3
    assert summary.failed_count == 0
    assert summary.warning_count == 1
    assert success_asset is not None and success_asset.metadata.ocr_status.value == "success"
    assert empty_asset is not None and empty_asset.metadata.ocr_status.value == "empty"
    assert failed_asset is not None and failed_asset.metadata.ocr_status.value == "failed"
    assert summary.warnings[0].stage == "ocr"


def test_index_build_skips_unchanged_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    _create_image(source_dir / "friend_text.png", (255, 0, 0))
    _create_image(source_dir / "reaction_face.png", (0, 255, 0))

    container = build_container(_build_settings(tmp_path))

    first_run = container.indexing_service.build_index(source_dir)
    second_run = container.indexing_service.build_index(source_dir)

    assert first_run.indexed_count == 2
    assert second_run.skipped_count == 2


def test_index_build_records_failures_without_aborting_batch(tmp_path: Path) -> None:
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    _create_image(source_dir / "ok_text.png", (255, 0, 0))
    _create_image(source_dir / "broken_fail.png", (0, 255, 0))
    _create_image(source_dir / "ok_face.png", (0, 0, 255))

    container = build_container(_build_settings(tmp_path))
    summary = container.indexing_service.build_index(source_dir)

    assert summary.processed_count == 3
    assert summary.indexed_count == 2
    assert summary.failed_count == 1
    assert len(summary.errors) == 1


def test_index_build_retries_when_previous_store_write_failed(tmp_path: Path) -> None:
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    image_path = source_dir / "friend_text.png"
    _create_image(image_path, (255, 0, 0))

    repository = SQLiteMemeRepository(tmp_path / "memetalk.sqlite3")
    repository.initialize()
    vector_store = FlakyInMemoryVectorStore()
    providers = _build_provider_bundle()
    service = IndexingService(repository, vector_store, providers)

    first_run = service.build_index(source_dir)
    assert first_run.failed_count == 1
    assert repository.get_asset_by_sha256(_sha256_file(image_path)) is None

    second_run = service.build_index(source_dir)

    assert second_run.indexed_count == 1
    assert second_run.skipped_count == 0


def test_index_build_retries_asset_when_vector_documents_are_missing(tmp_path: Path) -> None:
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    image_path = source_dir / "friend_text.png"
    _create_image(image_path, (255, 0, 0))

    repository = SQLiteMemeRepository(tmp_path / "memetalk.sqlite3")
    repository.initialize()
    vector_store = InMemoryVectorStore()
    providers = _build_provider_bundle()
    service = IndexingService(repository, vector_store, providers)

    metadata = providers.metadata_provider.analyze_image(image_path)
    (
        metadata.template_canonical_id,
        metadata.template_aliases,
        metadata.template_family,
    ) = normalize_template_fields(metadata.template_name)
    metadata.embedding_text = compose_embedding_text(metadata)
    probe_dimension = providers.embedding_provider.embedding_dimensions()
    semantic_index_version = build_index_version(
        providers.embedding_provider.index_identity(),
        probe_dimension,
        "semantic",
    )
    reply_index_version = build_index_version(
        providers.embedding_provider.index_identity(),
        probe_dimension,
        "reply_text",
    )
    repository.upsert_asset(
        MemeAsset(
            image_id=_sha256_file(image_path)[:16],
            file_path=str(image_path.resolve()),
            file_sha256=_sha256_file(image_path),
            metadata=metadata,
            index_status="ready",
            semantic_index_version=semantic_index_version,
            reply_index_version=reply_index_version,
        )
    )

    summary = service.build_index(source_dir)

    assert summary.indexed_count == 1
    assert summary.skipped_count == 0
    assert vector_store.has_document(f"{_sha256_file(image_path)[:16]}:semantic", semantic_index_version)
    assert vector_store.has_document(f"{_sha256_file(image_path)[:16]}:reply_text", reply_index_version)


def test_index_build_retries_asset_when_embedding_version_changes(tmp_path: Path) -> None:
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    image_path = source_dir / "friend_text.png"
    _create_image(image_path, (255, 0, 0))

    repository = SQLiteMemeRepository(tmp_path / "memetalk.sqlite3")
    repository.initialize()
    vector_store = InMemoryVectorStore()

    first_service = IndexingService(repository, vector_store, _build_provider_bundle(dimensions=24))
    second_service = IndexingService(repository, vector_store, _build_provider_bundle(dimensions=12))

    first_run = first_service.build_index(source_dir)
    second_run = second_service.build_index(source_dir)
    asset = repository.get_asset_by_sha256(_sha256_file(image_path))

    assert first_run.indexed_count == 1
    assert second_run.indexed_count == 1
    assert second_run.skipped_count == 0
    assert asset is not None
    assert asset.semantic_index_version == build_index_version(
        second_service.providers.embedding_provider.index_identity(),
        12,
        "semantic",
    )
