from pathlib import Path

from PIL import Image

from memetalk.app.indexer import _sha256_file
from memetalk.app.container import build_container
from memetalk.config import AppSettings


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
