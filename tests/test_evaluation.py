import json
from pathlib import Path

from PIL import Image

from memetalk.app.container import build_container
from memetalk.app.indexer import _sha256_file
from memetalk.config import AppSettings
from memetalk.core.models import SearchMode


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


def test_evaluation_service_reports_precision_and_mrr(tmp_path: Path) -> None:
    settings = AppSettings(
        sqlite_path=tmp_path / "memetalk.sqlite3",
        vector_backend="memory",
        provider_backend="mock",
        ocr_backend="mock",
        api_base_url="http://testserver",
    )
    container = build_container(settings)
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    target_path = source_dir / "friends_late_text.png"
    other_path = source_dir / "office_burnout.png"
    _create_image(target_path, (255, 0, 0))
    _create_image(other_path, (0, 255, 0))
    container.indexing_service.build_index(source_dir)

    target_id = container.repository.get_asset_by_sha256(_sha256_file(target_path)).image_id
    other_id = container.repository.get_asset_by_sha256(_sha256_file(other_path)).image_id

    cases_path = tmp_path / "eval_cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "reply-late",
                    "query": "朋友說快到了但還沒出門",
                    "mode": SearchMode.REPLY.value,
                    "positives": [target_id],
                    "hard_negatives": [other_id],
                    "top_k": 2,
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cases = container.evaluation_service.load_cases(cases_path)
    report = container.evaluation_service.run_cases(cases)

    assert report.case_count == 1
    assert report.precision_at_k > 0
    assert report.mrr == 1.0
    assert report.cases[0].case_id == "reply-late"
