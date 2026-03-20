from pathlib import Path

from memetalk.core.models import MemeAsset, MemeMetadata, OCRStatus
from memetalk.core.retrieval import normalize_template_fields
from memetalk.storage.sqlite_store import SQLiteMemeRepository


def _build_asset(
    image_id: str,
    file_sha256: str,
    template_name: str,
    ocr_text: str,
    meme_usage: str,
) -> MemeAsset:
    template_canonical_id, template_aliases, template_family = normalize_template_fields(template_name)
    metadata = MemeMetadata(
        has_text=bool(ocr_text),
        ocr_text=ocr_text,
        ocr_status=OCRStatus.SUCCESS if ocr_text else OCRStatus.EMPTY,
        ocr_confidence=0.98 if ocr_text else 0.0,
        ocr_lines=[ocr_text] if ocr_text else [],
        template_name=template_name,
        template_canonical_id=template_canonical_id,
        template_aliases=template_aliases,
        template_family=template_family,
        scene_description=f"{template_name} scene",
        meme_usage=meme_usage,
        emotion_tags=["傻眼"],
        intent_tags=["吐槽"],
        style_tags=["冷幽默"],
        embedding_text=f"{template_name} {meme_usage} {ocr_text}",
    )
    return MemeAsset(
        image_id=image_id,
        file_path=str(Path("memes") / f"{image_id}.png"),
        file_sha256=file_sha256,
        metadata=metadata,
    )


def test_keyword_candidate_search_tracks_updated_template_text(tmp_path: Path) -> None:
    repository = SQLiteMemeRepository(tmp_path / "memetalk.sqlite3")
    repository.initialize()

    anime_asset = _build_asset(
        image_id="img-1",
        file_sha256="sha-1",
        template_name="AnimeReaction",
        ocr_text="我真的會笑死",
        meme_usage="適合拿來冷淡吐槽",
    )
    repository.upsert_asset(anime_asset)

    initial_matches = repository.search_keyword_candidates(["anime reaction"], ["anime reaction"], top_k=3)

    assert initial_matches
    assert initial_matches[0].image_id == "img-1"

    updated_asset = anime_asset.model_copy(
        update={
            "metadata": anime_asset.metadata.model_copy(
                update={
                    "template_name": "OfficeBurnout",
                    "template_canonical_id": "office-burnout",
                    "template_aliases": ["OfficeBurnout", "office burnout", "office_burnout"],
                    "template_family": "office",
                    "ocr_text": "今天真的不想上班",
                    "ocr_lines": ["今天真的不想上班"],
                    "meme_usage": "適合拿來回應職場崩潰",
                    "embedding_text": "OfficeBurnout 適合拿來回應職場崩潰 今天真的不想上班",
                }
            )
        }
    )
    repository.upsert_asset(updated_asset)

    old_matches = repository.search_keyword_candidates(["anime reaction"], ["anime reaction"], top_k=3)
    new_matches = repository.search_keyword_candidates(["office burnout"], ["office burnout"], top_k=3)

    assert not any(match.image_id == "img-1" for match in old_matches)
    assert new_matches
    assert new_matches[0].image_id == "img-1"
