from memetalk.core.models import MemeMetadata, OCRStatus, SearchRequest, compose_embedding_text, compose_reply_embedding_text
from memetalk.core.retrieval import normalize_template_fields


def test_compose_embedding_text_uses_required_order() -> None:
    metadata = MemeMetadata(
        has_text=True,
        ocr_text="我快到了",
        ocr_status=OCRStatus.SUCCESS,
        ocr_confidence=0.95,
        ocr_lines=["我快到了"],
        template_name="Distracted Boyfriend",
        scene_description="男主角回頭看別人，女友一臉傻眼。",
        meme_usage="適合用來表達注意力跑掉或臨陣倒戈。",
        visual_description="左側男子回頭望向經過的女子，右側女友露出不滿表情，三人構圖形成對比。",
        aesthetic_tags=["對比構圖", "表情誇張", "三人構圖"],
        usage_scenario="適合在朋友見異思遷或注意力不集中時使用。",
        emotion_tags=["傻眼", "無奈"],
        intent_tags=["吐槽", "打臉"],
        style_tags=["誇張", "冷幽默"],
    )

    embedding_text = compose_embedding_text(metadata)

    assert embedding_text.splitlines() == [
        "核心文字：我快到了",
        "模板資訊：Distracted Boyfriend",
        "視覺描述：左側男子回頭望向經過的女子，右側女友露出不滿表情，三人構圖形成對比。",
        "畫面描述：男主角回頭看別人，女友一臉傻眼。",
        "常見用途：適合用來表達注意力跑掉或臨陣倒戈。",
        "使用情境：適合在朋友見異思遷或注意力不集中時使用。",
        "OCR 文字：我快到了",
        "情緒標籤：傻眼、無奈",
        "意圖標籤：吐槽、打臉",
        "風格標籤：誇張、冷幽默",
        "視覺風格：對比構圖、表情誇張、三人構圖",
    ]


def test_reply_embedding_without_ocr_does_not_fallback_to_full_embedding_text() -> None:
    metadata = MemeMetadata(
        has_text=False,
        ocr_text="",
        ocr_status=OCRStatus.EMPTY,
        template_name="anime reaction",
        scene_description="角色露出震驚表情。",
        meme_usage="適合用來當作敷衍回覆。",
        emotion_tags=["傻眼"],
        intent_tags=["吐槽"],
        style_tags=["網路梗"],
    )

    reply_embedding = compose_reply_embedding_text(metadata)
    semantic_embedding = compose_embedding_text(metadata)

    assert reply_embedding != semantic_embedding
    assert "OCR 狀態：無可用台詞" in reply_embedding


def test_template_normalization_groups_aliases_to_same_canonical_id() -> None:
    left = normalize_template_fields("AnimeReaction")
    right = normalize_template_fields("anime_reaction")

    assert left[0] == "anime-reaction"
    assert right[0] == "anime-reaction"
    assert left[2] == "anime"
    assert right[2] == "anime"


def test_search_request_rejects_missing_text_and_image() -> None:
    try:
        SearchRequest(query="   ")
    except ValueError as exc:
        assert "text query or query image" in str(exc)
    else:
        raise AssertionError("Search requests without text or image must fail validation.")


def test_search_request_allows_image_only_input() -> None:
    request = SearchRequest(query="   ", query_image_base64="ZGF0YQ==")

    assert request.query is None
    assert request.query_image_base64 == "ZGF0YQ=="


def test_search_request_trims_optional_preferred_tone() -> None:
    request = SearchRequest(query="主管又改需求", preferred_tone="  陰陽怪氣  ")

    assert request.preferred_tone == "陰陽怪氣"
