from memetalk.core.models import MemeMetadata, SearchRequest, compose_embedding_text


def test_compose_embedding_text_uses_required_order() -> None:
    metadata = MemeMetadata(
        has_text=True,
        ocr_text="我快到了",
        template_name="Distracted Boyfriend",
        scene_description="男主角回頭看別人，女友一臉傻眼。",
        meme_usage="適合用來表達注意力跑掉或臨陣倒戈。",
        emotion_tags=["傻眼", "無奈"],
        intent_tags=["吐槽", "打臉"],
        style_tags=["誇張", "冷幽默"],
    )

    embedding_text = compose_embedding_text(metadata)

    assert embedding_text.splitlines() == [
        "核心文字：我快到了",
        "模板資訊：Distracted Boyfriend",
        "畫面描述：男主角回頭看別人，女友一臉傻眼。",
        "常見用途：適合用來表達注意力跑掉或臨陣倒戈。",
        "OCR 文字：我快到了",
        "情緒標籤：傻眼、無奈",
        "意圖標籤：吐槽、打臉",
        "風格標籤：誇張、冷幽默",
    ]


def test_search_request_rejects_blank_query() -> None:
    try:
        SearchRequest(query="   ")
    except ValueError as exc:
        assert "blank" in str(exc).lower()
    else:
        raise AssertionError("Blank queries must fail validation.")
