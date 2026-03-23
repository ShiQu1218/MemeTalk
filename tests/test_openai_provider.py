import sys
from types import ModuleType, SimpleNamespace

from PIL import Image

from memetalk.config import AppSettings
from memetalk.core.models import MemeMetadata, OCRStatus, QueryAnalysis, RerankCandidate, SearchMode
from memetalk.providers.openai_provider import (
    CompatibleEmbeddingProvider,
    CompatibleQueryAnalyzer,
    CompatibleReranker,
    _build_image_data_url,
    _extract_json_object,
    build_lmstudio_profile,
)


def test_extract_json_object_accepts_code_fence() -> None:
    payload = """```json
    {"hello": "world", "items": [1, 2, 3]}
    ```"""

    parsed = _extract_json_object(payload)

    assert parsed["hello"] == "world"
    assert parsed["items"] == [1, 2, 3]


def test_extract_json_object_wraps_top_level_array_for_rerank_results() -> None:
    payload = """排序如下：
    [
      {"image_id": "meme-1", "score": 0.93, "reason": "文字夠嗆很適合回覆"}
    ]"""

    parsed = _extract_json_object(payload, array_field="results")

    assert parsed == {
        "results": [
            {
                "image_id": "meme-1",
                "score": 0.93,
                "reason": "文字夠嗆很適合回覆",
            }
        ]
    }


def test_lmstudio_webp_payload_is_transcoded_to_png(tmp_path) -> None:
    image_path = tmp_path / "sample.webp"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path, format="PNG")

    payload = _build_image_data_url(image_path, "lmstudio")

    assert payload.startswith("data:image/png;base64,")


def test_lmstudio_query_analyzer_uses_compatible_client(monkeypatch) -> None:
    class DummyOpenAI:
        init_kwargs: dict[str, str] | None = None
        create_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs):
            DummyOpenAI.init_kwargs = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            DummyOpenAI.create_kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "```json\n"
                                '{"situation":"主管突然改需求","emotions":["厭世"],'
                                '"tone":"吐槽型回覆","reply_intent":"抱怨",'
                                '"preferred_tone":"陰陽怪氣",'
                                '"query_embedding_text":"主管突然改需求 厭世 吐槽"}\n'
                                "```"
                            )
                        )
                    )
                ]
            )

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = DummyOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    settings = AppSettings(provider_backend="lmstudio", lmstudio_chat_model="local-chat-model")
    analyzer = CompatibleQueryAnalyzer(build_lmstudio_profile(settings))

    result = analyzer.analyze_query("主管又改需求", preferred_tone="陰陽怪氣")

    assert result.situation == "主管突然改需求"
    assert result.emotions == ["厭世"]
    assert result.reply_intent == "抱怨"
    assert result.preferred_tone == "陰陽怪氣"
    assert DummyOpenAI.init_kwargs == {"api_key": "lm-studio", "base_url": "http://127.0.0.1:1234/v1"}
    assert DummyOpenAI.create_kwargs is not None
    assert DummyOpenAI.create_kwargs["model"] == "local-chat-model"
    assert DummyOpenAI.create_kwargs["temperature"] == 0
    assert "response_format" not in DummyOpenAI.create_kwargs
    assert '"preferred_tone": "陰陽怪氣"' in DummyOpenAI.create_kwargs["messages"][1]["content"]


def test_lmstudio_embedding_error_is_actionable(monkeypatch) -> None:
    class DummyOpenAI:
        def __init__(self, **kwargs):
            self.embeddings = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            raise RuntimeError(
                "Error code: 400 - {'error': \"No models loaded. Please load a model in the developer page or use the 'lms load' command.\"}"
            )

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = DummyOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    settings = AppSettings(provider_backend="lmstudio", lmstudio_embedding_model="local-embed-model")
    provider = CompatibleEmbeddingProvider(build_lmstudio_profile(settings))

    try:
        provider.embed_texts(["hello"])
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "embedding model is loaded" in str(exc)
        assert "MEMETALK_LMSTUDIO_EMBEDDING_MODEL" in str(exc)


def test_query_analyzer_retries_on_malformed_json(monkeypatch) -> None:
    class DummyOpenAI:
        call_count = 0

        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            DummyOpenAI.call_count += 1
            if DummyOpenAI.call_count == 1:
                content = (
                    "{\n"
                    '  "situation": "主管突然改需求"\n'
                    '  "emotions": ["厭世"],\n'
                    '  "tone": "吐槽型回覆",\n'
                    '  "reply_intent": "抱怨",\n'
                    '  "query_embedding_text": "主管突然改需求 厭世 吐槽"\n'
                    "}"
                )
            else:
                content = (
                    '{'
                    '"situation":"主管突然改需求",'
                    '"emotions":["厭世"],'
                    '"tone":"吐槽型回覆",'
                    '"reply_intent":"抱怨",'
                    '"query_embedding_text":"主管突然改需求 厭世 吐槽"'
                    '}'
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = DummyOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    settings = AppSettings(provider_backend="lmstudio", lmstudio_chat_model="local-chat-model")
    analyzer = CompatibleQueryAnalyzer(build_lmstudio_profile(settings))

    result = analyzer.analyze_query("主管又改需求")

    assert DummyOpenAI.call_count == 2
    assert result.reply_intent == "抱怨"


def test_reranker_repairs_top_level_array_response(monkeypatch) -> None:
    class DummyOpenAI:
        call_count = 0

        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            DummyOpenAI.call_count += 1
            content = (
                "以下是排序結果：\n"
                "[\n"
                '  {"image_id": "meme-1", "score": 0.91, "reason": "台詞夠酸，回覆感很強"}\n'
                "]"
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = DummyOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    settings = AppSettings(provider_backend="lmstudio", lmstudio_chat_model="local-chat-model")
    reranker = CompatibleReranker(build_lmstudio_profile(settings))
    query_analysis = QueryAnalysis(
        original_query="主管又改需求",
        situation="主管又改需求",
        emotions=["厭世"],
        tone="吐槽型回覆",
        reply_intent="抱怨",
        query_embedding_text="適合拿來抱怨主管改需求的回覆梗圖",
    )
    candidates = [
        RerankCandidate(
            image_id="meme-1",
            vector_score=0.75,
            file_path="D:/Develop/MemeTalk/tests/data/meme-1.png",
            metadata=MemeMetadata(
                has_text=True,
                ocr_text="又改？",
                ocr_status=OCRStatus.SUCCESS,
                ocr_lines=["又改？"],
                template_name="Office Reaction",
                scene_description="角色露出不耐煩表情",
                meme_usage="拿來吐槽反覆變更需求",
                emotion_tags=["厭世"],
                intent_tags=["抱怨"],
            ),
            candidate_sources=["reply_text"],
            feature_scores={"ocr_overlap": 0.8},
            deterministic_score=0.88,
        )
    ]

    result = reranker.rerank("主管又改需求", query_analysis, candidates, top_n=1, mode=SearchMode.REPLY)

    assert DummyOpenAI.call_count == 1
    assert len(result) == 1
    assert result[0].image_id == "meme-1"
    assert result[0].score == 0.91
    assert result[0].reason == "台詞夠酸，回覆感很強"
