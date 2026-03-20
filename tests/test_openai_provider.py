import sys
from types import ModuleType, SimpleNamespace

from PIL import Image

from memetalk.config import AppSettings
from memetalk.providers.openai_provider import (
    CompatibleQueryAnalyzer,
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

    result = analyzer.analyze_query("主管又改需求")

    assert result.situation == "主管突然改需求"
    assert result.emotions == ["厭世"]
    assert result.reply_intent == "抱怨"
    assert DummyOpenAI.init_kwargs == {"api_key": "lm-studio", "base_url": "http://127.0.0.1:1234/v1"}
    assert DummyOpenAI.create_kwargs is not None
    assert DummyOpenAI.create_kwargs["model"] == "local-chat-model"
    assert "response_format" not in DummyOpenAI.create_kwargs
