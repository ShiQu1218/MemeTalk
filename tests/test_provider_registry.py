import pytest

from memetalk.config import AppSettings
from memetalk.providers.mock import UnsupportedLocalCapabilityError
from memetalk.providers.registry import build_default_registry


def test_provider_registry_returns_mock_bundle() -> None:
    settings = AppSettings(provider_backend="mock", vector_backend="memory")

    bundle = build_default_registry().create_bundle("mock", settings)

    trace = bundle.trace()
    assert trace["ocr"] == "mock-ocr"
    assert trace["reranker"] == "mock-reranker"


def test_provider_registry_returns_lmstudio_bundle() -> None:
    settings = AppSettings(
        provider_backend="lmstudio",
        vector_backend="memory",
        ocr_backend="mock",
        lmstudio_chat_model="local-chat",
        lmstudio_vision_model="local-vision",
        lmstudio_embedding_model="local-embedding",
    )

    bundle = build_default_registry().create_bundle("lmstudio", settings)

    trace = bundle.trace()
    assert trace["ocr"] == "mock-ocr"
    assert trace["embedding"] == "lmstudio-embedding"
    assert trace["query_analyzer"] == "lmstudio-query-analyzer"


def test_provider_registry_unknown_backend_raises() -> None:
    with pytest.raises(KeyError):
        build_default_registry().create_bundle("missing", AppSettings())


def test_local_bundle_is_explicitly_unsupported() -> None:
    settings = AppSettings(provider_backend="local", vector_backend="memory")
    bundle = build_default_registry().create_bundle("local", settings)

    with pytest.raises(UnsupportedLocalCapabilityError):
        bundle.query_analyzer.analyze_query("今天又被主管念")
