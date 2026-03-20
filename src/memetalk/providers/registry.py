from __future__ import annotations

from collections.abc import Callable

from memetalk.config import AppSettings
from memetalk.core.providers import ProviderBundle
from memetalk.providers.mock import (
    MockEmbeddingProvider,
    MockMetadataProvider,
    MockOCRProvider,
    MockQueryAnalyzer,
    MockReranker,
    UnsupportedLocalEmbeddingProvider,
    UnsupportedLocalMetadataProvider,
    UnsupportedLocalOCRProvider,
    UnsupportedLocalQueryAnalyzer,
    UnsupportedLocalReranker,
)
from memetalk.providers.openai_provider import (
    CompatibleEmbeddingProvider,
    CompatibleMetadataProvider,
    CompatibleQueryAnalyzer,
    CompatibleReranker,
    build_lmstudio_profile,
    build_openai_profile,
)
from memetalk.providers.paddleocr_provider import PaddleOCRProvider

BundleBuilder = Callable[[AppSettings], ProviderBundle]


class ProviderRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, BundleBuilder] = {}

    def register(self, name: str, builder: BundleBuilder) -> None:
        self._builders[name] = builder

    def create_bundle(self, name: str, settings: AppSettings) -> ProviderBundle:
        try:
            builder = self._builders[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider backend: {name}") from exc
        return builder(settings)


def _build_mock_bundle(_settings: AppSettings) -> ProviderBundle:
    return ProviderBundle(
        ocr_provider=MockOCRProvider(),
        metadata_provider=MockMetadataProvider(),
        embedding_provider=MockEmbeddingProvider(),
        query_analyzer=MockQueryAnalyzer(),
        reranker=MockReranker(),
    )


def _build_local_bundle(_settings: AppSettings) -> ProviderBundle:
    return ProviderBundle(
        ocr_provider=UnsupportedLocalOCRProvider(),
        metadata_provider=UnsupportedLocalMetadataProvider(),
        embedding_provider=UnsupportedLocalEmbeddingProvider(),
        query_analyzer=UnsupportedLocalQueryAnalyzer(),
        reranker=UnsupportedLocalReranker(),
    )


def _build_openai_bundle(settings: AppSettings) -> ProviderBundle:
    ocr_provider = MockOCRProvider() if settings.ocr_backend == "mock" else PaddleOCRProvider()
    profile = build_openai_profile(settings)
    return ProviderBundle(
        ocr_provider=ocr_provider,
        metadata_provider=CompatibleMetadataProvider(profile),
        embedding_provider=CompatibleEmbeddingProvider(profile),
        query_analyzer=CompatibleQueryAnalyzer(profile),
        reranker=CompatibleReranker(profile),
    )


def _build_lmstudio_bundle(settings: AppSettings) -> ProviderBundle:
    ocr_provider = MockOCRProvider() if settings.ocr_backend == "mock" else PaddleOCRProvider()
    profile = build_lmstudio_profile(settings)
    return ProviderBundle(
        ocr_provider=ocr_provider,
        metadata_provider=CompatibleMetadataProvider(profile),
        embedding_provider=CompatibleEmbeddingProvider(profile),
        query_analyzer=CompatibleQueryAnalyzer(profile),
        reranker=CompatibleReranker(profile),
    )


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("mock", _build_mock_bundle)
    registry.register("local", _build_local_bundle)
    registry.register("openai", _build_openai_bundle)
    registry.register("lmstudio", _build_lmstudio_bundle)
    return registry


def build_provider_bundle(settings: AppSettings) -> ProviderBundle:
    return build_default_registry().create_bundle(settings.provider_backend, settings)
