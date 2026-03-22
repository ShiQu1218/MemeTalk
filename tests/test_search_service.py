from pathlib import Path

from PIL import Image

from memetalk.app.container import build_container
from memetalk.config import AppSettings
from memetalk.core.models import QueryAnalysis, RerankCandidate, RerankResult, SearchMode
from memetalk.core.providers import EmbeddingProvider, ProviderBundle, QueryAnalyzer, Reranker
from memetalk.core.retrieval import default_retrieval_weights


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


class FixedQueryAnalyzer(QueryAnalyzer):
    name = "fixed-query-analyzer"

    def __init__(
        self,
        query_terms: list[str],
        template_hints: list[str],
        mode: SearchMode,
        preferred_tone: str | None = None,
        emotions: list[str] | None = None,
        reply_intent: str = "吐槽",
    ) -> None:
        self.query_terms = query_terms
        self.template_hints = template_hints
        self.mode = mode
        self.preferred_tone = preferred_tone
        self.emotions = emotions or ["傻眼"]
        self.reply_intent = reply_intent

    def analyze_query(
        self,
        query: str,
        mode: SearchMode = SearchMode.REPLY,
        preferred_tone: str | None = None,
    ) -> QueryAnalysis:
        active_mode = mode if mode is not None else self.mode
        active_preferred_tone = preferred_tone or self.preferred_tone
        return QueryAnalysis(
            original_query=query,
            situation=query,
            emotions=self.emotions,
            tone="吐槽",
            reply_intent=self.reply_intent,
            preferred_tone=active_preferred_tone,
            query_embedding_text=query,
            query_terms=self.query_terms + ([active_preferred_tone] if active_preferred_tone else []),
            template_hints=self.template_hints,
            retrieval_weights=default_retrieval_weights(active_mode),
        )


class CountingQueryAnalyzer(FixedQueryAnalyzer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def analyze_query(
        self,
        query: str,
        mode: SearchMode = SearchMode.REPLY,
        preferred_tone: str | None = None,
    ) -> QueryAnalysis:
        self.calls += 1
        return super().analyze_query(query, mode=mode, preferred_tone=preferred_tone)


class CountingEmbeddingProvider(EmbeddingProvider):
    name = "counting-embedding"

    def __init__(self, delegate: EmbeddingProvider) -> None:
        self.delegate = delegate
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.delegate.embed_texts(texts)

    def index_identity(self) -> str:
        return self.delegate.index_identity()

    def embedding_dimensions(self) -> int | None:
        return self.delegate.embedding_dimensions()


class CountingReranker(Reranker):
    name = "counting-reranker"

    def __init__(self, delegate: Reranker) -> None:
        self.delegate = delegate
        self.calls = 0

    def rerank(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> list[RerankResult]:
        self.calls += 1
        return self.delegate.rerank(query, query_analysis, candidates, top_n, mode=mode)


def _build_container(tmp_path: Path):
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
    _create_image(source_dir / "AnimeReaction_text.png", (255, 0, 0))
    _create_image(source_dir / "anime_reaction.png", (0, 255, 0))
    _create_image(source_dir / "office_burnout.png", (0, 0, 255))
    container.indexing_service.build_index(source_dir)
    return container


def test_search_uses_keyword_and_vector_routes_for_template_like_queries(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    providers = container.providers
    container.search_service.providers = ProviderBundle(
        ocr_provider=providers.ocr_provider,
        metadata_provider=providers.metadata_provider,
        embedding_provider=providers.embedding_provider,
        query_analyzer=FixedQueryAnalyzer(["anime reaction", "text"], ["anime reaction"], SearchMode.REPLY),
        reranker=providers.reranker,
    )

    response = container.search_service.search("找 anime reaction 那張有字的", top_n=2, candidate_k=4, mode=SearchMode.REPLY)

    assert response.results
    top_result = response.results[0]
    assert "keyword" in top_result.debug.candidate_sources
    assert "semantic" in top_result.debug.candidate_sources


def test_reply_mode_downgrades_candidates_without_ocr_text(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    providers = container.providers
    container.search_service.providers = ProviderBundle(
        ocr_provider=providers.ocr_provider,
        metadata_provider=providers.metadata_provider,
        embedding_provider=providers.embedding_provider,
        query_analyzer=FixedQueryAnalyzer(["anime reaction", "text"], ["anime reaction"], SearchMode.REPLY),
        reranker=providers.reranker,
    )

    response = container.search_service.search("要拿來回覆的 anime reaction", top_n=2, candidate_k=4, mode=SearchMode.REPLY)

    assert len(response.results) == 2
    assert response.results[0].debug.degradation_flags == []
    assert response.results[1].debug.degradation_flags == ["reply_no_ocr:empty"]


def test_reply_mode_uses_ocr_backed_candidates_first_in_rerank_pool(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    providers = container.providers
    container.search_service.providers = ProviderBundle(
        ocr_provider=providers.ocr_provider,
        metadata_provider=providers.metadata_provider,
        embedding_provider=providers.embedding_provider,
        query_analyzer=FixedQueryAnalyzer(["anime reaction", "text"], ["anime reaction"], SearchMode.REPLY),
        reranker=providers.reranker,
    )

    response = container.search_service.search("找一張有明確台詞可以回嘴的圖", top_n=3, candidate_k=6, mode=SearchMode.REPLY)

    assert response.results[0].debug.degradation_flags == []
    assert "reply_text" in response.search_trace.routes_used
    assert response.search_trace.routes_used[:2] == ["keyword", "reply_text"]


def test_preferred_tone_biases_reply_ranking(tmp_path: Path) -> None:
    settings = AppSettings(
        sqlite_path=tmp_path / "memetalk.sqlite3",
        vector_backend="memory",
        provider_backend="mock",
        ocr_backend="mock",
        api_base_url="http://testserver",
    )
    container = build_container(settings)
    source_dir = tmp_path / "tone_memes"
    source_dir.mkdir()
    _create_image(source_dir / "委屈_text.png", (255, 0, 0))
    _create_image(source_dir / "無奈_text.png", (0, 255, 0))
    container.indexing_service.build_index(source_dir)
    providers = container.providers
    container.search_service.providers = ProviderBundle(
        ocr_provider=providers.ocr_provider,
        metadata_provider=providers.metadata_provider,
        embedding_provider=providers.embedding_provider,
        query_analyzer=FixedQueryAnalyzer(
            ["text"],
            [],
            SearchMode.REPLY,
            preferred_tone="戲劇化",
            emotions=[],
            reply_intent="回應",
        ),
        reranker=providers.reranker,
    )

    response = container.search_service.search(
        "給我一張拿來裝可憐的回覆圖",
        top_n=2,
        candidate_k=4,
        mode=SearchMode.REPLY,
        preferred_tone="戲劇化",
    )

    assert response.query_analysis.preferred_tone == "戲劇化"
    assert response.results[0].template_name == "委屈 text"
    assert response.results[0].debug.feature_scores["preferred_tone_match"] > 0


def test_image_only_search_uses_uploaded_image_metadata(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    query_image = tmp_path / "AnimeReaction_text_reference.png"
    _create_image(query_image, (128, 128, 128))

    response = container.search_service.search(
        query=None,
        top_n=2,
        candidate_k=4,
        mode=SearchMode.REPLY,
        query_image_path=query_image,
    )

    assert response.results
    assert response.results[0].template_name == "AnimeReaction text"
    assert "anime" in response.query_analysis.query_terms
    assert "keyword" in response.search_trace.routes_used


def test_mixed_text_and_image_query_merges_image_hints_into_analysis(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    query_image = tmp_path / "AnimeReaction_text_reference.png"
    _create_image(query_image, (64, 64, 64))

    response = container.search_service.search(
        query="給我更嘴一點的回覆圖",
        top_n=2,
        candidate_k=4,
        mode=SearchMode.REPLY,
        query_image_path=query_image,
    )

    assert "參考圖片資訊" in response.query_analysis.query_embedding_text
    assert any("AnimeReaction" in hint for hint in response.query_analysis.template_hints)


def test_repeated_identical_searches_reuse_cached_query_artifacts(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    providers = container.providers
    counting_analyzer = CountingQueryAnalyzer(["anime reaction", "text"], ["anime reaction"], SearchMode.REPLY)
    counting_embedding = CountingEmbeddingProvider(providers.embedding_provider)
    counting_reranker = CountingReranker(providers.reranker)
    container.search_service.providers = ProviderBundle(
        ocr_provider=providers.ocr_provider,
        metadata_provider=providers.metadata_provider,
        embedding_provider=counting_embedding,
        query_analyzer=counting_analyzer,
        reranker=counting_reranker,
    )

    first = container.search_service.search("找 anime reaction 那張有字的", top_n=2, candidate_k=4, mode=SearchMode.REPLY)
    second = container.search_service.search("找 anime reaction 那張有字的", top_n=2, candidate_k=4, mode=SearchMode.REPLY)

    assert first.results
    assert second.results
    assert counting_analyzer.calls == 1
    assert counting_reranker.calls == 1
    assert len(counting_embedding.calls) == 1
    assert len(counting_embedding.calls[0]) == 2
