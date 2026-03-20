from pathlib import Path

from PIL import Image

from memetalk.app.container import build_container
from memetalk.config import AppSettings
from memetalk.core.models import QueryAnalysis, SearchMode
from memetalk.core.providers import ProviderBundle, QueryAnalyzer
from memetalk.core.retrieval import default_retrieval_weights


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


class FixedQueryAnalyzer(QueryAnalyzer):
    name = "fixed-query-analyzer"

    def __init__(self, query_terms: list[str], template_hints: list[str], mode: SearchMode) -> None:
        self.query_terms = query_terms
        self.template_hints = template_hints
        self.mode = mode

    def analyze_query(self, query: str, mode: SearchMode = SearchMode.REPLY) -> QueryAnalysis:
        active_mode = mode if mode is not None else self.mode
        return QueryAnalysis(
            original_query=query,
            situation=query,
            emotions=["傻眼"],
            tone="吐槽",
            reply_intent="吐槽",
            query_embedding_text=query,
            query_terms=self.query_terms,
            template_hints=self.template_hints,
            retrieval_weights=default_retrieval_weights(active_mode),
        )


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
    assert response.results[1].score <= 0.48


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
