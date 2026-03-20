from __future__ import annotations

from memetalk.core.models import QueryAnalysis, RerankCandidate, SearchMode, SearchResponse, SearchResult
from memetalk.core.providers import ProviderBundle
from memetalk.storage.sqlite_store import SQLiteMemeRepository
from memetalk.storage.vector_store import VectorStore

FALLBACK_REASON = "向量排序 fallback：暫時無法完成 rerank，保留語意最接近的候選結果。"


class SearchService:
    def __init__(
        self,
        repository: SQLiteMemeRepository,
        vector_store: VectorStore,
        providers: ProviderBundle,
        api_base_url: str,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.providers = providers
        self.api_base_url = api_base_url.rstrip("/")

    def search(self, query: str, top_n: int, candidate_k: int, mode: SearchMode = SearchMode.REPLY) -> SearchResponse:
        query_analysis = self.providers.query_analyzer.analyze_query(query, mode=mode)
        query_vector = self.providers.embedding_provider.embed_texts([query_analysis.query_embedding_text])[0]
        vector_matches = self.vector_store.query(query_vector, top_k=max(top_n, candidate_k), search_mode=mode.value)
        candidates: list[RerankCandidate] = []
        for match in vector_matches:
            asset = self.repository.get_asset_by_id(match.image_id)
            if not asset:
                continue
            candidates.append(
                RerankCandidate(
                    image_id=asset.image_id,
                    vector_score=match.score,
                    file_path=asset.file_path,
                    metadata=asset.metadata,
                )
            )
        results = self._rerank_or_fallback(query, query_analysis, candidates, top_n, mode=mode)
        return SearchResponse(query_analysis=query_analysis, results=results, provider_trace=self.providers.trace())

    def _rerank_or_fallback(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        try:
            reranked = self.providers.reranker.rerank(query, query_analysis, candidates, top_n, mode=mode)
            reranked_by_id = {item.image_id: item for item in reranked}
            ordered_candidates = [candidate for candidate in candidates if candidate.image_id in reranked_by_id]
            ordered_candidates.sort(key=lambda item: reranked_by_id[item.image_id].score, reverse=True)
            return [
                SearchResult(
                    image_id=candidate.image_id,
                    image_url=f"{self.api_base_url}/api/v1/assets/{candidate.image_id}",
                    reason=reranked_by_id[candidate.image_id].reason,
                    score=reranked_by_id[candidate.image_id].score,
                    template_name=candidate.metadata.template_name,
                    emotion_tags=candidate.metadata.emotion_tags,
                    intent_tags=candidate.metadata.intent_tags,
                )
                for candidate in ordered_candidates[:top_n]
            ]
        except Exception:
            fallback_candidates = sorted(candidates, key=lambda item: item.vector_score, reverse=True)[:top_n]
            return [
                SearchResult(
                    image_id=candidate.image_id,
                    image_url=f"{self.api_base_url}/api/v1/assets/{candidate.image_id}",
                    reason=FALLBACK_REASON,
                    score=candidate.vector_score,
                    template_name=candidate.metadata.template_name,
                    emotion_tags=candidate.metadata.emotion_tags,
                    intent_tags=candidate.metadata.intent_tags,
                )
                for candidate in fallback_candidates
            ]
