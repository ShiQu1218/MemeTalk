from __future__ import annotations

from dataclasses import dataclass, field

from memetalk.core.models import (
    MemeAsset,
    OCRStatus,
    QueryAnalysis,
    RerankCandidate,
    SearchMatch,
    SearchMode,
    SearchResponse,
    SearchResult,
    SearchResultDebug,
    SearchTrace,
)
from memetalk.core.providers import ProviderBundle
from memetalk.core.retrieval import (
    build_index_version,
    build_reply_query_text,
    build_semantic_query_text,
    lexical_overlap_score,
    ocr_penalty,
    template_hint_score,
)
from memetalk.storage.sqlite_store import SQLiteMemeRepository
from memetalk.storage.vector_store import VectorStore

FALLBACK_REASON = "deterministic 排序 fallback：暫時無法完成 rerank，保留綜合檢索與規則評分最高的候選結果。"
RERANK_POOL_MULTIPLIER = 3
REPLY_NON_OCR_SCORE_CAP = 0.48


@dataclass(slots=True)
class _MergedCandidateState:
    candidate_sources: set[str] = field(default_factory=set)
    feature_scores: dict[str, float] = field(default_factory=dict)


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
        route_scores, search_trace = self._retrieve_candidates(query_analysis, top_n, candidate_k, mode)
        asset_map = self.repository.get_assets_by_ids(list(route_scores.keys()))
        candidates = self._build_candidates(asset_map, route_scores, query_analysis, mode)
        candidates.sort(key=lambda item: item.deterministic_score, reverse=True)
        results, rerank_strategy = self._rerank_or_fallback(query, query_analysis, candidates, top_n, mode=mode)
        search_trace.rerank_strategy = rerank_strategy
        if mode == SearchMode.REPLY and ("reply_text" not in search_trace.candidate_counts or search_trace.candidate_counts.get("reply_text", 0) == 0):
            search_trace.degraded_routes.append("reply_text")
        if mode == SearchMode.REPLY and any(result.debug.degradation_flags for result in results):
            search_trace.degraded_routes = sorted(set(search_trace.degraded_routes + ["reply_text"]))
        return SearchResponse(
            query_analysis=query_analysis,
            results=results,
            provider_trace=self.providers.trace(),
            search_trace=search_trace,
        )

    def _retrieve_candidates(
        self,
        query_analysis: QueryAnalysis,
        top_n: int,
        candidate_k: int,
        mode: SearchMode,
    ) -> tuple[dict[str, _MergedCandidateState], SearchTrace]:
        limit = max(top_n, candidate_k)
        routes_used: list[str] = []
        candidate_counts: dict[str, int] = {}
        index_versions: dict[str, str] = {}
        merged: dict[str, _MergedCandidateState] = {}

        if mode == SearchMode.REPLY:
            keyword_matches = self.repository.search_keyword_candidates(
                query_terms=query_analysis.query_terms or [query_analysis.original_query],
                template_hints=query_analysis.template_hints,
                top_k=limit,
            )
            self._merge_matches(merged, keyword_matches, "keyword", "keyword_route")
            routes_used.append("keyword")
            candidate_counts["keyword"] = len(keyword_matches)

            reply_query_text = build_reply_query_text(query_analysis)
            reply_vector = self.providers.embedding_provider.embed_texts([reply_query_text])[0]
            reply_index_version = build_index_version(
                self.providers.embedding_provider.index_identity(),
                len(reply_vector),
                "reply_text",
            )
            index_versions["reply_text"] = reply_index_version
            reply_matches = self.vector_store.query(
                reply_vector,
                top_k=limit,
                channel="reply_text",
                index_version=reply_index_version,
            )
            self._merge_matches(merged, reply_matches, "reply_text", "reply_vector")
            routes_used.append("reply_text")
            candidate_counts["reply_text"] = len(reply_matches)
            semantic_limit = max(top_n, limit // 2)
        else:
            semantic_limit = limit

        semantic_query_text = build_semantic_query_text(query_analysis)
        semantic_vector = self.providers.embedding_provider.embed_texts([semantic_query_text])[0]
        semantic_index_version = build_index_version(
            self.providers.embedding_provider.index_identity(),
            len(semantic_vector),
            "semantic",
        )
        index_versions["semantic"] = semantic_index_version
        semantic_matches = self.vector_store.query(
            semantic_vector,
            top_k=semantic_limit,
            channel="semantic",
            index_version=semantic_index_version,
        )
        self._merge_matches(merged, semantic_matches, "semantic", "semantic_vector")
        routes_used.append("semantic")
        candidate_counts["semantic"] = len(semantic_matches)

        if mode != SearchMode.REPLY:
            keyword_matches = self.repository.search_keyword_candidates(
                query_terms=query_analysis.query_terms or [query_analysis.original_query],
                template_hints=query_analysis.template_hints,
                top_k=limit,
            )
            self._merge_matches(merged, keyword_matches, "keyword", "keyword_route")
            routes_used.append("keyword")
            candidate_counts["keyword"] = len(keyword_matches)

        return (
            merged,
            SearchTrace(
                routes_used=routes_used,
                candidate_counts=candidate_counts,
                index_versions=index_versions,
            ),
        )

    def _merge_matches(
        self,
        merged: dict[str, _MergedCandidateState],
        matches: list[SearchMatch],
        route_name: str,
        feature_name: str,
    ) -> None:
        for match in matches:
            state = merged.setdefault(match.image_id, _MergedCandidateState())
            state.candidate_sources.add(route_name)
            state.feature_scores[feature_name] = max(state.feature_scores.get(feature_name, 0.0), match.score)
            if route_name == "keyword":
                state.feature_scores["template_route_score"] = max(
                    state.feature_scores.get("template_route_score", 0.0),
                    float(match.metadata.get("template_route_score", 0.0)),
                )
                state.feature_scores["ocr_route_score"] = max(
                    state.feature_scores.get("ocr_route_score", 0.0),
                    float(match.metadata.get("ocr_route_score", 0.0)),
                )

    def _build_candidates(
        self,
        asset_map: dict[str, MemeAsset],
        route_scores: dict[str, _MergedCandidateState],
        query_analysis: QueryAnalysis,
        mode: SearchMode,
    ) -> list[RerankCandidate]:
        candidates: list[RerankCandidate] = []
        for image_id, state in route_scores.items():
            asset = asset_map.get(image_id)
            if asset is None:
                continue
            feature_scores = dict(state.feature_scores)
            feature_scores["ocr_overlap"] = lexical_overlap_score(
                query_analysis.query_terms or [query_analysis.original_query],
                asset.metadata.ocr_text,
            )
            feature_scores["template_match"] = template_hint_score(query_analysis.template_hints, asset.metadata)
            feature_scores["emotion_overlap"] = lexical_overlap_score(
                query_analysis.emotions,
                " ".join(asset.metadata.emotion_tags),
            )
            feature_scores["intent_match"] = lexical_overlap_score(
                [query_analysis.reply_intent],
                " ".join(asset.metadata.intent_tags),
            )
            feature_scores["semantic_text_overlap"] = lexical_overlap_score(
                query_analysis.query_terms or [query_analysis.original_query],
                f"{asset.metadata.scene_description}\n{asset.metadata.meme_usage}",
            )
            deterministic_score = self._deterministic_score(feature_scores, asset, query_analysis, mode)
            feature_scores["deterministic_score"] = deterministic_score
            degradation_flags: list[str] = []
            if mode == SearchMode.REPLY and asset.metadata.ocr_status != OCRStatus.SUCCESS:
                degradation_flags.append(f"reply_no_ocr:{asset.metadata.ocr_status.value}")
            candidates.append(
                RerankCandidate(
                    image_id=image_id,
                    vector_score=max(
                        feature_scores.get("semantic_vector", 0.0),
                        feature_scores.get("reply_vector", 0.0),
                    ),
                    file_path=asset.file_path,
                    metadata=asset.metadata,
                    candidate_sources=sorted(state.candidate_sources),
                    feature_scores=feature_scores,
                    degradation_flags=degradation_flags,
                    deterministic_score=deterministic_score,
                )
            )
        return candidates

    def _deterministic_score(
        self,
        feature_scores: dict[str, float],
        asset: MemeAsset,
        query_analysis: QueryAnalysis,
        mode: SearchMode,
    ) -> float:
        weights = query_analysis.retrieval_weights
        semantic_vector = feature_scores.get("semantic_vector", 0.0)
        reply_vector = feature_scores.get("reply_vector", 0.0)
        keyword_route = feature_scores.get("keyword_route", 0.0)
        template_route = max(feature_scores.get("template_route_score", 0.0), feature_scores.get("template_match", 0.0))
        ocr_overlap = max(feature_scores.get("ocr_route_score", 0.0), feature_scores.get("ocr_overlap", 0.0))
        emotion_overlap = feature_scores.get("emotion_overlap", 0.0)
        intent_match = feature_scores.get("intent_match", 0.0)
        semantic_text_overlap = feature_scores.get("semantic_text_overlap", 0.0)
        penalty = ocr_penalty(asset.metadata, mode)

        if mode == SearchMode.REPLY:
            score = (
                (semantic_vector * weights.semantic * 0.18)
                + (reply_vector * weights.reply_text * 0.3)
                + (keyword_route * weights.keyword * 0.26)
                + (template_route * weights.template * 0.14)
                + (ocr_overlap * 0.42)
                + (intent_match * 0.13)
                + (emotion_overlap * 0.05)
                - penalty
            )
            if asset.metadata.ocr_status != OCRStatus.SUCCESS:
                return min(score, REPLY_NON_OCR_SCORE_CAP)
            return score
        return (
            (semantic_vector * weights.semantic * 0.45)
            + (reply_vector * weights.reply_text * 0.04)
            + (keyword_route * weights.keyword * 0.12)
            + (template_route * weights.template * 0.08)
            + (semantic_text_overlap * 0.22)
            + (emotion_overlap * 0.08)
            + (intent_match * 0.03)
            - (penalty * 0.3)
        )

    def _rerank_or_fallback(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> tuple[list[SearchResult], str]:
        if not candidates:
            return [], "no_candidates"
        rerank_pool_size = max(top_n, min(len(candidates), top_n * RERANK_POOL_MULTIPLIER))
        rerank_pool = self._build_rerank_pool(candidates, rerank_pool_size, mode)
        try:
            reranked = self.providers.reranker.rerank(query, query_analysis, rerank_pool, top_n, mode=mode)
            reranked_by_id = {item.image_id: item for item in reranked}
            ordered_candidates = [candidate for candidate in rerank_pool if candidate.image_id in reranked_by_id]
            if len(ordered_candidates) < top_n:
                ordered_ids = {candidate.image_id for candidate in ordered_candidates}
                ordered_candidates.extend(candidate for candidate in candidates if candidate.image_id not in ordered_ids)
            ordered_candidates.sort(
                key=lambda item: reranked_by_id.get(item.image_id, None).score if item.image_id in reranked_by_id else item.deterministic_score,
                reverse=True,
            )
            return (
                [
                    SearchResult(
                        image_id=candidate.image_id,
                        image_url=f"{self.api_base_url}/api/v1/assets/{candidate.image_id}",
                        reason=reranked_by_id.get(candidate.image_id).reason if candidate.image_id in reranked_by_id else FALLBACK_REASON,
                        score=reranked_by_id.get(candidate.image_id).score if candidate.image_id in reranked_by_id else candidate.deterministic_score,
                        template_name=candidate.metadata.template_name,
                        emotion_tags=candidate.metadata.emotion_tags,
                        intent_tags=candidate.metadata.intent_tags,
                        debug=SearchResultDebug(
                            candidate_sources=candidate.candidate_sources,
                            degradation_flags=candidate.degradation_flags,
                            feature_scores=candidate.feature_scores,
                        ),
                    )
                    for candidate in ordered_candidates[:top_n]
                ],
                "llm_rerank",
            )
        except Exception:
            fallback_candidates = sorted(candidates, key=lambda item: item.deterministic_score, reverse=True)[:top_n]
            return (
                [
                    SearchResult(
                        image_id=candidate.image_id,
                        image_url=f"{self.api_base_url}/api/v1/assets/{candidate.image_id}",
                        reason=FALLBACK_REASON,
                        score=candidate.deterministic_score,
                        template_name=candidate.metadata.template_name,
                        emotion_tags=candidate.metadata.emotion_tags,
                        intent_tags=candidate.metadata.intent_tags,
                        debug=SearchResultDebug(
                            candidate_sources=candidate.candidate_sources,
                            degradation_flags=candidate.degradation_flags,
                            feature_scores=candidate.feature_scores,
                        ),
                    )
                    for candidate in fallback_candidates
                ],
                "deterministic_fallback",
            )

    def _build_rerank_pool(
        self,
        candidates: list[RerankCandidate],
        rerank_pool_size: int,
        mode: SearchMode,
    ) -> list[RerankCandidate]:
        if mode != SearchMode.REPLY:
            return candidates[:rerank_pool_size]
        ocr_backed = [candidate for candidate in candidates if candidate.metadata.ocr_status == OCRStatus.SUCCESS]
        degraded = [candidate for candidate in candidates if candidate.metadata.ocr_status != OCRStatus.SUCCESS]
        ordered = ocr_backed + degraded
        return ordered[:rerank_pool_size]
