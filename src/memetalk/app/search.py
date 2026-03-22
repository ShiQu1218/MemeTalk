from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from memetalk.core.models import (
    MemeAsset,
    MemeMetadata,
    OCRStatus,
    QueryAnalysis,
    RerankCandidate,
    SearchMatch,
    SearchMode,
    SearchResponse,
    SearchResult,
    SearchResultDebug,
    SearchTrace,
    SearchScoringProfile,
    compose_embedding_text,
)
from memetalk.core.providers import ProviderBundle
from memetalk.core.retrieval import (
    build_index_version,
    build_reply_query_text,
    build_semantic_query_text,
    default_retrieval_weights,
    default_search_scoring_profile,
    lexical_overlap_score,
    split_terms,
    template_hint_score,
)
from memetalk.storage.sqlite_store import SQLiteMemeRepository
from memetalk.storage.vector_store import VectorStore

FALLBACK_REASON = "deterministic 排序 fallback：暫時無法完成 rerank，保留綜合檢索與規則評分最高的候選結果。"
RERANK_POOL_SIZE = 16
QUERY_ANALYSIS_CACHE_SIZE = 128
QUERY_EMBEDDING_CACHE_SIZE = 256
RERANK_RESULT_CACHE_SIZE = 128
QUERY_IMAGE_CACHE_SIZE = 64
DETERMINISTIC_ONLY_REASON = "deterministic-only 排序：使用已調整的規則權重直接排序，未啟用 rerank。"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _MergedCandidateState:
    candidate_sources: set[str] = field(default_factory=set)
    feature_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _ResolvedSearchInput:
    effective_query: str
    query_analysis: QueryAnalysis


class SearchService:
    def __init__(
        self,
        repository: SQLiteMemeRepository,
        vector_store: VectorStore,
        providers: ProviderBundle,
        api_base_url: str,
        scoring_profile: SearchScoringProfile | None = None,
        rerank_pool_size: int = RERANK_POOL_SIZE,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.providers = providers
        self.api_base_url = api_base_url.rstrip("/")
        self.scoring_profile = scoring_profile or default_search_scoring_profile()
        self.rerank_pool_size = rerank_pool_size
        self._query_analysis_cache: OrderedDict[str, QueryAnalysis] = OrderedDict()
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_image_cache: OrderedDict[str, MemeMetadata] = OrderedDict()
        self._rerank_cache: OrderedDict[str, list] = OrderedDict()
        self._provider_cache_identity = self._build_provider_cache_identity()

    def search(
        self,
        query: str | None,
        top_n: int,
        candidate_k: int,
        mode: SearchMode = SearchMode.REPLY,
        preferred_tone: str | None = None,
        rerank_enabled: bool = True,
        query_image_path: Path | None = None,
    ) -> SearchResponse:
        self._ensure_cache_identity()
        resolved_query = self._resolve_search_input(query, mode, preferred_tone, query_image_path)
        query_analysis = resolved_query.query_analysis
        route_scores, search_trace = self._retrieve_candidates(query_analysis, top_n, candidate_k, mode)
        asset_map = self.repository.get_assets_by_ids(list(route_scores.keys()))
        candidates = self._build_candidates(asset_map, route_scores, query_analysis, mode)
        candidates.sort(key=lambda item: item.deterministic_score, reverse=True)
        results, rerank_strategy = self._rerank_or_fallback(
            resolved_query.effective_query,
            query_analysis,
            candidates,
            top_n,
            mode=mode,
            rerank_enabled=rerank_enabled,
        )
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

    def _resolve_search_input(
        self,
        query: str | None,
        mode: SearchMode,
        preferred_tone: str | None,
        query_image_path: Path | None,
    ) -> _ResolvedSearchInput:
        text_query = (query or "").strip() or None
        if query_image_path is None:
            if text_query is None:
                raise ValueError("Search requires either text query or query image.")
            return _ResolvedSearchInput(
                effective_query=text_query,
                query_analysis=self._get_query_analysis(text_query, mode, preferred_tone),
            )

        image_metadata = self._get_query_image_metadata(query_image_path)
        if text_query is None:
            analysis = self._build_image_query_analysis(image_metadata, mode, preferred_tone, query_image_path.name)
            effective_query = self._build_image_query_label(image_metadata, query_image_path.name)
            return _ResolvedSearchInput(effective_query=effective_query, query_analysis=analysis)

        text_analysis = self._get_query_analysis(text_query, mode, preferred_tone)
        merged_analysis = self._merge_query_analysis_with_image(text_analysis, image_metadata)
        return _ResolvedSearchInput(effective_query=text_query, query_analysis=merged_analysis)

    def _get_query_image_metadata(self, image_path: Path) -> MemeMetadata:
        file_hash = self._sha256_file(image_path)
        cache_key = json.dumps(
            {
                "metadata_provider": self.providers.metadata_provider.name,
                "file_sha256": file_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = self._cache_get(self._query_image_cache, cache_key)
        if cached is not None:
            return cached
        metadata = self.providers.metadata_provider.analyze_image(image_path)
        self._cache_set(self._query_image_cache, cache_key, metadata, QUERY_IMAGE_CACHE_SIZE)
        return metadata

    def _build_image_query_analysis(
        self,
        image_metadata: MemeMetadata,
        mode: SearchMode,
        preferred_tone: str | None,
        original_query: str,
    ) -> QueryAnalysis:
        tone_values = self._merge_values([*image_metadata.style_tags[:1], *image_metadata.aesthetic_tags[:1]])
        return QueryAnalysis(
            original_query=original_query,
            situation=image_metadata.usage_scenario or image_metadata.scene_description or self._build_image_query_label(image_metadata, original_query),
            emotions=image_metadata.emotion_tags,
            tone=tone_values[0] if tone_values else "圖片相似查詢",
            reply_intent=image_metadata.intent_tags[0] if image_metadata.intent_tags else "回應",
            preferred_tone=preferred_tone,
            query_embedding_text=compose_embedding_text(image_metadata),
            query_terms=self._build_image_query_terms(image_metadata),
            template_hints=self._build_image_template_hints(image_metadata),
            retrieval_weights=default_retrieval_weights(mode),
        )

    def _merge_query_analysis_with_image(self, text_analysis: QueryAnalysis, image_metadata: MemeMetadata) -> QueryAnalysis:
        image_summary = compose_embedding_text(image_metadata)
        image_context = image_metadata.usage_scenario or image_metadata.scene_description
        merged_situation = "；".join(
            part
            for part in [
                text_analysis.situation,
                f"參考圖片：{image_context}" if image_context else "",
            ]
            if part and part.strip()
        )
        merged_tone_values = self._merge_values(
            [
                text_analysis.tone,
                *image_metadata.style_tags[:1],
                *image_metadata.aesthetic_tags[:1],
            ]
        )
        return QueryAnalysis(
            original_query=text_analysis.original_query,
            situation=merged_situation or text_analysis.situation,
            emotions=self._merge_values([*text_analysis.emotions, *image_metadata.emotion_tags]),
            tone="、".join(merged_tone_values) if merged_tone_values else text_analysis.tone,
            reply_intent=text_analysis.reply_intent or (image_metadata.intent_tags[0] if image_metadata.intent_tags else "回應"),
            preferred_tone=text_analysis.preferred_tone,
            query_embedding_text=f"{text_analysis.query_embedding_text}\n參考圖片資訊：\n{image_summary}",
            query_terms=self._merge_values([*text_analysis.query_terms, *self._build_image_query_terms(image_metadata)]),
            template_hints=self._merge_values([*text_analysis.template_hints, *self._build_image_template_hints(image_metadata)]),
            retrieval_weights=text_analysis.retrieval_weights,
        )

    def _build_image_template_hints(self, image_metadata: MemeMetadata) -> list[str]:
        return self._merge_values(
            [
                image_metadata.template_name or "",
                image_metadata.template_canonical_id,
                *image_metadata.template_aliases,
                image_metadata.template_family,
            ]
        )

    def _build_image_query_terms(self, image_metadata: MemeMetadata) -> list[str]:
        terms: list[str] = []
        for value in [
            image_metadata.template_name or "",
            image_metadata.template_canonical_id,
            *image_metadata.template_aliases,
            image_metadata.template_family,
            image_metadata.ocr_text,
            *image_metadata.ocr_lines,
            image_metadata.scene_description,
            image_metadata.meme_usage,
            image_metadata.visual_description,
            image_metadata.usage_scenario,
            *image_metadata.emotion_tags,
            *image_metadata.intent_tags,
            *image_metadata.style_tags,
            *image_metadata.aesthetic_tags,
        ]:
            if not value:
                continue
            stripped = value.strip()
            if not stripped:
                continue
            terms.append(stripped)
            terms.extend(split_terms(stripped))
        return self._merge_values(terms)[:24]

    def _build_image_query_label(self, image_metadata: MemeMetadata, fallback: str) -> str:
        for value in [image_metadata.ocr_text, image_metadata.template_name, image_metadata.scene_description, fallback]:
            if value and value.strip():
                return value.strip()
        return "圖片查詢"

    def _merge_values(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
            semantic_limit = max(top_n, limit // 2)
        else:
            reply_query_text = None
            semantic_limit = limit

        semantic_query_text = build_semantic_query_text(query_analysis)
        query_texts = {"semantic": semantic_query_text}
        if reply_query_text is not None:
            query_texts["reply_text"] = reply_query_text
        query_vectors = self._embed_query_texts(query_texts)
        degraded_routes: list[str] = []
        if reply_query_text is not None:
            reply_vector = query_vectors["reply_text"]
            reply_index_version = build_index_version(
                self.providers.embedding_provider.index_identity(),
                len(reply_vector),
                "reply_text",
            )
            index_versions["reply_text"] = reply_index_version
            try:
                reply_matches = self.vector_store.query(
                    reply_vector,
                    top_k=limit,
                    channel="reply_text",
                    index_version=reply_index_version,
                )
            except Exception:
                logger.warning("reply_text vector route failed, degrading gracefully", exc_info=True)
                reply_matches = []
                degraded_routes.append("reply_text")
            self._merge_matches(merged, reply_matches, "reply_text", "reply_vector")
            routes_used.append("reply_text")
            candidate_counts["reply_text"] = len(reply_matches)
        semantic_vector = query_vectors["semantic"]
        semantic_index_version = build_index_version(
            self.providers.embedding_provider.index_identity(),
            len(semantic_vector),
            "semantic",
        )
        index_versions["semantic"] = semantic_index_version
        try:
            semantic_matches = self.vector_store.query(
                semantic_vector,
                top_k=semantic_limit,
                channel="semantic",
                index_version=semantic_index_version,
            )
        except Exception:
            logger.warning("semantic vector route failed, degrading gracefully", exc_info=True)
            semantic_matches = []
            degraded_routes.append("semantic")
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
                degraded_routes=degraded_routes,
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
            feature_scores["preferred_tone_match"] = lexical_overlap_score(
                [query_analysis.preferred_tone] if query_analysis.preferred_tone else [],
                "\n".join(
                    [
                        asset.metadata.meme_usage,
                        " ".join(asset.metadata.style_tags),
                        " ".join(asset.metadata.emotion_tags),
                        " ".join(asset.metadata.intent_tags),
                        asset.metadata.ocr_text,
                    ]
                ),
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
            if (
                mode == SearchMode.REPLY
                and asset.metadata.ocr_status == OCRStatus.SUCCESS
                and feature_scores.get("ocr_overlap", 0.0) < self.scoring_profile.reply.ocr_mismatch_threshold
            ):
                degradation_flags.append("reply_ocr_mismatch")
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
        preferred_tone_match = feature_scores.get("preferred_tone_match", 0.0)
        semantic_text_overlap = feature_scores.get("semantic_text_overlap", 0.0)
        profile = self.scoring_profile.reply if mode == SearchMode.REPLY else self.scoring_profile.semantic

        if mode == SearchMode.REPLY:
            score = (
                (semantic_vector * weights.semantic * profile.semantic_vector)
                + (reply_vector * weights.reply_text * profile.reply_vector)
                + (keyword_route * weights.keyword * profile.keyword_route)
                + (template_route * weights.template * profile.template_route)
                + (ocr_overlap * profile.ocr_overlap)
                + (intent_match * profile.intent_match)
                + (emotion_overlap * profile.emotion_overlap)
                + (preferred_tone_match * profile.preferred_tone_match)
            )
            if (
                asset.metadata.ocr_status == OCRStatus.SUCCESS
                and profile.ocr_mismatch_score_cap is not None
                and ocr_overlap < profile.ocr_mismatch_threshold
            ):
                return min(score, profile.ocr_mismatch_score_cap)
            return score
        return (
            (semantic_vector * weights.semantic * profile.semantic_vector)
            + (reply_vector * weights.reply_text * profile.reply_vector)
            + (keyword_route * weights.keyword * profile.keyword_route)
            + (template_route * weights.template * profile.template_route)
            + (semantic_text_overlap * profile.semantic_text_overlap)
            + (emotion_overlap * profile.emotion_overlap)
            + (intent_match * profile.intent_match)
            + (preferred_tone_match * profile.preferred_tone_match)
        )

    def _rerank_or_fallback(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
        rerank_enabled: bool = True,
    ) -> tuple[list[SearchResult], str]:
        if not candidates:
            return [], "no_candidates"
        if not rerank_enabled:
            deterministic_candidates = sorted(candidates, key=lambda item: item.deterministic_score, reverse=True)[:top_n]
            return (
                [
                    SearchResult(
                        image_id=candidate.image_id,
                        image_url=f"{self.api_base_url}/api/v1/assets/{candidate.image_id}",
                        reason=DETERMINISTIC_ONLY_REASON,
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
                    for candidate in deterministic_candidates
                ],
                "deterministic_only",
            )
        rerank_pool_size = max(top_n, min(len(candidates), self.rerank_pool_size))
        rerank_pool = self._build_rerank_pool(candidates, rerank_pool_size, mode)
        try:
            reranked = self._rerank_candidates(query, query_analysis, rerank_pool, top_n, mode)
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
            logger.warning("Rerank failed, falling back to deterministic sort", exc_info=True)
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

    def _get_query_analysis(
        self,
        query: str,
        mode: SearchMode,
        preferred_tone: str | None,
    ) -> QueryAnalysis:
        cache_key = json.dumps(
            {"query": query, "mode": mode.value, "preferred_tone": preferred_tone},
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = self._cache_get(self._query_analysis_cache, cache_key)
        if cached is not None:
            return cached
        analysis = self.providers.query_analyzer.analyze_query(
            query,
            mode=mode,
            preferred_tone=preferred_tone,
        )
        self._cache_set(self._query_analysis_cache, cache_key, analysis, QUERY_ANALYSIS_CACHE_SIZE)
        return analysis

    def _embed_query_texts(self, query_texts: dict[str, str]) -> dict[str, list[float]]:
        embedding_identity = self.providers.embedding_provider.index_identity()
        vectors: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_routes: list[str] = []
        route_cache_keys: dict[str, str] = {}
        for route_name, text in query_texts.items():
            cache_key = json.dumps(
                {"embedding_identity": embedding_identity, "text": text},
                ensure_ascii=False,
                sort_keys=True,
            )
            route_cache_keys[route_name] = cache_key
            cached = self._cache_get(self._query_embedding_cache, cache_key)
            if cached is not None:
                vectors[route_name] = cached
                continue
            missing_routes.append(route_name)
            missing_texts.append(text)
        if missing_texts:
            embedded = self.providers.embedding_provider.embed_texts(missing_texts)
            for route_name, vector in zip(missing_routes, embedded, strict=False):
                cache_key = route_cache_keys[route_name]
                self._cache_set(self._query_embedding_cache, cache_key, vector, QUERY_EMBEDDING_CACHE_SIZE)
                vectors[route_name] = vector
        return vectors

    def _rerank_candidates(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode,
    ) -> list:
        cache_key = json.dumps(
            {
                "query": query,
                "mode": mode.value,
                "top_n": top_n,
                "query_analysis": query_analysis.model_dump(mode="json"),
                "candidates": [
                    {
                        "image_id": candidate.image_id,
                        "vector_score": round(candidate.vector_score, 6),
                        "candidate_sources": candidate.candidate_sources,
                        "feature_scores": {
                            key: round(value, 6) for key, value in sorted(candidate.feature_scores.items())
                        },
                        "degradation_flags": candidate.degradation_flags,
                        "ocr_status": candidate.metadata.ocr_status.value,
                    }
                    for candidate in candidates
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = self._cache_get(self._rerank_cache, cache_key)
        if cached is not None:
            return cached
        reranked = self.providers.reranker.rerank(query, query_analysis, candidates, top_n, mode=mode)
        self._cache_set(self._rerank_cache, cache_key, reranked, RERANK_RESULT_CACHE_SIZE)
        return reranked

    def _build_provider_cache_identity(self) -> str:
        return json.dumps(
            {
                "trace": self.providers.trace(),
                "embedding_identity": self.providers.embedding_provider.index_identity(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _ensure_cache_identity(self) -> None:
        current_identity = self._build_provider_cache_identity()
        if current_identity == self._provider_cache_identity:
            return
        self._query_analysis_cache.clear()
        self._query_embedding_cache.clear()
        self._query_image_cache.clear()
        self._rerank_cache.clear()
        self._provider_cache_identity = current_identity

    def _cache_get(self, cache: OrderedDict[str, object], key: str) -> object | None:
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _cache_set(self, cache: OrderedDict[str, object], key: str, value: object, max_size: int) -> None:
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)

    def set_scoring_profile(self, profile: SearchScoringProfile) -> None:
        self.scoring_profile = profile
        self._rerank_cache.clear()
