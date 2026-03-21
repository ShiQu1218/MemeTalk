from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from memetalk.core.models import (
    DeterministicModeScoringProfile,
    MemeMetadata,
    QueryAnalysis,
    RetrievalWeights,
    SearchMode,
    SearchScoringProfile,
)

_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip()
    normalized = _CAMEL_CASE_BOUNDARY.sub(" ", normalized)
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = _NON_WORD.sub(" ", normalized)
    return " ".join(normalized.lower().split())


def split_terms(value: str) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    return normalized.split()


def normalize_template_fields(template_name: str | None) -> tuple[str, list[str], str]:
    raw_name = (template_name or "").strip()
    if not raw_name:
        return "", [], ""
    normalized = normalize_text(raw_name)
    tokens = normalized.split()
    canonical_id = "-".join(tokens) if tokens else normalized
    aliases = [raw_name]
    if tokens:
        aliases.extend([" ".join(tokens), "_".join(tokens), "-".join(tokens)])
    deduped = list(dict.fromkeys(alias for alias in aliases if alias))
    family = tokens[0] if tokens else raw_name
    return canonical_id, deduped, family


def build_keyword_text(metadata: MemeMetadata) -> str:
    parts = [
        metadata.template_name or "",
        metadata.template_canonical_id,
        " ".join(metadata.template_aliases),
        metadata.template_family,
        metadata.ocr_text,
        metadata.scene_description,
        metadata.meme_usage,
        metadata.visual_description,
        metadata.usage_scenario,
        " ".join(metadata.emotion_tags),
        " ".join(metadata.intent_tags),
        " ".join(metadata.style_tags),
        " ".join(metadata.aesthetic_tags),
        metadata.ocr_status.value,
    ]
    return "\n".join(part for part in parts if part).strip()


def build_semantic_query_text(query_analysis: QueryAnalysis) -> str:
    parts = [
        f"情境：{query_analysis.situation}",
        f"情緒：{'、'.join(query_analysis.emotions) or '無'}",
        f"語氣：{query_analysis.tone}",
        f"回覆意圖：{query_analysis.reply_intent}",
    ]
    if query_analysis.preferred_tone:
        parts.append(f"偏好梗圖語氣：{query_analysis.preferred_tone}")
    parts.append(f"語意查詢：{query_analysis.query_embedding_text}")
    return "\n".join(parts)


def build_reply_query_text(query_analysis: QueryAnalysis) -> str:
    query_terms = "、".join(query_analysis.query_terms) or query_analysis.original_query
    parts = [
        f"適合回覆的字句：{query_terms}",
        f"語氣：{query_analysis.tone}",
        f"回覆意圖：{query_analysis.reply_intent}",
    ]
    if query_analysis.preferred_tone:
        parts.append(f"偏好梗圖語氣：{query_analysis.preferred_tone}")
    parts.append(f"模板提示：{'、'.join(query_analysis.template_hints) or '無'}")
    return "\n".join(parts)


def default_retrieval_weights(mode: SearchMode) -> RetrievalWeights:
    if mode == SearchMode.REPLY:
        return RetrievalWeights(semantic=0.2, reply_text=1.15, keyword=1.3, template=1.0)
    return RetrievalWeights(semantic=1.0, reply_text=0.35, keyword=0.6, template=0.5)


def default_search_scoring_profile() -> SearchScoringProfile:
    return SearchScoringProfile(
        reply=DeterministicModeScoringProfile(
            semantic_vector=0.18,
            reply_vector=0.3,
            keyword_route=0.26,
            template_route=0.14,
            ocr_overlap=0.42,
            intent_match=0.13,
            emotion_overlap=0.05,
            preferred_tone_match=0.16,
            penalty_multiplier=1.0,
            ocr_mismatch_score_cap=0.52,
            ocr_mismatch_threshold=0.1,
        ),
        semantic=DeterministicModeScoringProfile(
            semantic_vector=0.45,
            reply_vector=0.04,
            keyword_route=0.12,
            template_route=0.08,
            semantic_text_overlap=0.22,
            emotion_overlap=0.08,
            intent_match=0.03,
            preferred_tone_match=0.05,
            penalty_multiplier=0.3,
        ),
    )


def lexical_overlap_score(terms: Iterable[str], haystack: str) -> float:
    cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
    if not cleaned_terms:
        return 0.0
    normalized_haystack = unicodedata.normalize("NFKC", haystack or "").lower()
    matches = sum(1 for term in cleaned_terms if term in normalized_haystack)
    return matches / len(cleaned_terms)


def template_hint_score(template_hints: Iterable[str], metadata: MemeMetadata) -> float:
    searchable = " ".join([metadata.template_name or "", metadata.template_canonical_id, *metadata.template_aliases]).lower()
    hints = [hint.strip().lower() for hint in template_hints if hint.strip()]
    if not hints:
        return 0.0
    matches = sum(1 for hint in hints if hint in searchable)
    return matches / len(hints)


def build_index_version(identity: str, vector_length: int, channel: str) -> str:
    return f"{identity}|dim={vector_length}|channel={channel}"
