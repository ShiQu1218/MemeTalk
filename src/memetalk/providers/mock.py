from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from memetalk.core.models import (
    MemeMetadata,
    OCRExtraction,
    OCRStatus,
    QueryAnalysis,
    RerankCandidate,
    RerankResult,
    SearchMode,
)
from memetalk.core.providers import EmbeddingProvider, MetadataProvider, OCRProvider, QueryAnalyzer, Reranker
from memetalk.core.retrieval import default_retrieval_weights

TOKEN_RULES = {
    "無奈": ("無奈", "自嘲", "冷幽默"),
    "崩潰": ("崩潰", "抱怨", "誇張"),
    "傻眼": ("傻眼", "吐槽", "冷幽默"),
    "生氣": ("不爽", "打臉", "直接"),
    "厭世": ("厭世", "裝沒事", "黑色幽默"),
    "得意": ("得意", "炫耀", "誇張"),
    "委屈": ("委屈", "裝可憐", "戲劇化"),
    "朋友": ("社交", "吐槽", "日常"),
    "主管": ("壓力", "抱怨", "職場"),
    "上班": ("厭世", "抱怨", "職場"),
    "遲到": ("慌張", "裝可憐", "日常"),
    "text": ("有字", "吐槽", "截圖感"),
}


def _tokenize(value: str) -> list[str]:
    return [token for token in re.split(r"[\W_]+", value.lower()) if token]


def _derive_tags(text: str) -> tuple[list[str], list[str], list[str]]:
    emotions: list[str] = []
    intents: list[str] = []
    styles: list[str] = []
    for keyword, (emotion, intent, style) in TOKEN_RULES.items():
        if keyword.lower() in text.lower():
            emotions.append(emotion)
            intents.append(intent)
            styles.append(style)
    if not emotions:
        emotions.append("微妙")
    if not intents:
        intents.append("回應")
    if not styles:
        styles.append("網路梗")
    return emotions, intents, styles


class MockOCRProvider(OCRProvider):
    name = "mock-ocr"

    def extract_text(self, image_path: Path) -> OCRExtraction:
        stem = image_path.stem.lower()
        if "ocrfail" in stem or "ocr_fail" in stem:
            raise RuntimeError("Mock OCR provider forced failure.")
        if "text" in stem or "caption" in stem:
            raw_text = stem.replace("_", " ")
            return OCRExtraction(
                text=raw_text,
                has_text=True,
                raw_lines=[raw_text],
                status=OCRStatus.SUCCESS,
                confidence=0.98,
            )
        return OCRExtraction(text="", has_text=False, raw_lines=[], status=OCRStatus.EMPTY, confidence=0.0)


class MockMetadataProvider(MetadataProvider):
    name = "mock-metadata"

    def analyze_image(self, image_path: Path, ocr_result: OCRExtraction) -> MemeMetadata:
        stem = image_path.stem.replace("_", " ")
        lowered = stem.lower()
        if "broken fail" in lowered or "metadata fail" in lowered or "metadatafail" in lowered:
            raise RuntimeError("Mock metadata provider forced failure.")
        emotions, intents, styles = _derive_tags(f"{stem} {ocr_result.text}")
        return MemeMetadata(
            has_text=ocr_result.has_text,
            ocr_text=ocr_result.text,
            ocr_status=ocr_result.status,
            ocr_confidence=ocr_result.confidence,
            ocr_lines=ocr_result.raw_lines,
            template_name=stem,
            scene_description=f"模擬分析：{stem} 的反應梗圖場景。",
            meme_usage=f"適合用來回應與「{stem}」相近的吐槽或情緒情境。",
            emotion_tags=emotions,
            intent_tags=intents,
            style_tags=styles,
        )


class MockEmbeddingProvider(EmbeddingProvider):
    name = "mock-embedding"

    def __init__(self, dimensions: int = 24) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def index_identity(self) -> str:
        return f"{self.name}:dim-{self.dimensions}"

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokenize(text)
        if not tokens:
            tokens = ["empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimensions):
                vector[index] += digest[index] / 255.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class MockQueryAnalyzer(QueryAnalyzer):
    name = "mock-query-analyzer"

    def analyze_query(self, query: str, mode: SearchMode = SearchMode.REPLY) -> QueryAnalysis:
        emotions, intents, _styles = _derive_tags(query)
        tone = "吐槽型回覆" if "?" not in query else "帶疑問的回覆"
        reply_intent = intents[0]
        terms = [token for token in re.split(r"[，。！？、\s]+", query) if token]
        template_hints = [term for term in terms if "meme" in term.lower() or "template" in term.lower()]
        return QueryAnalysis(
            original_query=query,
            situation=f"使用者描述：{query}",
            emotions=emotions,
            tone=tone,
            reply_intent=reply_intent,
            query_embedding_text=(
                f"情境：{query}\n"
                f"情緒：{'、'.join(emotions)}\n"
                f"語氣：{tone}\n"
                f"回覆意圖：{reply_intent}"
            ),
            query_terms=terms,
            template_hints=template_hints,
            retrieval_weights=default_retrieval_weights(mode),
        )


class MockReranker(Reranker):
    name = "mock-reranker"

    def rerank(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> list[RerankResult]:
        ranked: list[tuple[float, RerankResult]] = []
        for candidate in candidates:
            score = candidate.deterministic_score
            if mode == SearchMode.REPLY and candidate.metadata.ocr_status == OCRStatus.SUCCESS:
                score += 0.15
            reason = (
                f"候選來源：{'/'.join(candidate.candidate_sources) or 'deterministic'}；"
                f"主要依據為台詞/意圖匹配與向量相似度，適合回應「{query}」。"
            )
            ranked.append((score, RerankResult(image_id=candidate.image_id, score=score, reason=reason)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in ranked[:top_n]]


class UnsupportedLocalCapabilityError(RuntimeError):
    pass


class UnsupportedLocalOCRProvider(OCRProvider):
    name = "local-unsupported-ocr"

    def extract_text(self, image_path: Path) -> OCRExtraction:
        raise UnsupportedLocalCapabilityError("Local OCR provider is not implemented in the MVP.")


class UnsupportedLocalMetadataProvider(MetadataProvider):
    name = "local-unsupported-metadata"

    def analyze_image(self, image_path: Path, ocr_result: OCRExtraction) -> MemeMetadata:
        raise UnsupportedLocalCapabilityError("Local metadata provider is not implemented in the MVP.")


class UnsupportedLocalEmbeddingProvider(EmbeddingProvider):
    name = "local-unsupported-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise UnsupportedLocalCapabilityError("Local embedding provider is not implemented in the MVP.")

    def index_identity(self) -> str:
        return f"{self.name}:unsupported"


class UnsupportedLocalQueryAnalyzer(QueryAnalyzer):
    name = "local-unsupported-query"

    def analyze_query(self, query: str, mode: SearchMode = SearchMode.REPLY) -> QueryAnalysis:
        raise UnsupportedLocalCapabilityError("Local query analyzer is not implemented in the MVP.")


class UnsupportedLocalReranker(Reranker):
    name = "local-unsupported-reranker"

    def rerank(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> list[RerankResult]:
        raise UnsupportedLocalCapabilityError("Local reranker is not implemented in the MVP.")
