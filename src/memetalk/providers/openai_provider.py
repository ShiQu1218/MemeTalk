from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from memetalk.config import AppSettings
from memetalk.core.models import MemeMetadata, OCRExtraction, QueryAnalysis, RerankCandidate, RerankResult, SearchMode
from memetalk.core.providers import EmbeddingProvider, MetadataProvider, QueryAnalyzer, Reranker
from memetalk.core.retrieval import default_retrieval_weights

JSON_COMPLETION_MAX_ATTEMPTS = 3


def _build_image_data_url(image_path: Path, provider_label: str) -> str:
    suffix = image_path.suffix.lower()
    if provider_label == "lmstudio" and suffix == ".webp":
        buffer = io.BytesIO()
        with Image.open(image_path) as image:
            image.save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{payload}"

    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    payload = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def _extract_json_object(payload: str) -> dict[str, Any]:
    text = payload.strip()
    if not text:
        return {}
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Provider response does not contain a JSON object.")
    return json.loads(text[start : end + 1])


@dataclass(frozen=True, slots=True)
class CompatibleProviderProfile:
    label: str
    base_url: str | None
    api_key: str | None
    chat_model: str
    vision_model: str
    embedding_model: str


def build_openai_profile(settings: AppSettings) -> CompatibleProviderProfile:
    return CompatibleProviderProfile(
        label="openai",
        base_url=(settings.openai_base_url or None),
        api_key=settings.openai_api_key,
        chat_model=settings.openai_chat_model,
        vision_model=settings.openai_vision_model,
        embedding_model=settings.openai_embedding_model,
    )


def build_lmstudio_profile(settings: AppSettings) -> CompatibleProviderProfile:
    return CompatibleProviderProfile(
        label="lmstudio",
        base_url=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key or "lm-studio",
        chat_model=settings.lmstudio_chat_model or settings.openai_chat_model,
        vision_model=settings.lmstudio_vision_model or settings.openai_vision_model,
        embedding_model=settings.lmstudio_embedding_model or settings.openai_embedding_model,
    )


class _OpenAICompatibleBase:
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        self.profile = profile

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai extra is not installed. Install with `pip install -e .[openai]`.") from exc

        client_kwargs: dict[str, str] = {"api_key": self._resolve_api_key()}
        if self.profile.base_url:
            client_kwargs["base_url"] = self.profile.base_url
        return OpenAI(**client_kwargs)

    def _resolve_api_key(self) -> str:
        if self.profile.api_key:
            return self.profile.api_key
        if self.profile.base_url:
            return "lm-studio"
        raise RuntimeError(
            "MEMETALK_OPENAI_API_KEY is required for the OpenAI provider unless a compatible base URL is configured."
        )

    def _translate_provider_error(self, exc: Exception, capability: str) -> Exception:
        message = str(exc)
        if self.profile.label != "lmstudio":
            return exc
        if "No models loaded" not in message:
            return exc
        if capability == "embedding":
            return RuntimeError(
                "LM Studio reports no embedding model is loaded. "
                "Load an embedding-capable model in LM Studio and set MEMETALK_LMSTUDIO_EMBEDDING_MODEL to that model id."
            )
        if capability == "vision":
            return RuntimeError(
                "LM Studio reports no vision/chat model is loaded. "
                "Load a vision-capable chat model in LM Studio and set MEMETALK_LMSTUDIO_VISION_MODEL to that model id."
            )
        return RuntimeError(
            "LM Studio reports no chat model is loaded. "
            "Load a chat-capable model in LM Studio and set MEMETALK_LMSTUDIO_CHAT_MODEL to that model id."
        )

    def _build_json_system_prompt(self, prompt: str, attempt: int) -> str:
        retry_note = ""
        if attempt:
            retry_note = (
                "上一次輸出的 JSON 格式錯誤。"
                "這次所有欄位與陣列元素之間都必須使用半形逗號，"
                "並確保輸出可以被 Python json.loads 直接解析。"
            )
        return (
            f"{prompt}"
            f"{retry_note}"
            "只回傳單一 JSON 物件。"
            "不要使用 Markdown、不要加前言、不要加結尾說明。"
        )

    def _json_completion(self, prompt: str, user_content: str | list[dict[str, Any]], model: str) -> dict[str, Any]:
        capability = "vision" if isinstance(user_content, list) else "chat"
        last_error: Exception | None = None
        for attempt in range(JSON_COMPLETION_MAX_ATTEMPTS):
            try:
                response = self._client().chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": self._build_json_system_prompt(prompt, attempt),
                        },
                        {"role": "user", "content": user_content},
                    ],
                )
            except Exception as exc:
                raise self._translate_provider_error(exc, capability) from exc
            payload = response.choices[0].message.content or ""
            try:
                return _extract_json_object(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Provider returned malformed JSON after {JSON_COMPLETION_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error


class CompatibleEmbeddingProvider(_OpenAICompatibleBase, EmbeddingProvider):
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        super().__init__(profile)
        self.name = f"{profile.label}-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client().embeddings.create(model=self.profile.embedding_model, input=texts)
        except Exception as exc:
            raise self._translate_provider_error(exc, "embedding") from exc
        return [item.embedding for item in response.data]

    def index_identity(self) -> str:
        return f"{self.name}:{self.profile.embedding_model}"


class CompatibleQueryAnalyzer(_OpenAICompatibleBase, QueryAnalyzer):
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        super().__init__(profile)
        self.name = f"{profile.label}-query-analyzer"

    def analyze_query(
        self,
        query: str,
        mode: SearchMode = SearchMode.REPLY,
        preferred_tone: str | None = None,
    ) -> QueryAnalysis:
        if mode == SearchMode.REPLY:
            prompt = (
                "你是繁體中文的梗圖查詢分析器，目前處於「回覆模式」。"
                "使用者輸入的是一句話，他想找到一張梗圖來當作回覆。"
                "請把使用者輸入分析成 JSON，欄位為 situation, emotions, tone, reply_intent, "
                "preferred_tone, query_embedding_text, query_terms, template_hints, retrieval_weights。"
                "query_embedding_text 必須描述「適合回覆這句話的梗圖會包含什麼文字或表達什麼意思」，"
                "而不是重述使用者的輸入。重點放在回應的語氣、態度和可能的文字內容。"
                "若使用者另有指定 preferred_tone，請把它當成希望梗圖呈現的額外語氣偏好，"
                "但不要覆蓋原本的情境與回覆意圖。"
                "query_terms 請列出適合做關鍵字檢索的短詞或短句。"
                "template_hints 只放使用者明確提到或暗示的模板線索。"
                "retrieval_weights 必須包含 semantic, reply_text, keyword, template 四個 0 到 1.5 的數字。"
            )
        else:
            prompt = (
                "你是繁體中文的梗圖查詢分析器。"
                "請把使用者輸入分析成 JSON，欄位為 situation, emotions, tone, reply_intent, "
                "preferred_tone, query_embedding_text, query_terms, template_hints, retrieval_weights。"
                "query_embedding_text 必須是適合向量搜尋的繁體中文敘述，必須保留查詢中的關鍵名詞和具體用語，不要過度抽象化。"
                "若使用者有提供 preferred_tone，請把它視為額外偏好，協助你判斷希望的梗圖風格。"
                "query_terms 請列出適合做關鍵字檢索的短詞或短句。"
                "template_hints 只放使用者明確提到或暗示的模板線索。"
                "retrieval_weights 必須包含 semantic, reply_text, keyword, template 四個 0 到 1.5 的數字。"
            )
        data = self._json_completion(
            prompt,
            json.dumps(
                {
                    "query": query,
                    "preferred_tone": preferred_tone,
                },
                ensure_ascii=False,
            ),
            self.profile.chat_model,
        )
        return QueryAnalysis(
            original_query=query,
            situation=data["situation"],
            emotions=data.get("emotions", []),
            tone=data["tone"],
            reply_intent=data["reply_intent"],
            preferred_tone=data.get("preferred_tone") or preferred_tone,
            query_embedding_text=data["query_embedding_text"],
            query_terms=data.get("query_terms", []),
            template_hints=data.get("template_hints", []),
            retrieval_weights=data.get("retrieval_weights", default_retrieval_weights(mode).model_dump()),
        )


class CompatibleMetadataProvider(_OpenAICompatibleBase, MetadataProvider):
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        super().__init__(profile)
        self.name = f"{profile.label}-metadata"

    def analyze_image(self, image_path: Path, ocr_result: OCRExtraction) -> MemeMetadata:
        image_data_url = _build_image_data_url(image_path, self.profile.label)
        prompt = (
            "你是繁體中文的梗圖 metadata 分析器。"
            "請回傳 JSON，欄位為 template_name, scene_description, meme_usage, emotion_tags, intent_tags, style_tags。"
            "回覆時要考慮圖片內容以及 OCR 文字。"
        )
        data = self._json_completion(
            prompt,
            [
                {"type": "text", "text": f"OCR 文字：{ocr_result.text or '無'}"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
            self.profile.vision_model,
        )
        return MemeMetadata(
            has_text=ocr_result.has_text,
            ocr_text=ocr_result.text,
            template_name=data.get("template_name"),
            scene_description=data["scene_description"],
            meme_usage=data["meme_usage"],
            emotion_tags=data.get("emotion_tags", []),
            intent_tags=data.get("intent_tags", []),
            style_tags=data.get("style_tags", []),
        )


class CompatibleReranker(_OpenAICompatibleBase, Reranker):
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        super().__init__(profile)
        self.name = f"{profile.label}-reranker"

    def rerank(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> list[RerankResult]:
        if mode == SearchMode.REPLY:
            serialized = [
                {
                    "image_id": candidate.image_id,
                    "vector_score": candidate.vector_score,
                    "template_name": candidate.metadata.template_name,
                    "meme_usage": candidate.metadata.meme_usage,
                    "ocr_text": candidate.metadata.ocr_text,
                    "ocr_status": candidate.metadata.ocr_status,
                    "intent_tags": candidate.metadata.intent_tags,
                    "candidate_sources": candidate.candidate_sources,
                    "feature_scores": candidate.feature_scores,
                    "degradation_flags": candidate.degradation_flags,
                    "deterministic_score": candidate.deterministic_score,
                }
                for candidate in candidates
            ]
        else:
            serialized = [
                {
                    "image_id": candidate.image_id,
                    "vector_score": candidate.vector_score,
                    "template_name": candidate.metadata.template_name,
                    "scene_description": candidate.metadata.scene_description,
                    "meme_usage": candidate.metadata.meme_usage,
                    "ocr_text": candidate.metadata.ocr_text,
                    "ocr_status": candidate.metadata.ocr_status,
                    "emotion_tags": candidate.metadata.emotion_tags,
                    "intent_tags": candidate.metadata.intent_tags,
                    "candidate_sources": candidate.candidate_sources,
                    "feature_scores": candidate.feature_scores,
                    "degradation_flags": candidate.degradation_flags,
                    "deterministic_score": candidate.deterministic_score,
                }
                for candidate in candidates
            ]
        if mode == SearchMode.REPLY:
            prompt = (
                "你是梗圖搜尋 reranker，目前處於「回覆模式」。"
                "先尊重 deterministic_score 與 feature_scores 的排序，再判斷哪張梗圖最適合當作回覆。"
                "評分重點是 ocr_text 是否構成機智回應、反駁或吐槽；"
                "如果 query_analysis.preferred_tone 有值，請優先選擇更符合該梗圖語氣偏好的候選。"
                "若 ocr_status 不是 success，除非其他訊號特別強，否則應降低分數。"
                "背景或場景資訊只可當作次要 tie-break，不可蓋過台詞本身。"
                "回傳 JSON 物件，欄位為 results，內容是陣列，每個元素有 image_id, score, reason。"
                "reason 必須是繁體中文且解釋語氣與情境。"
            )
        else:
            prompt = (
                "你是梗圖搜尋 reranker。請依照 query 與候選 metadata 選出語意最相近的結果。"
                "先尊重 deterministic_score 與 feature_scores 的排序，再判斷哪張梗圖最符合語意。"
                "如果 query_analysis.preferred_tone 有值，請把它視為次要偏好，協助挑選更接近使用者想要語氣的結果。"
                "重要：梗圖的整體語意、情境描述與情緒標籤的匹配度是最關鍵的排序依據。"
                "回傳 JSON 物件，欄位為 results，內容是陣列，每個元素有 image_id, score, reason。"
                "reason 必須是繁體中文且解釋語氣與情境。"
            )
        data = self._json_completion(
            prompt,
            json.dumps(
                {
                    "query": query,
                    "query_analysis": query_analysis.model_dump(),
                    "candidates": serialized,
                    "top_n": top_n,
                },
                ensure_ascii=False,
            ),
            self.profile.chat_model,
        )
        return [RerankResult(**item) for item in data.get("results", [])[:top_n]]
