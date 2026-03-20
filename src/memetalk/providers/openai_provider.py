from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from memetalk.config import AppSettings
from memetalk.core.models import MemeMetadata, OCRExtraction, QueryAnalysis, RerankCandidate, RerankResult
from memetalk.core.providers import EmbeddingProvider, MetadataProvider, QueryAnalyzer, Reranker


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

    def _json_completion(self, prompt: str, user_content: str | list[dict[str, Any]], model: str) -> dict[str, Any]:
        response = self._client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{prompt}"
                        "只回傳單一 JSON 物件。"
                        "不要使用 Markdown、不要加前言、不要加結尾說明。"
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        )
        payload = response.choices[0].message.content or ""
        return _extract_json_object(payload)


class CompatibleEmbeddingProvider(_OpenAICompatibleBase, EmbeddingProvider):
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        super().__init__(profile)
        self.name = f"{profile.label}-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client().embeddings.create(model=self.profile.embedding_model, input=texts)
        return [item.embedding for item in response.data]


class CompatibleQueryAnalyzer(_OpenAICompatibleBase, QueryAnalyzer):
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        super().__init__(profile)
        self.name = f"{profile.label}-query-analyzer"

    def analyze_query(self, query: str) -> QueryAnalysis:
        prompt = (
            "你是繁體中文的梗圖查詢分析器。"
            "請把使用者輸入分析成 JSON，欄位為 situation, emotions, tone, reply_intent, query_embedding_text。"
            "query_embedding_text 必須是適合向量搜尋的繁體中文敘述。"
        )
        data = self._json_completion(prompt, query, self.profile.chat_model)
        return QueryAnalysis(
            original_query=query,
            situation=data["situation"],
            emotions=data.get("emotions", []),
            tone=data["tone"],
            reply_intent=data["reply_intent"],
            query_embedding_text=data["query_embedding_text"],
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
    ) -> list[RerankResult]:
        serialized = [
            {
                "image_id": candidate.image_id,
                "vector_score": candidate.vector_score,
                "template_name": candidate.metadata.template_name,
                "scene_description": candidate.metadata.scene_description,
                "meme_usage": candidate.metadata.meme_usage,
                "emotion_tags": candidate.metadata.emotion_tags,
                "intent_tags": candidate.metadata.intent_tags,
            }
            for candidate in candidates
        ]
        prompt = (
            "你是梗圖搜尋 reranker。請依照 query 與候選 metadata 選出最適合回覆的結果。"
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
