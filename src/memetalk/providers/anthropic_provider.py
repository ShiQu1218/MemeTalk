from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from memetalk.config import AppSettings
from memetalk.core.models import MemeMetadata, OCRExtraction, OCRStatus, QueryAnalysis, RerankCandidate, RerankResult, SearchMode
from memetalk.core.providers import MetadataProvider, QueryAnalyzer, Reranker
from memetalk.core.retrieval import default_retrieval_weights

JSON_COMPLETION_MAX_ATTEMPTS = 3


def _build_image_content(image_path: Path) -> dict[str, Any]:
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix)
    if media_type is None or suffix == ".webp":
        buffer = io.BytesIO()
        with Image.open(image_path) as img:
            img.save(buffer, format="PNG")
        data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        media_type = "image/png"
    else:
        data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


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


class _AnthropicBase:
    def __init__(self, api_key: str | None, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic extra is not installed. Install with `pip install -e .[anthropic]`."
            ) from exc
        if not self._api_key:
            raise RuntimeError("MEMETALK_CLAUDE_API_KEY is required for the Claude provider.")
        return anthropic.Anthropic(api_key=self._api_key)

    def _json_completion(
        self,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(JSON_COMPLETION_MAX_ATTEMPTS):
            retry_note = ""
            if attempt:
                retry_note = (
                    "上一次輸出的 JSON 格式錯誤。"
                    "這次所有欄位與陣列元素之間都必須使用半形逗號，"
                    "並確保輸出可以被 Python json.loads 直接解析。"
                )
            full_system = (
                f"{system_prompt}{retry_note}"
                "只回傳單一 JSON 物件。"
                "不要使用 Markdown、不要加前言、不要加結尾說明。"
            )
            if isinstance(user_content, str):
                messages = [{"role": "user", "content": user_content}]
            else:
                messages = [{"role": "user", "content": user_content}]
            try:
                response = self._client().messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=full_system,
                    messages=messages,
                )
            except Exception as exc:
                raise RuntimeError(f"Claude API 呼叫失敗：{exc}") from exc
            payload = ""
            for block in response.content:
                if block.type == "text":
                    payload += block.text
            try:
                return _extract_json_object(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Claude returned malformed JSON after {JSON_COMPLETION_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error


class AnthropicQueryAnalyzer(_AnthropicBase, QueryAnalyzer):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings.claude_api_key, settings.claude_chat_model)
        self.name = "claude-query-analyzer"

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
            json.dumps({"query": query, "preferred_tone": preferred_tone}, ensure_ascii=False),
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


class AnthropicMetadataProvider(_AnthropicBase, MetadataProvider):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings.claude_api_key, settings.claude_vision_model)
        self.name = "claude-metadata"

    def analyze_image(self, image_path: Path, ocr_hint: OCRExtraction | None = None) -> MemeMetadata:
        prompt = (
            "你是繁體中文的梗圖分析器，需要一次完成 OCR 辨識與 metadata 分析。"
            "請回傳 JSON，欄位如下："
            "ocr_text（圖片上的所有文字，如果沒有文字則為空字串）、"
            "ocr_lines（文字逐行陣列）、"
            "template_name、scene_description、meme_usage、"
            "visual_description（用繁體中文描述圖片的視覺構圖、人物表情、色調風格等審美特徵）、"
            "aesthetic_tags（視覺風格短標籤陣列，例如「對比構圖」「表情誇張」「二段式」）、"
            "usage_scenario（用繁體中文描述這張梗圖最適合在什麼對話情境下使用）、"
            "emotion_tags、intent_tags、style_tags。"
            "請仔細辨識圖片上的所有文字，包含中文、英文、符號。"
        )
        hint_text = ""
        if ocr_hint and ocr_hint.text:
            hint_text = f"參考 OCR 提示：{ocr_hint.text}\n請驗證並修正上述 OCR 結果。"
        user_content: list[dict[str, Any]] = []
        if hint_text:
            user_content.append({"type": "text", "text": hint_text})
        user_content.append(_build_image_content(image_path))
        data = self._json_completion(prompt, user_content)
        ocr_text = (data.get("ocr_text") or "").strip()
        ocr_lines = data.get("ocr_lines") or []
        if not ocr_lines and ocr_text:
            ocr_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        has_text = bool(ocr_text)
        ocr_status = OCRStatus.SUCCESS if has_text else OCRStatus.EMPTY
        return MemeMetadata(
            has_text=has_text,
            ocr_text=ocr_text,
            ocr_status=ocr_status,
            ocr_confidence=None,
            ocr_lines=ocr_lines,
            template_name=data.get("template_name"),
            scene_description=data.get("scene_description", ""),
            meme_usage=data.get("meme_usage", ""),
            visual_description=data.get("visual_description", ""),
            aesthetic_tags=data.get("aesthetic_tags", []),
            usage_scenario=data.get("usage_scenario", ""),
            emotion_tags=data.get("emotion_tags", []),
            intent_tags=data.get("intent_tags", []),
            style_tags=data.get("style_tags", []),
        )


class AnthropicReranker(_AnthropicBase, Reranker):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings.claude_api_key, settings.claude_chat_model)
        self.name = "claude-reranker"

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
                    "image_id": c.image_id,
                    "vector_score": c.vector_score,
                    "template_name": c.metadata.template_name,
                    "meme_usage": c.metadata.meme_usage,
                    "ocr_text": c.metadata.ocr_text,
                    "ocr_status": c.metadata.ocr_status,
                    "intent_tags": c.metadata.intent_tags,
                    "candidate_sources": c.candidate_sources,
                    "feature_scores": c.feature_scores,
                    "degradation_flags": c.degradation_flags,
                    "deterministic_score": c.deterministic_score,
                }
                for c in candidates
            ]
            prompt = (
                "你是梗圖搜尋 reranker，目前處於「回覆模式」。"
                "先尊重 deterministic_score 與 feature_scores 的排序，再判斷哪張梗圖最適合當作回覆。"
                "評分重點是 ocr_text 是否構成機智回應、反駁或吐槽；"
                "如果 query_analysis.preferred_tone 有值，請優先選擇更符合該梗圖語氣偏好的候選。"
                "若 ocr_status 是 success 但 ocr_text 與查詢無關（degradation_flags 含 reply_ocr_mismatch），應降低分數。"
                "無 OCR 的梗圖若視覺表達與查詢高度相關，不需降低分數。"
                "背景或場景資訊只可當作次要 tie-break，不可蓋過台詞本身。"
                "回傳 JSON 物件，欄位為 results，內容是陣列，每個元素有 image_id, score, reason。"
                "reason 必須是繁體中文且解釋語氣與情境。"
            )
        else:
            serialized = [
                {
                    "image_id": c.image_id,
                    "vector_score": c.vector_score,
                    "template_name": c.metadata.template_name,
                    "scene_description": c.metadata.scene_description,
                    "meme_usage": c.metadata.meme_usage,
                    "ocr_text": c.metadata.ocr_text,
                    "ocr_status": c.metadata.ocr_status,
                    "emotion_tags": c.metadata.emotion_tags,
                    "intent_tags": c.metadata.intent_tags,
                    "candidate_sources": c.candidate_sources,
                    "feature_scores": c.feature_scores,
                    "degradation_flags": c.degradation_flags,
                    "deterministic_score": c.deterministic_score,
                }
                for c in candidates
            ]
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
        )
        return [RerankResult(**item) for item in data.get("results", [])[:top_n]]
