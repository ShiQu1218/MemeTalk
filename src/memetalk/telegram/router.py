from __future__ import annotations

import logging
from typing import Literal, Protocol

from pydantic import BaseModel, field_validator, model_validator

from memetalk.config import AppSettings
from memetalk.providers.json_utils import extract_json_object
from memetalk.providers.openai_provider import (
    CompatibleProviderProfile,
    build_gemini_profile,
    build_llama_cpp_profile,
    build_lmstudio_profile,
    build_ollama_profile,
    build_openai_profile,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是 MemeTalk 的 Telegram 聊天機器人。你的任務是判斷使用者的訊息應該用什麼方式回覆。

你有三種回覆方式：
- "text"：純文字回覆。適合一般問答、需要詳細解釋或查詢資訊的訊息。
- "meme"：用梗圖回覆。適合抱怨、吐槽、情緒性發言、玩笑、反應型對話。
- "both"：同時用文字和梗圖回覆。適合需要接話，但也很適合配梗圖的情境。

回覆規則：
1. 不是所有訊息都要用梗圖，只在適合的時候用。
2. 如果訊息帶有明顯情緒、吐槽、無奈、尷尬、崩潰、炫耀或玩笑感，優先考慮 meme 或 both。
3. 如果是正經問題、資訊查詢、步驟教學、知識型問題，優先用 text。
4. search_query 應該是能在 MemeTalk 找到適合梗圖的短關鍵詞或短語。
5. search_mode 固定用 "reply"。

你必須回傳 JSON 物件，欄位如下：
{
  "action": "text" | "meme" | "both",
  "text_response": "當 action 為 text 或 both 時必填",
  "search_query": "當 action 為 meme 或 both 時必填",
  "search_mode": "reply"
}

只回傳單一 JSON 物件。不要使用 Markdown、不要加前言、不要加結尾說明。
"""


class TelegramDecision(BaseModel):
    action: Literal["text", "meme", "both"]
    text_response: str | None = None
    search_query: str | None = None
    search_mode: Literal["reply", "semantic"] = "reply"

    @model_validator(mode="after")
    def validate_required_fields(self) -> "TelegramDecision":
        if self.action in {"text", "both"} and not (self.text_response or "").strip():
            raise ValueError("text_response is required when action is text or both.")
        if self.action in {"meme", "both"} and not (self.search_query or "").strip():
            raise ValueError("search_query is required when action is meme or both.")
        return self


class TelegramConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Telegram conversation history entries must not be empty.")
        return trimmed


class TelegramRouter(Protocol):
    name: str

    async def decide(
        self,
        user_message: str,
        conversation_history: list[TelegramConversationMessage] | None = None,
    ) -> TelegramDecision:
        raise NotImplementedError


class MockTelegramRouter:
    name = "mock-telegram-router"

    async def decide(
        self,
        user_message: str,
        conversation_history: list[TelegramConversationMessage] | None = None,
    ) -> TelegramDecision:
        text = user_message.strip()
        lowered = text.lower()
        emotive_terms = (
            "氣死",
            "崩潰",
            "傻眼",
            "笑死",
            "哭",
            "無言",
            "扯",
            "靠北",
            "尷尬",
            "煩",
            "爽",
            "可悲",
            "xddd",
            "lol",
        )
        has_emotion = any(term in text or term in lowered for term in emotive_terms)
        looks_like_question = "?" in text or "？" in text or "嗎" in text or "怎麼" in text

        if has_emotion and len(text) <= 18:
            return TelegramDecision(action="meme", search_query=text)
        if has_emotion:
            return TelegramDecision(
                action="both",
                text_response="這句很適合直接配一張圖回去。",
                search_query=text,
            )
        if looks_like_question:
            return TelegramDecision(
                action="text",
                text_response="這句比較像要直接回文字，我先用文字接住。",
            )
        return TelegramDecision(
            action="both",
            text_response="我先幫你接一句，再找張適合的梗圖。",
            search_query=text,
        )


class OpenAICompatibleTelegramRouter:
    def __init__(self, profile: CompatibleProviderProfile) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Telegram chat requires the openai extra. Install with `pip install -e .[openai]`.") from exc

        client_kwargs: dict[str, str] = {"api_key": _resolve_compatible_api_key(profile)}
        if profile.base_url:
            client_kwargs["base_url"] = profile.base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = profile.chat_model
        self.name = f"{profile.label}-telegram-router"

    async def decide(
        self,
        user_message: str,
        conversation_history: list[TelegramConversationMessage] | None = None,
    ) -> TelegramDecision:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await self._call_llm(user_message, conversation_history=conversation_history)
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("Telegram router call failed, retrying once", exc_info=True)
                    continue
                break
        raise RuntimeError("Telegram router failed to decide a response.") from last_error

    async def _call_llm(
        self,
        user_message: str,
        conversation_history: list[TelegramConversationMessage] | None = None,
    ) -> TelegramDecision:
        messages = build_conversation_messages(user_message, conversation_history)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
        except Exception:
            logger.info("json_object format unavailable for Telegram router, falling back to prompt-only JSON", exc_info=True)
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.3,
            )
        payload = response.choices[0].message.content or "{}"
        return TelegramDecision.model_validate(extract_json_object(payload))


class ClaudeTelegramRouter:
    def __init__(self, settings: AppSettings) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Telegram chat requires the anthropic extra when provider_backend=claude.") from exc

        if not settings.claude_api_key:
            raise RuntimeError("MEMETALK_CLAUDE_API_KEY is required when Telegram chat uses the Claude provider.")

        self._client = anthropic.AsyncAnthropic(api_key=settings.claude_api_key)
        self._model = settings.claude_chat_model
        self.name = "claude-telegram-router"

    async def decide(
        self,
        user_message: str,
        conversation_history: list[TelegramConversationMessage] | None = None,
    ) -> TelegramDecision:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await self._call_llm(user_message, conversation_history=conversation_history)
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("Claude Telegram router call failed, retrying once", exc_info=True)
                    continue
                break
        raise RuntimeError("Claude Telegram router failed to decide a response.") from last_error

    async def _call_llm(
        self,
        user_message: str,
        conversation_history: list[TelegramConversationMessage] | None = None,
    ) -> TelegramDecision:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=build_anthropic_messages(user_message, conversation_history),
        )
        payload = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return TelegramDecision.model_validate(extract_json_object(payload))


def build_telegram_router(settings: AppSettings) -> TelegramRouter:
    if settings.provider_backend == "mock":
        return MockTelegramRouter()
    if settings.provider_backend == "claude":
        return ClaudeTelegramRouter(settings)
    if settings.provider_backend == "local":
        raise RuntimeError("Telegram chat does not support provider_backend=local.")

    profile_builder = {
        "openai": build_openai_profile,
        "lmstudio": build_lmstudio_profile,
        "ollama": build_ollama_profile,
        "llama_cpp": build_llama_cpp_profile,
        "gemini": build_gemini_profile,
    }.get(settings.provider_backend)
    if profile_builder is None:
        raise RuntimeError(f"Unsupported Telegram provider backend: {settings.provider_backend}")
    return OpenAICompatibleTelegramRouter(profile_builder(settings))


def _resolve_compatible_api_key(profile: CompatibleProviderProfile) -> str:
    if profile.api_key:
        return profile.api_key
    if profile.base_url:
        return "lm-studio"
    raise RuntimeError(
        f"Telegram chat requires an API key for provider `{profile.label}` unless a compatible base URL is configured."
    )


def build_conversation_messages(
    user_message: str,
    conversation_history: list[TelegramConversationMessage] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in conversation_history or []:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": user_message.strip()})
    return messages


def build_anthropic_messages(
    user_message: str,
    conversation_history: list[TelegramConversationMessage] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in conversation_history or []:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": user_message.strip()})
    return messages
