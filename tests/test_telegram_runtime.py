import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from memetalk.config import AppSettings
from memetalk.core.models import QueryAnalysis, SearchMode, SearchResponse, SearchResult, SearchResultDebug, SearchTrace
from memetalk.telegram.bot import (
    TELEGRAM_HISTORY_LIMIT,
    _append_conversation_history,
    _load_conversation_history,
    _send_meme_only_reply,
    validate_telegram_settings,
)
from memetalk.telegram.router import TelegramConversationMessage, TelegramDecision, build_conversation_messages
from memetalk.telegram.runtime import DirectTelegramSearchClient


class FakeSearchService:
    def __init__(self, response: SearchResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, int, int, SearchMode]] = []

    def search(
        self,
        query: str,
        top_n: int,
        candidate_k: int,
        mode: SearchMode,
    ) -> SearchResponse:
        self.calls.append((query, top_n, candidate_k, mode))
        return self.response


class FakeRepository:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.requested_ids: list[str] = []

    def get_asset_by_id(self, image_id: str):
        self.requested_ids.append(image_id)
        return SimpleNamespace(file_path=str(self.file_path))


class FakeMessage:
    def __init__(self) -> None:
        self.photo_calls: list[dict[str, object]] = []
        self.text_calls: list[str] = []

    async def reply_photo(self, *, photo, caption=None) -> None:
        self.photo_calls.append({"photo": photo, "caption": caption})

    async def reply_text(self, text: str) -> None:
        self.text_calls.append(text)


class FakeTelegramSearchClient:
    def __init__(self, results: list[SearchResult], image_bytes: bytes) -> None:
        self.results = results
        self.image_bytes = image_bytes

    async def search_memes(self, query: str, mode: str = "reply", top_n: int = 3) -> list[SearchResult]:
        return self.results

    async def get_meme_image(self, image_id: str) -> bytes:
        return self.image_bytes


def test_validate_telegram_settings_requires_enable_flag_and_token() -> None:
    with pytest.raises(RuntimeError, match="disabled"):
        validate_telegram_settings(AppSettings())

    with pytest.raises(RuntimeError, match="token"):
        validate_telegram_settings(AppSettings(telegram_enabled=True))

    assert validate_telegram_settings(AppSettings(telegram_enabled=True, telegram_bot_token=" 123:token ")) == "123:token"


def test_direct_telegram_search_client_uses_search_service_and_repository(tmp_path: Path) -> None:
    image_path = tmp_path / "meme.png"
    image_path.write_bytes(b"fake-image")

    response = SearchResponse(
        query_analysis=QueryAnalysis(
            original_query="今天好崩潰",
            situation="工作出錯",
            emotions=["崩潰"],
            tone="吐槽",
            reply_intent="回嘴",
            query_embedding_text="找一張崩潰吐槽的圖",
            query_terms=["崩潰"],
            template_hints=[],
        ),
        results=[
            SearchResult(
                image_id="img-1",
                image_url="http://localhost/api/v1/assets/img-1",
                reason="台詞和情緒都很像在回嘴",
                score=0.92,
                template_name="崩潰貓",
                emotion_tags=["崩潰"],
                intent_tags=["吐槽"],
                debug=SearchResultDebug(candidate_sources=["reply_text"]),
            )
        ],
        provider_trace={"query_analyzer": "mock"},
        search_trace=SearchTrace(routes_used=["reply_text"]),
    )
    search_service = FakeSearchService(response)
    repository = FakeRepository(image_path)
    container = SimpleNamespace(search_service=search_service, repository=repository)
    client = DirectTelegramSearchClient(container, candidate_k=12)

    results = asyncio.run(client.search_memes("今天好崩潰", top_n=3))
    image_bytes = asyncio.run(client.get_meme_image("img-1"))

    assert [item.image_id for item in results] == ["img-1"]
    assert search_service.calls == [("今天好崩潰", 3, 12, SearchMode.REPLY)]
    assert repository.requested_ids == ["img-1"]
    assert image_bytes == b"fake-image"


def test_send_meme_only_reply_sends_photo_without_caption_or_extra_text() -> None:
    message = FakeMessage()
    results = [
        SearchResult(
            image_id="img-1",
            image_url="http://localhost/api/v1/assets/img-1",
            reason="這段 reason 不應該出現在 Telegram caption",
            score=0.91,
            template_name="崩潰貓",
            emotion_tags=["崩潰"],
            intent_tags=["吐槽"],
            debug=SearchResultDebug(candidate_sources=["reply_text"]),
        )
    ]
    search_client = FakeTelegramSearchClient(results, b"image-bytes")

    sent, history_summary = asyncio.run(
        _send_meme_only_reply(
            message,
            TelegramDecision(action="both", text_response="這段文字不應該另外送出", search_query="崩潰"),
            search_client,
        )
    )

    assert sent is True
    assert history_summary == "已用梗圖回覆（搜尋詞：崩潰）"
    assert len(message.photo_calls) == 1
    assert message.photo_calls[0]["caption"] is None
    assert message.text_calls == []


def test_conversation_history_is_bounded_and_normalized() -> None:
    chat_data: dict[str, object] = {}

    for index in range(TELEGRAM_HISTORY_LIMIT + 3):
        _append_conversation_history(chat_data, "user", f" user-{index} ")

    history = _load_conversation_history(chat_data)

    assert len(history) == TELEGRAM_HISTORY_LIMIT
    assert history[0].content == "user-3"
    assert history[-1].content == f"user-{TELEGRAM_HISTORY_LIMIT + 2}"


def test_build_conversation_messages_includes_recent_history_before_current_user() -> None:
    history = [
        TelegramConversationMessage(role="user", content="昨天真的很慘"),
        TelegramConversationMessage(role="assistant", content="已用梗圖回覆（搜尋詞：崩潰）"),
    ]

    messages = build_conversation_messages("今天又來一次", history)

    assert messages[0]["role"] == "system"
    assert messages[1:] == [
        {"role": "user", "content": "昨天真的很慘"},
        {"role": "assistant", "content": "已用梗圖回覆（搜尋詞：崩潰）"},
        {"role": "user", "content": "今天又來一次"},
    ]
