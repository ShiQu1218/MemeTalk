import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from memetalk.config import AppSettings
from memetalk.core.models import QueryAnalysis, SearchMode, SearchResponse, SearchResult, SearchResultDebug, SearchTrace
from memetalk.telegram.bot import validate_telegram_settings
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
