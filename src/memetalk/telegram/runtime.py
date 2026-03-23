from __future__ import annotations

import asyncio
from pathlib import Path

from memetalk.app.container import AppContainer
from memetalk.core.models import SearchMode, SearchResult


class DirectTelegramSearchClient:
    def __init__(self, container: AppContainer, candidate_k: int) -> None:
        self._container = container
        self._candidate_k = candidate_k

    async def search_memes(
        self,
        query: str,
        mode: str = "reply",
        top_n: int = 3,
    ) -> list[SearchResult]:
        search_mode = SearchMode(mode)
        response = await asyncio.to_thread(
            self._container.search_service.search,
            query,
            top_n,
            max(self._candidate_k, top_n),
            search_mode,
        )
        return response.results

    async def get_meme_image(self, image_id: str) -> bytes:
        asset = await asyncio.to_thread(self._container.repository.get_asset_by_id, image_id)
        if asset is None:
            raise FileNotFoundError(f"Meme asset not found: {image_id}")
        return await asyncio.to_thread(Path(asset.file_path).read_bytes)

    async def close(self) -> None:
        return None
