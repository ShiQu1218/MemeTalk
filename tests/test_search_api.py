import base64
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from memetalk.api.main import create_app
from memetalk.app.container import AppContainer, build_container
from memetalk.app.search import FALLBACK_REASON
from memetalk.config import AppSettings
from memetalk.core.models import QueryAnalysis, RerankCandidate, RerankResult, SearchMode
from memetalk.core.providers import ProviderBundle, Reranker
from memetalk.core.retrieval import default_retrieval_weights


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


def _build_container(tmp_path: Path) -> AppContainer:
    settings = AppSettings(
        sqlite_path=tmp_path / "memetalk.sqlite3",
        vector_backend="memory",
        provider_backend="mock",
        ocr_backend="mock",
        api_base_url="http://testserver",
    )
    container = build_container(settings)
    source_dir = tmp_path / "memes"
    source_dir.mkdir()
    _create_image(source_dir / "friends_late_text.png", (255, 0, 0))
    _create_image(source_dir / "office_burnout.png", (0, 255, 0))
    _create_image(source_dir / "shocked_reaction.png", (0, 0, 255))
    container.indexing_service.build_index(source_dir)
    return container


class BrokenReranker(Reranker):
    name = "broken-reranker"

    def rerank(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[RerankCandidate],
        top_n: int,
        mode: SearchMode = SearchMode.REPLY,
    ) -> list[RerankResult]:
        raise RuntimeError("rerank unavailable")


def test_health_and_search_endpoints_return_expected_shape(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    client = TestClient(create_app(container=container))

    health_response = client.get("/api/v1/health")
    search_response = client.post(
        "/api/v1/search",
        json={"query": "朋友說快到了但還沒出門", "top_n": 3, "candidate_k": 8, "preferred_tone": "陰陽怪氣"},
    )

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert search_response.status_code == 200
    payload = search_response.json()
    assert set(payload.keys()) == {"query_analysis", "results", "provider_trace", "search_trace"}
    assert len(payload["results"]) == 3
    assert payload["query_analysis"]["preferred_tone"] == "陰陽怪氣"
    assert {"routes_used", "candidate_counts", "rerank_strategy"} <= set(payload["search_trace"].keys())
    assert {"image_id", "image_url", "reason", "score", "template_name", "emotion_tags", "intent_tags", "debug"} <= set(
        payload["results"][0].keys()
    )


def test_asset_endpoint_serves_indexed_image(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    client = TestClient(create_app(container=container))

    search_payload = client.post(
        "/api/v1/search",
        json={"query": "上班好厭世", "top_n": 3, "candidate_k": 8},
    ).json()
    image_id = search_payload["results"][0]["image_id"]

    asset_response = client.get(f"/api/v1/assets/{image_id}")

    assert asset_response.status_code == 200
    assert asset_response.headers["content-type"].startswith("image/")


def test_search_endpoint_accepts_image_only_query(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    client = TestClient(create_app(container=container))
    query_image = tmp_path / "friends_late_text_reference.png"
    _create_image(query_image, (100, 100, 100))

    response = client.post(
        "/api/v1/search",
        json={
            "query_image_base64": base64.b64encode(query_image.read_bytes()).decode("utf-8"),
            "query_image_filename": query_image.name,
            "query_image_media_type": "image/png",
            "top_n": 3,
            "candidate_k": 8,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    assert payload["query_analysis"]["query_terms"]


def test_search_endpoint_rejects_missing_text_and_image(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    client = TestClient(create_app(container=container))

    response = client.post(
        "/api/v1/search",
        json={"query": "   ", "top_n": 3, "candidate_k": 8},
    )

    assert response.status_code == 422


def test_search_falls_back_when_reranker_fails(tmp_path: Path) -> None:
    container = _build_container(tmp_path)
    providers = container.providers
    container.search_service.providers = ProviderBundle(
        ocr_provider=providers.ocr_provider,
        metadata_provider=providers.metadata_provider,
        embedding_provider=providers.embedding_provider,
        query_analyzer=providers.query_analyzer,
        reranker=BrokenReranker(),
    )
    client = TestClient(create_app(container=container))

    response = client.post(
        "/api/v1/search",
        json={"query": "主管又突然改需求", "top_n": 3, "candidate_k": 8},
    )

    assert response.status_code == 200
    payload = response.json()
    reasons = [item["reason"] for item in payload["results"]]
    assert reasons and all(reason == FALLBACK_REASON for reason in reasons)
    assert payload["search_trace"]["rerank_strategy"] == "deterministic_fallback"
