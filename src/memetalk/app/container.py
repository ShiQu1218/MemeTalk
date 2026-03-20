from __future__ import annotations

from dataclasses import dataclass

from memetalk.app.evaluation import EvaluationService
from memetalk.app.indexer import IndexingService
from memetalk.app.search import SearchService
from memetalk.config import AppSettings
from memetalk.core.providers import ProviderBundle
from memetalk.providers.registry import build_provider_bundle
from memetalk.storage.sqlite_store import SQLiteMemeRepository
from memetalk.storage.vector_store import ChromaVectorStore, InMemoryVectorStore, VectorStore


@dataclass(slots=True)
class AppContainer:
    settings: AppSettings
    repository: SQLiteMemeRepository
    vector_store: VectorStore
    providers: ProviderBundle
    indexing_service: IndexingService
    search_service: SearchService
    evaluation_service: EvaluationService


def build_vector_store(settings: AppSettings) -> VectorStore:
    if settings.vector_backend == "memory":
        return InMemoryVectorStore()
    return ChromaVectorStore(settings.chroma_path, settings.chroma_collection)


def build_container(settings: AppSettings | None = None) -> AppContainer:
    active_settings = settings or AppSettings.from_env()
    active_settings.ensure_runtime_dirs()
    repository = SQLiteMemeRepository(active_settings.sqlite_path)
    repository.initialize()
    vector_store = build_vector_store(active_settings)
    providers = build_provider_bundle(active_settings)
    indexing_service = IndexingService(repository, vector_store, providers)
    search_service = SearchService(repository, vector_store, providers, active_settings.api_base_url)
    evaluation_service = EvaluationService(search_service, active_settings.search_candidate_k_default)
    return AppContainer(
        settings=active_settings,
        repository=repository,
        vector_store=vector_store,
        providers=providers,
        indexing_service=indexing_service,
        search_service=search_service,
        evaluation_service=evaluation_service,
    )
