import json
from pathlib import Path

from PIL import Image

from memetalk.app.container import build_container
from memetalk.app.indexer import _sha256_file
from memetalk.config import AppSettings
from memetalk.core.models import SearchMode, SearchResponse, SearchResult, SearchTrace
from memetalk.core.models import SearchScoringProfile
from memetalk.core.retrieval import default_search_scoring_profile


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


def test_evaluation_service_reports_precision_and_mrr(tmp_path: Path) -> None:
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
    target_path = source_dir / "friends_late_text.png"
    other_path = source_dir / "office_burnout.png"
    _create_image(target_path, (255, 0, 0))
    _create_image(other_path, (0, 255, 0))
    container.indexing_service.build_index(source_dir)

    target_id = container.repository.get_asset_by_sha256(_sha256_file(target_path)).image_id
    other_id = container.repository.get_asset_by_sha256(_sha256_file(other_path)).image_id

    cases_path = tmp_path / "eval_cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "reply-late",
                    "query": "朋友說快到了但還沒出門",
                    "mode": SearchMode.REPLY.value,
                    "positives": [target_id],
                    "hard_negatives": [other_id],
                    "top_k": 2,
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cases = container.evaluation_service.load_cases(cases_path)
    report = container.evaluation_service.run_cases(cases)

    assert report.case_count == 1
    assert report.precision_at_k > 0
    assert report.mrr == 1.0
    assert report.hard_negative_hit_rate == 0.5
    assert report.cases[0].case_id == "reply-late"


class FakeSearchService:
    def __init__(self) -> None:
        self.scoring_profile = default_search_scoring_profile()

    def set_scoring_profile(self, profile: SearchScoringProfile) -> None:
        self.scoring_profile = profile

    def search(
        self,
        query: str,
        top_n: int,
        candidate_k: int,
        mode: SearchMode = SearchMode.REPLY,
        preferred_tone: str | None = None,
        rerank_enabled: bool = True,
    ) -> SearchResponse:
        positive_score = self.scoring_profile.reply.semantic_vector * 6.0
        negative_score = self.scoring_profile.reply.keyword_route * 5.0
        ordered = [
            SearchResult(
                image_id="positive" if positive_score >= negative_score else "negative",
                image_url="http://testserver/api/v1/assets/positive",
                reason="fake",
                score=max(positive_score, negative_score),
            ),
            SearchResult(
                image_id="negative" if positive_score >= negative_score else "positive",
                image_url="http://testserver/api/v1/assets/negative",
                reason="fake",
                score=min(positive_score, negative_score),
            ),
        ]
        return SearchResponse(
            query_analysis={
                "original_query": query,
                "situation": query,
                "emotions": [],
                "tone": "吐槽",
                "reply_intent": "回應",
                "preferred_tone": preferred_tone,
                "query_embedding_text": query,
                "query_terms": [query],
                "template_hints": [],
                "retrieval_weights": {
                    "semantic": 1.0,
                    "reply_text": 1.0,
                    "keyword": 1.0,
                    "template": 1.0,
                },
            },
            results=ordered[:top_n],
            provider_trace={},
            search_trace=SearchTrace(rerank_strategy="deterministic_only"),
        )


def test_evaluation_service_can_tune_scoring_profile() -> None:
    from memetalk.app.evaluation import EvaluationCase, EvaluationService

    service = EvaluationService(FakeSearchService(), candidate_k_default=5)
    cases = [
        EvaluationCase(
            case_id="needs-semantic-boost",
            query="給我一張最準的回覆圖",
            mode=SearchMode.REPLY,
            positives=["positive"],
            hard_negatives=["negative"],
            top_k=1,
        )
    ]

    tuning_report = service.tune_scoring_profile(cases, passes=2)

    assert tuning_report.best_objective > tuning_report.baseline_objective
    assert tuning_report.tuned_report.precision_at_k == 1.0
    assert tuning_report.best_profile.reply.semantic_vector > default_search_scoring_profile().reply.semantic_vector
