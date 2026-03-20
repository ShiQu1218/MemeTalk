from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from memetalk.app.search import SearchService
from memetalk.core.models import SearchMode


class EvaluationCase(BaseModel):
    case_id: str
    query: str
    mode: SearchMode = SearchMode.REPLY
    positives: list[str] = Field(default_factory=list)
    hard_negatives: list[str] = Field(default_factory=list)
    top_k: int = 3


class EvaluationCaseResult(BaseModel):
    case_id: str
    mode: SearchMode
    top_results: list[str]
    precision_at_k: float
    mrr: float
    first_positive_rank: int | None = None
    hard_negative_hits: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    case_count: int
    precision_at_k: float
    mrr: float
    cases: list[EvaluationCaseResult]


class EvaluationService:
    def __init__(self, search_service: SearchService, candidate_k_default: int) -> None:
        self.search_service = search_service
        self.candidate_k_default = candidate_k_default

    def load_cases(self, path: Path) -> list[EvaluationCase]:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
        return [EvaluationCase(**item) for item in raw_cases]

    def run_cases(self, cases: list[EvaluationCase]) -> EvaluationReport:
        results: list[EvaluationCaseResult] = []
        for case in cases:
            response = self.search_service.search(
                query=case.query,
                top_n=case.top_k,
                candidate_k=max(self.candidate_k_default, case.top_k * 3),
                mode=case.mode,
            )
            top_results = [result.image_id for result in response.results[: case.top_k]]
            positives = set(case.positives)
            precision_hits = sum(1 for image_id in top_results if image_id in positives)
            first_positive_rank = next((index + 1 for index, image_id in enumerate(top_results) if image_id in positives), None)
            mrr = (1 / first_positive_rank) if first_positive_rank else 0.0
            hard_negative_hits = [image_id for image_id in top_results if image_id in set(case.hard_negatives)]
            results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    mode=case.mode,
                    top_results=top_results,
                    precision_at_k=(precision_hits / case.top_k) if case.top_k else 0.0,
                    mrr=mrr,
                    first_positive_rank=first_positive_rank,
                    hard_negative_hits=hard_negative_hits,
                )
            )
        case_count = len(results)
        return EvaluationReport(
            case_count=case_count,
            precision_at_k=(sum(item.precision_at_k for item in results) / case_count) if case_count else 0.0,
            mrr=(sum(item.mrr for item in results) / case_count) if case_count else 0.0,
            cases=results,
        )
