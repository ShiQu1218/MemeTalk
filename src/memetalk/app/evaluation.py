from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from memetalk.core.models import DeterministicModeScoringProfile, SearchMode, SearchResponse, SearchScoringProfile

logger = logging.getLogger(__name__)


@runtime_checkable
class SearchServiceProtocol(Protocol):
    scoring_profile: SearchScoringProfile

    def search(
        self,
        query: str,
        top_n: int,
        candidate_k: int,
        mode: SearchMode = ...,
        preferred_tone: str | None = ...,
        rerank_enabled: bool = ...,
    ) -> SearchResponse: ...

    def set_scoring_profile(self, profile: SearchScoringProfile) -> None: ...


class EvaluationCase(BaseModel):
    case_id: str
    query: str
    mode: SearchMode = SearchMode.REPLY
    preferred_tone: str | None = None
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
    hard_negative_hit_rate: float = 0.0
    cases: list[EvaluationCaseResult]


class EvaluationTuningReport(BaseModel):
    baseline_objective: float
    best_objective: float
    baseline_report: EvaluationReport
    tuned_report: EvaluationReport
    best_profile: SearchScoringProfile
    passes: int


class EvaluationService:
    def __init__(self, search_service: SearchServiceProtocol, candidate_k_default: int) -> None:
        self.search_service = search_service
        self.candidate_k_default = candidate_k_default

    def load_cases(self, path: Path) -> list[EvaluationCase]:
        try:
            raw_cases = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Evaluation cases file not found: {path}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in evaluation cases file {path}: {exc}") from exc
        return [EvaluationCase(**item) for item in raw_cases]

    def run_cases(self, cases: list[EvaluationCase], rerank_enabled: bool = True) -> EvaluationReport:
        results: list[EvaluationCaseResult] = []
        total_hard_negative_hits = 0
        total_slots = 0
        for case in cases:
            response = self.search_service.search(
                query=case.query,
                top_n=case.top_k,
                candidate_k=max(self.candidate_k_default, case.top_k * 3),
                mode=case.mode,
                preferred_tone=case.preferred_tone,
                rerank_enabled=rerank_enabled,
            )
            top_results = [result.image_id for result in response.results[: case.top_k]]
            positives = set(case.positives)
            precision_hits = sum(1 for image_id in top_results if image_id in positives)
            first_positive_rank = next((index + 1 for index, image_id in enumerate(top_results) if image_id in positives), None)
            mrr = (1 / first_positive_rank) if first_positive_rank else 0.0
            hard_negative_hits = [image_id for image_id in top_results if image_id in set(case.hard_negatives)]
            total_hard_negative_hits += len(hard_negative_hits)
            total_slots += case.top_k
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
            hard_negative_hit_rate=(total_hard_negative_hits / total_slots) if total_slots else 0.0,
            cases=results,
        )

    def tune_scoring_profile(self, cases: list[EvaluationCase], passes: int = 2) -> EvaluationTuningReport:
        baseline_profile = self.search_service.scoring_profile.model_copy(deep=True)
        baseline_report = self._run_cases_with_profile(cases, baseline_profile, rerank_enabled=False)
        best_profile = baseline_profile.model_copy(deep=True)
        best_report = baseline_report
        best_objective = self._objective(best_report)

        for _ in range(passes):
            improved = False
            for mode in (SearchMode.REPLY, SearchMode.SEMANTIC):
                if not any(case.mode == mode for case in cases):
                    continue
                for field_name in self._tunable_fields(mode):
                    current_section = getattr(best_profile, mode.value)
                    current_value = getattr(current_section, field_name)
                    for candidate_value in self._candidate_values(field_name, current_value):
                        if candidate_value == current_value:
                            continue
                        candidate_profile = best_profile.model_copy(deep=True)
                        setattr(getattr(candidate_profile, mode.value), field_name, candidate_value)
                        candidate_report = self._run_cases_with_profile(
                            cases,
                            candidate_profile,
                            rerank_enabled=False,
                        )
                        candidate_objective = self._objective(candidate_report)
                        if candidate_objective <= best_objective:
                            continue
                        best_profile = candidate_profile
                        best_report = candidate_report
                        best_objective = candidate_objective
                        improved = True
            if not improved:
                break

        return EvaluationTuningReport(
            baseline_objective=self._objective(baseline_report),
            best_objective=best_objective,
            baseline_report=baseline_report,
            tuned_report=best_report,
            best_profile=best_profile,
            passes=passes,
        )

    def _run_cases_with_profile(
        self,
        cases: list[EvaluationCase],
        profile: SearchScoringProfile,
        rerank_enabled: bool,
    ) -> EvaluationReport:
        original_profile = self.search_service.scoring_profile.model_copy(deep=True)
        try:
            self.search_service.set_scoring_profile(profile)
            return self.run_cases(cases, rerank_enabled=rerank_enabled)
        finally:
            self.search_service.set_scoring_profile(original_profile)

    def _objective(self, report: EvaluationReport) -> float:
        return report.precision_at_k + report.mrr - report.hard_negative_hit_rate

    def _tunable_fields(self, mode: SearchMode) -> list[str]:
        valid_fields = set(DeterministicModeScoringProfile.model_fields)
        fields = self._tunable_fields_unchecked(mode)
        invalid = set(fields) - valid_fields
        if invalid:
            raise ValueError(f"Tunable field names not in DeterministicModeScoringProfile: {invalid}")
        return fields

    def _tunable_fields_unchecked(self, mode: SearchMode) -> list[str]:
        if mode == SearchMode.REPLY:
            return [
                "semantic_vector",
                "reply_vector",
                "keyword_route",
                "template_route",
                "ocr_overlap",
                "emotion_overlap",
                "intent_match",
                "preferred_tone_match",
                "penalty_multiplier",
                "non_ocr_score_cap",
            ]
        return [
            "semantic_vector",
            "reply_vector",
            "keyword_route",
            "template_route",
            "semantic_text_overlap",
            "emotion_overlap",
            "intent_match",
            "preferred_tone_match",
            "penalty_multiplier",
        ]

    def _candidate_values(self, field_name: str, current_value: float | None) -> list[float]:
        if field_name == "non_ocr_score_cap":
            baseline = float(current_value if current_value is not None else 0.48)
            return sorted({0.32, 0.4, round(baseline, 3), 0.56, 0.64})
        baseline = float(current_value if current_value is not None else 0.0)
        factors = (0.5, 0.75, 1.0, 1.25, 1.5)
        candidates = {0.0}
        candidates.update(round(baseline * factor, 3) for factor in factors)
        if baseline == 0.0:
            candidates.update({0.03, 0.05, 0.08, 0.12})
        return sorted(candidate for candidate in candidates if candidate >= 0.0)
