from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.minerva_math import MinervaMathTask

_PASS_AT_32_METRICS = {
    "acc": AccuracyMetric(scorer=MinervaMathScorer),
    "k1": PassAtKMetric(k=1, scorer=MinervaMathScorer),
    "k4": PassAtKMetric(k=4, scorer=MinervaMathScorer),
    "k8": PassAtKMetric(k=8, scorer=MinervaMathScorer),
    "k16": PassAtKMetric(k=16, scorer=MinervaMathScorer),
    "k32": PassAtKMetric(k=32, scorer=MinervaMathScorer),
}

_PASS_AT_32_SAMPLING = SamplingParams(
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    num_samples=32,
)
_PASS_AT_32_16K_SAMPLING = replace(_PASS_AT_32_SAMPLING, max_tokens=16384)

_COT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."

_PASS_AT_32_FORMATTER = ChatFormatter(
    user_template="{question}" + _COT_SUFFIX,
)

_RLZERO_FORMATTER = ChatFormatter(user_template="{question}")

_RLZERO_SAMPLING = SamplingParams(
    max_tokens=16384,
    temperature=1.0,
    top_p=0.95,
    num_samples=32,
)


@dataclass(frozen=True, slots=True)
class HMMTCompetition:
    dataset_path: str
    year: int
    season: str
    date: str

    @property
    def competition(self) -> str:
        return f"{self.season}_{self.year}"


_HMMT_COMPETITIONS = (
    HMMTCompetition(
        dataset_path="MathArena/hmmt_feb_2025",
        year=2025,
        season="feb",
        date="2025-02-15",
    ),
    HMMTCompetition(
        dataset_path="MathArena/hmmt_nov_2025",
        year=2025,
        season="nov",
        date="2025-11-08",
    ),
    HMMTCompetition(
        dataset_path="MathArena/hmmt_feb_2026",
        year=2026,
        season="feb",
        date="2026-02-14",
    ),
)

_HMMT_COMPETITIONS_BY_PATH = {
    competition.dataset_path: competition for competition in _HMMT_COMPETITIONS
}


def _normalize_problem_types(problem_types: Any) -> list[str] | None:
    if problem_types is None:
        return None

    if isinstance(problem_types, str):
        problem_types = [problem_types]

    normalized = [str(problem_type).strip() for problem_type in problem_types]
    filtered = [problem_type for problem_type in normalized if problem_type]
    return filtered or None


class HMMTTask(MinervaMathTask):
    split = Split.TRAIN  # MathArena HMMT datasets only expose train
    formatter = ChatFormatter(user_template="{question}")
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 0
    sampling_params = SamplingParams(max_tokens=32768, temperature=0.0)

    years: tuple[int, ...] | None = None
    seasons: tuple[str, ...] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def _selected_competitions(self) -> tuple[HMMTCompetition, ...]:
        competitions = []
        for competition in _HMMT_COMPETITIONS:
            if self.years is not None and competition.year not in self.years:
                continue
            if self.seasons is not None and competition.season not in self.seasons:
                continue
            competitions.append(competition)
        return tuple(competitions)

    def _resolve_competition(self, doc: dict[str, Any]) -> HMMTCompetition:
        competition = doc.get("_hmmt_competition")
        if competition is not None:
            return competition

        dataset_path = doc.get("dataset_path")
        if dataset_path in _HMMT_COMPETITIONS_BY_PATH:
            return _HMMT_COMPETITIONS_BY_PATH[dataset_path]

        config_source = self.config.data_source
        if (
            isinstance(config_source, DataSource)
            and config_source.path in _HMMT_COMPETITIONS_BY_PATH
        ):
            return _HMMT_COMPETITIONS_BY_PATH[config_source.path]

        selected = self._selected_competitions()
        if len(selected) == 1:
            return selected[0]

        year = doc.get("year")
        season = doc.get("season")
        for candidate in selected:
            if candidate.year == year and candidate.season == season:
                return candidate

        raise ValueError("Could not determine HMMT competition for document")

    def _build_identifier(
        self,
        *,
        problem_idx: Any,
        fallback_index: int,
        competition: HMMTCompetition,
    ) -> Any:
        base_identifier = problem_idx if problem_idx is not None else fallback_index
        if len(self._selected_competitions()) == 1:
            return base_identifier
        return f"{competition.competition}:{base_identifier}"

    def _load_instances_cached(self, split: str | None = None) -> Iterator[Instance]:
        if self._instances_cache is not None:
            yield from self._instances_cache
            return

        current_split = split or self.config.split.value
        self._instances_cache = []
        loader = DataLoader()

        for competition in self._selected_competitions():
            source = DataSource(path=competition.dataset_path, split=current_split)
            for index, doc in enumerate(loader.load(source)):
                doc_with_metadata = dict(doc)
                doc_with_metadata["_hmmt_competition"] = competition
                instance = self.process_doc(doc_with_metadata, index)
                if instance is not None:
                    self._instances_cache.append(instance)

        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        competition = self._resolve_competition(doc)

        if self.years is not None and competition.year not in self.years:
            return None
        if self.seasons is not None and competition.season not in self.seasons:
            return None

        problem_idx = doc.get("problem_idx")
        gold_answer = str(doc["answer"]).strip()

        return Instance(
            question=doc["problem"],
            gold_answer=gold_answer,
            metadata={
                "id": self._build_identifier(
                    problem_idx=problem_idx,
                    fallback_index=index,
                    competition=competition,
                ),
                "year": competition.year,
                "season": competition.season,
                "competition": competition.competition,
                "date": competition.date,
                "problem_number": problem_idx,
                "problem_types": _normalize_problem_types(doc.get("problem_type")),
                "all_gold_answers": [gold_answer],
            },
        )


@register("hmmt_feb_2025")
class HMMTFeb2025Task(HMMTTask):
    data_source = DataSource(path="MathArena/hmmt_feb_2025")
    years = (2025,)
    seasons = ("feb",)


@register("hmmt_nov_2025")
class HMMTNov2025Task(HMMTTask):
    data_source = DataSource(path="MathArena/hmmt_nov_2025")
    years = (2025,)
    seasons = ("nov",)


@register("hmmt_feb_2026")
class HMMTFeb2026Task(HMMTTask):
    data_source = DataSource(path="MathArena/hmmt_feb_2026")
    years = (2026,)
    seasons = ("feb",)


for _task_name in ("hmmt_feb_2025", "hmmt_nov_2025", "hmmt_feb_2026"):
    register_variant(
        _task_name,
        "pass_at_32",
        formatter=_PASS_AT_32_FORMATTER,
        metrics=tuple(_PASS_AT_32_METRICS.values()),
        primary_metric=_PASS_AT_32_METRICS["k1"],
        sampling_params=_PASS_AT_32_SAMPLING,
    )

    register_variant(
        _task_name,
        "pass_at_32_rlzero",
        formatter=_RLZERO_FORMATTER,
        metrics=tuple(_PASS_AT_32_METRICS.values()),
        primary_metric=_PASS_AT_32_METRICS["k1"],
        sampling_params=_RLZERO_SAMPLING,
    )

for _task_name in ("hmmt_nov_2025", "hmmt_feb_2026"):
    register_variant(
        _task_name,
        "16k",
        sampling_params=_PASS_AT_32_16K_SAMPLING,
    )
