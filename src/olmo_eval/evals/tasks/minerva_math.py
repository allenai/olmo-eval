from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    AccuracyMetric,
    BitsPerByteScorer,
    BPBMetric,
    CompletionFormatter,
    ExactMatchFlexScorer,
    ExactMatchScorer,
    Instance,
    LMOutput,
    LMRequest,
    MathVerifyScorer,
    PPLFormatter,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant


MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MinervaMathTask(Task):
    fewshot_split: str = "train"

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                instance = self.process_doc(doc)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        return self.config.get_data_source(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        solution_text = doc.get("solution", "")
        extracted_answers = MathExtractor.extract_answer(solution_text)
        
        # Use first as primary gold_answer for backward compatibility
        primary_answer = extracted_answers[0] if extracted_answers else None
        all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=primary_answer,
            metadata={
                "level": doc.get("level"),
                "type": doc.get("type"),
                "solution_text": solution_text,
                "all_gold_answers": all_gold_answers,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str | None:
        answers = MathExtractor.extract_answer(output.text)
        # Store all extracted answers in metadata for flexible scorers
        output.metadata["all_extracted_answers"] = answers if answers else []
        return answers[0] if answers else None


class Math500Task(MinervaMathTask):
    def _get_source_for_split(self, split: str) -> DataSource:
        # MATH-500 only has test split; use full MATH for train/dev
        if split != "test":
            return DataSource(
                path="EleutherAI/hendrycks_math",
                subset="algebra",  # Use algebra subset for few-shot
                split=split,
            )
        return self.config.get_data_source(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # MATH-500 provides the answer directly
        gold_answer = doc.get("answer")
        all_gold_answers: list[str] = []

        if gold_answer is not None:
            all_gold_answers = [gold_answer]
        else:
            # Fall back to extraction from solution
            solution_text = doc.get("solution", "")
            extracted_answers = MathExtractor.extract_answer(solution_text)
            gold_answer = extracted_answers[0] if extracted_answers else None
            all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=gold_answer,
            metadata={
                "level": doc.get("level"),
                "type": doc.get("type", doc.get("subject")),
                "solution_text": doc.get("solution", ""),
                "all_gold_answers": all_gold_answers,
            },
        )


def _minerva_math_config(subset: str | None = None) -> TaskConfig:
    data_source = DataSource(
        path="EleutherAI/hendrycks_math",
        subset=subset,
    )

    return TaskConfig(
        name=f"minerva_math_{subset}" if subset else "minerva_math",
        data_source=data_source,
        formatter=CompletionFormatter(
            template="Problem: {question}\nSolution:",
            fewshot_answer_key="solution_text",  # Use full solution for few-shot
        ),
        scorers=(ExactMatchScorer(case_sensitive=False, remove_whitespace=True),),
        metrics=(AccuracyMetric(),),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.7,
        ),
    )


def _math500_config() -> TaskConfig:
    return TaskConfig(
        name="math500",
        data_source=DataSource(path="HuggingFaceH4/MATH-500"),
        formatter=CompletionFormatter(
            template="Problem: {question}\nSolution:",
            fewshot_answer_key="solution_text",  # Use full solution for few-shot
        ),
        scorers=(ExactMatchScorer(case_sensitive=False, remove_whitespace=True),),
        metrics=(AccuracyMetric(),),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.7,
        ),
    )


@register("minerva_math", lambda: _minerva_math_config(None))
class MinervaMath(MinervaMathTask):
    pass


for _subset in MATH_SUBSETS:

    def _make_config(s: str = _subset) -> TaskConfig:
        return _minerva_math_config(s)

    _task_name = f"minerva_math_{_subset}"
    _task_class = type(
        f"MinervaMath_{_subset.title().replace('_', '')}",
        (MinervaMathTask,),
        {},
    )
    register(_task_name, _make_config)(_task_class)


@register("math500", _math500_config)
class Math500(Math500Task):
    pass


# Create variants
register_variant(
    "minerva_math",
    "4shot",
    num_fewshot=4,
)

register_variant(
    "math500",
    "4shot",
    num_fewshot=4,
)

for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "4shot",
        num_fewshot=4,
    )

register_variant(
    "minerva_math",
    "bpb",
    formatter=PPLFormatter(),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

register_variant(
    "math500",
    "bpb",
    formatter=PPLFormatter(),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "bpb",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )

# Math-Verify variants (symbolic verification)
# NOTE: Must also update metrics to reference the correct scorer type
register_variant(
    "minerva_math",
    "math_verify",
    scorers=(MathVerifyScorer(),),
    metrics=(AccuracyMetric(scorer=MathVerifyScorer),),
)

register_variant(
    "math500",
    "math_verify",
    scorers=(MathVerifyScorer(),),
    metrics=(AccuracyMetric(scorer=MathVerifyScorer),),
)

for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "math_verify",
        scorers=(MathVerifyScorer(),),
        metrics=(AccuracyMetric(scorer=MathVerifyScorer),),
    )

# Combined variants: 4shot with math_verify
register_variant(
    "minerva_math",
    "4shot_math_verify",
    num_fewshot=4,
    scorers=(MathVerifyScorer(),),
    metrics=(AccuracyMetric(scorer=MathVerifyScorer),),
)

register_variant(
    "math500",
    "4shot_math_verify",
    num_fewshot=4,
    scorers=(MathVerifyScorer(),),
    metrics=(AccuracyMetric(scorer=MathVerifyScorer),),
)

for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "4shot_math_verify",
        num_fewshot=4,
        scorers=(MathVerifyScorer(),),
        metrics=(AccuracyMetric(scorer=MathVerifyScorer),),
    )

# Flexible exact match variants (match ANY extracted answer against ANY gold answer)
# NOTE: Must also update metrics to reference the correct scorer type
register_variant(
    "minerva_math",
    "exact_match_flex",
    scorers=(ExactMatchFlexScorer(case_sensitive=False, remove_whitespace=True),),
    metrics=(AccuracyMetric(scorer=ExactMatchFlexScorer),),
)

register_variant(
    "math500",
    "exact_match_flex",
    scorers=(ExactMatchFlexScorer(case_sensitive=False, remove_whitespace=True),),
    metrics=(AccuracyMetric(scorer=ExactMatchFlexScorer),),
)

for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "exact_match_flex",
        scorers=(ExactMatchFlexScorer(case_sensitive=False, remove_whitespace=True),),
        metrics=(AccuracyMetric(scorer=ExactMatchFlexScorer),),
    )

# Combined variants: 4shot with exact_match_flex
register_variant(
    "minerva_math",
    "4shot_exact_match_flex",
    num_fewshot=4,
    scorers=(ExactMatchFlexScorer(case_sensitive=False, remove_whitespace=True),),
    metrics=(AccuracyMetric(scorer=ExactMatchFlexScorer),),
)

register_variant(
    "math500",
    "4shot_exact_match_flex",
    num_fewshot=4,
    scorers=(ExactMatchFlexScorer(case_sensitive=False, remove_whitespace=True),),
    metrics=(AccuracyMetric(scorer=ExactMatchFlexScorer),),
)

for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "4shot_exact_match_flex",
        num_fewshot=4,
        scorers=(ExactMatchFlexScorer(case_sensitive=False, remove_whitespace=True),),
        metrics=(AccuracyMetric(scorer=ExactMatchFlexScorer),),
    )
