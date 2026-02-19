from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import AccuracyMetric, BPBMetric, PassAtKMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.common import Task, register, register_variant

MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Fixed 4 few-shot examples (from oe-eval)
# fmt: off
MINERVA_MATH_FIXED_FEWSHOT = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",  # noqa: E501
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",  # noqa: E501
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",  # noqa: E501
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",  # noqa: E501
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",  # noqa: E501
    },
    {
        "problem": "If the system of equations\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}\nhas a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",  # noqa: E501
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",  # noqa: E501
    },
]
# fmt: on


class MinervaMathTask(Task):
    fewshot_split: str = "train"
    formatter = CompletionFormatter(
        template="Problem:\n{question}\n\nSolution: ",
        fewshot_answer_key="solution_text",
    )
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 4
    sampling_params = SamplingParams(
        max_tokens=1024, temperature=0, stop_sequences=("Problem:", "\n\n")
    )
    dependencies = ["lark>=1.0"]

    def _build_fewshot(self) -> list[Instance]:
        """Use fixed 4 examples when fewshot_source is 'minerva_math_fixed'."""
        if getattr(self.config, "fewshot_source", None) == "minerva_math_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        """Build 4 fixed few-shot instances from MINERVA_MATH_FIXED_FEWSHOT."""
        instances = []
        for doc in MINERVA_MATH_FIXED_FEWSHOT:
            solution_text = doc["solution"]
            extracted = MathExtractor.extract_answer(solution_text)
            primary = extracted[0] if extracted else None
            instances.append(
                Instance(
                    question=doc["problem"],
                    gold_answer=primary,
                    metadata={
                        "solution_text": solution_text,
                        "all_gold_answers": extracted if extracted else [],
                    },
                )
            )
        return instances

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for index, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, index)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        return self.config.get_data_source(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        solution_text = doc["solution"]
        extracted_answers = MathExtractor.extract_answer(solution_text)

        primary_answer = extracted_answers[0] if extracted_answers else None
        all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=primary_answer,
            metadata={
                "id": doc.get("index", index),
                "level": doc.get("level"),
                "type": doc.get("type"),
                "solution_text": solution_text,
                "all_gold_answers": all_gold_answers,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
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
        solution_text = doc["solution"]
        extracted_answers = MathExtractor.extract_answer(solution_text)

        gold_answer = extracted_answers[0] if extracted_answers else None
        all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=gold_answer,
            metadata={
                "id": doc.get("index", index),
                "level": doc.get("level"),
                "type": doc.get("type", doc.get("subject")),
                "solution_text": doc.get("solution"),
                "all_gold_answers": all_gold_answers,
            },
        )


for subset in MATH_SUBSETS:
    task_name = f"minerva_math_{subset}"
    class_name = f"MinervaMath_{subset.title().replace('_', '')}"

    task_cls = type(
        class_name,
        (MinervaMathTask,),
        {
            "__module__": __name__,
            "__qualname__": class_name,
            "data_source": DataSource(
                path="EleutherAI/hendrycks_math",
                subset=subset,
            ),
        },
    )
    globals()[class_name] = task_cls
    register(task_name)(task_cls)


@register("math500")
class Math500(Math500Task):
    data_source = DataSource(path="HuggingFaceH4/MATH-500")


for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"

    register_variant(
        _task_name,
        "olmo3",
        fewshot_source="minerva_math_fixed",
        metrics=(
            AccuracyMetric(scorer=MinervaMathScorer),
            PassAtKMetric(k=1, scorer=MinervaMathScorer),
            PassAtKMetric(k=2, scorer=MinervaMathScorer),
            PassAtKMetric(k=4, scorer=MinervaMathScorer),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            stop_sequences=("Problem:", "\n\n"),
            num_samples=4,
        ),
    )

    register_variant(
        _task_name,
        "olmes_n4_v2",
        fewshot_source="minerva_math_fixed",
        metrics=(
            AccuracyMetric(scorer=MinervaMathScorer),
            PassAtKMetric(k=1, scorer=MinervaMathScorer),
            PassAtKMetric(k=2, scorer=MinervaMathScorer),
            PassAtKMetric(k=4, scorer=MinervaMathScorer),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            stop_sequences=("Problem:", "\n\n"),
            num_samples=4,
        ),
    )

    register_variant(
        _task_name,
        "bpb",
        formatter=PPLFormatter(),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )

register_variant(
    "math500",
    "bpb",
    formatter=PPLFormatter(),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)
