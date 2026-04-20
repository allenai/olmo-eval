"""BigCodeBench code generation task.

BigCodeBench evaluates practical programming capabilities with complex instructions
and diverse function calls, going beyond HumanEval-style simple function completion.

Paper: https://arxiv.org/pdf/2406.15877
Dataset: bigcode/bigcodebench
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import BIGCODEBENCH_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code_before_fence
from olmo_eval.evals.tasks.common import SandboxEnv, Task, register, register_variant

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BigCodeBenchScorer(CodeExecutionScorer):
    """Scorer for BigCodeBench that invokes unittest after test class definition.

    BigCodeBench test code defines unittest.TestCase classes but does not
    invoke the test runner. Without unittest.main(), the test classes are
    defined but never executed, causing all submissions to appear to pass.
    """

    timeout: float = 20.0

    async def ascore(
        self,
        instance: Instance,
        output: LMOutput,
        execution_env: ExecutionEnvironment,
    ) -> float:
        if output.extracted_answer is None:
            return 0.0

        test_code = instance.metadata.get("test", "")
        if not test_code:
            return 0.0

        full_code = (
            f"{output.extracted_answer}\n\n{test_code}\n\nimport unittest\nunittest.main()\n"
        )

        result = await execution_env.execute_code(
            full_code,
            language=self.language,
            timeout=self.timeout,
        )
        if not result.success and result.error:
            instance_id = instance.metadata.get("id", "?")
            logger.warning(f"Code execution failed [{instance_id}]: {result.error}")
        return 1.0 if result.success else 0.0


@register("bigcodebench")
class BigCodeBench(Task):
    """BigCodeBench code completion task (full subset, complete prompt variant)."""

    data_source = DataSource(path="bigcode/bigcodebench")
    # Python deps from the official BigCodeBench requirements-eval.txt:
    # https://github.com/bigcode-project/bigcodebench/blob/main/Requirements/requirements-eval.txt
    sandbox_env = SandboxEnv(
        "bigcodebench",
        (
            "beautifulsoup4",
            "blake3",
            "chardet",
            "cryptography",
            "datetime",
            "django",
            "dnspython",
            "docxtpl",
            "faker",
            "flask",
            "flask-login",
            "flask-mail",
            "flask-restful",
            "flask-wtf",
            "folium",
            "gensim",
            "geopandas",
            "geopy",
            "holidays",
            "keras",
            "Levenshtein",
            "librosa",
            "lxml",
            "matplotlib",
            "mechanize",
            "natsort",
            "networkx",
            "numba",
            "nltk",
            "numpy",
            "opencv-python-headless",
            "openpyxl",
            "pandas",
            "pillow",
            "prettytable",
            "psutil",
            "pycryptodome",
            "pyfakefs",
            "pyquery",
            "pytest",
            "pytesseract",
            "python-dateutil",
            "python-docx",
            "python-http-client",
            "pytz",
            "pyyaml",
            "requests",
            "requests-mock",
            "rsa",
            "scikit-image",
            "scikit-learn",
            "scipy",
            "seaborn",
            "selenium",
            "sendgrid",
            "shapely",
            "soundfile",
            "statsmodels",
            "sympy",
            "tensorflow",
            "textblob",
            "texttable",
            "werkzeug",
            "wikipedia",
            "wordcloud",
            "wordninja",
            "wtforms",
            "xlrd",
            "xlwt",
            "xmltodict",
        ),
    )
    sampling_params = SamplingParams(
        max_tokens=1280,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=5,
        stop_sequences=BIGCODEBENCH_STOP_SEQUENCES,
    )
    # BigCodeBench uses "v0.1.2" as split name (mapped as train on HF)
    fewshot_split: str = "v0.1.2"

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("v0.1.2")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        prompt = (
            "Please provide a self-contained Python script that solves the "
            "following problem in a markdown code block:\n```\n"
            + doc["complete_prompt"].strip()
            + "\n"
        )
        gold = doc["canonical_solution"] + "\n```"
        test_code = doc.get("test", "")

        return Instance(
            question=prompt,
            gold_answer=gold,
            metadata={
                "id": doc.get("task_id", str(index)),
                "entry_point": doc.get("entry_point", ""),
                "answer_prefix": doc["complete_prompt"],
                "test": test_code,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        """Sample one extra for per-instance dedup (fewshot == eval split)."""
        import random

        if self.config.num_fewshot == 0:
            return []

        loader = DataLoader()
        source = self._get_source_for_split(self.fewshot_split)
        all_instances = [
            inst for doc in loader.load(source) if (inst := self.process_doc(doc)) is not None
        ]

        if not all_instances:
            return []

        rng = random.Random(self.config.fewshot_seed)
        k = min(self.config.num_fewshot + 1, len(all_instances))
        return rng.sample(all_instances, k)

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            fewshot = self.get_fewshot()
            instance_id = instance.metadata.get("id")
            if instance_id is not None:
                filtered = [ex for ex in fewshot if ex.metadata.get("id") != instance_id]
            else:
                filtered = list(fewshot)
            filtered = filtered[: self.config.num_fewshot]
            return self.config.formatter.format(instance, filtered)

        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_code_before_fence(output.text)

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        from olmo_eval.evals.extract import sanitize_code

        for response in responses:
            entry_point = response.instance.metadata.get("entry_point", "")
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    full_code = response.instance.metadata["answer_prefix"] + code
                    if entry_point:
                        full_code = sanitize_code(full_code, entrypoint=entry_point)
                    output.extracted_answer = full_code
                else:
                    output.extracted_answer = None


register_variant(
    "bigcodebench",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(answer_prefix=""),
)

register_variant(
    "bigcodebench",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricByteAvg(),),
)

register_variant(
    "bigcodebench",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=BigCodeBenchScorer),),
)

register_variant(
    "bigcodebench",
    "olmo3base",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(answer_prefix=""),
    metrics=(PassAtKMetric(k=1, scorer=BigCodeBenchScorer),),
)
