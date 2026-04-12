"""BigCodeBench code generation task.

BigCodeBench evaluates practical programming capabilities with complex instructions
and diverse function calls, going beyond HumanEval-style simple function completion.

Paper: https://arxiv.org/pdf/2406.15877
Dataset: bigcode/bigcodebench
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.common import SandboxEnv, Task, register, register_variant

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment


@dataclass(frozen=True, slots=True)
class BigCodeBenchScorer(CodeExecutionScorer):
    """Scorer for BigCodeBench that invokes unittest after test class definition.

    BigCodeBench test code defines unittest.TestCase classes but does not
    invoke the test runner. Without unittest.main(), the test classes are
    defined but never executed, causing all submissions to appear to pass.
    """

    timeout: float = 60.0

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
        return 1.0 if result.success else 0.0


@register("bigcodebench")
class BigCodeBench(Task):
    """BigCodeBench code completion task (full subset, complete prompt variant)."""

    data_source = DataSource(path="bigcode/bigcodebench")
    # BigCodeBench tests import a wide range of third-party packages (pandas,
    # numpy, sklearn, etc.).  The sandbox must have them pre-installed so that
    # code execution does not fail with ImportError.  The list below mirrors the
    # official bigcodebench-evaluate Docker image's requirements-eval.txt.
    sandbox_env = SandboxEnv(
        "bigcodebench",
        (
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "scipy",
            "seaborn",
            "statsmodels",
            "sympy",
            "Pillow",
            "opencv-python-headless",
            "requests",
            "requests-mock",
            "beautifulsoup4",
            "lxml",
            "flask",
            "Flask-Login",
            "Flask-Mail",
            "flask-restful",
            "Flask-WTF",
            "WTForms",
            "django",
            "werkzeug",
            "openpyxl",
            "xlrd",
            "xlwt",
            "python-docx",
            "PyYAML",
            "xmltodict",
            "chardet",
            "cryptography",
            "pycryptodome",
            "rsa",
            "pytz",
            "python-dateutil",
            "faker",
            "nltk",
            "gensim",
            "textblob",
            "wordninja",
            "wordcloud",
            "python-Levenshtein",
            "geopandas",
            "shapely",
            "folium",
            "geopy",
            "holidays",
            "natsort",
            "networkx",
            "prettytable",
            "texttable",
            "librosa",
            "soundfile",
            "psutil",
            "pyquery",
            "sendgrid",
            "python_http_client",
            "blake3",
            "wikipedia",
            "regex",
            "pyfakefs",
            "scikit-image",
            "mechanize",
            "numba",
            "dnspython",
        ),
        # Pre-download NLTK data used by BCB tests (stopwords, punkt, etc.)
        dockerfile_extra=(
            "RUN /root/python/bin/python -c \"import nltk; nltk.download('all', quiet=True)\"",
        ),
    )
    sampling_params = SamplingParams(
        max_tokens=1280,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=5,
        # Stop sequences match oe-eval-internal's bigcodebench:3shot::olmo3:v2
        # exactly. We do NOT include "\n```" — the old suite omits it and uses
        # tree-sitter sanitize() for cleanup instead. Our _extract_answers
        # strips trailing markdown fences as a safety net.
        stop_sequences=(
            "<|endoftext|>",
            "<|endofmask|>",
            "</s>",
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
            "\ndef ",
            "\nclass ",
            "\nimport ",
            "\nfrom ",
            "\nassert ",
        ),
    )
    # BigCodeBench uses "v0.1.2" as split name (mapped as train on HF)
    fewshot_split: str = "v0.1.2"

    # Instruction prefix from the original BigCodeBench repo's make_raw_chat_prompt()
    INSTRUCTION_PREFIX = (
        "Please provide a self-contained Python script that solves the"
        " following problem in a markdown code block:"
    )

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
        prompt = self.INSTRUCTION_PREFIX + "\n```\n" + doc["complete_prompt"].strip() + "\n"
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
        """Sample fewshot examples from the training split.

        Matches oe-eval-internal behavior: sample exactly num_fewshot examples
        (no over-sampling for dedup) since has_training_docs() is True.
        """
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
        k = min(self.config.num_fewshot, len(all_instances))
        return rng.sample(all_instances, k)

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            # Match oe-eval-internal: no per-instance dedup when fewshot comes
            # from training split (fewshot_examples samples exactly k).
            fewshot = self.get_fewshot()[: self.config.num_fewshot]
            return self.config.formatter.format(instance, fewshot)

        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_code(output.text)

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract code from model outputs, prepending the complete_prompt.

        Matches oe-eval-internal's approach: complete_prompt + continuation.
        Stop sequences should truncate output during generation, but as a
        safety net we also truncate at the first occurrence of any stop
        sequence that may have been missed (e.g. due to provider limits).
        """
        # Stop sequences that indicate end of code (skip special tokens)
        _CODE_STOPS = (
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
            "\ndef ",
            "\nclass ",
            "\nimport ",
            "\nfrom ",
            "\nassert ",
            "\n```",
        )
        for response in responses:
            for output in response.outputs:
                text = output.text
                if not text or not text.strip():
                    output.extracted_answer = None
                    continue
                # Truncate at first stop sequence (safety net)
                for stop in _CODE_STOPS:
                    idx = text.find(stop)
                    if idx != -1:
                        text = text[:idx]
                # Strip trailing markdown fence if still present
                text = re.sub(r"\n?```\s*$", "", text)
                output.extracted_answer = (
                    response.instance.metadata["answer_prefix"] + text
                )


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
