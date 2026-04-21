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
    """Scorer for BigCodeBench matching the original BCB execution harness.

    Replicates the old oe-eval-internal execution pattern:
    - Calibration: prepends code_prompt + pass stub before solution
    - Module-based execution: runs code in a __test__ module via exec()
    - TestLoader: explicitly loads TestCases class and runs via suite.run()
    - Pass condition: no failures AND no errors in test_result
    - Environment: sets TZ=UTC, OMP_NUM_THREADS=1, TF_CPP_MIN_LOG_LEVEL=3
    """

    timeout: float = 3.0

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

        solution = output.extracted_answer
        code_prompt = instance.metadata.get("code_prompt", "")

        # Calibration: prepend code_prompt + pass stub (matches old BCB Lambda)
        if code_prompt:
            solution = code_prompt + "\n    pass\n" + solution

        # Build a script that replicates the old unsafe_execute pattern:
        # - exec code+test in a __test__ module
        # - use TestLoader to load TestCases class
        # - run suite and check failures+errors
        full_code = _build_bcb_execution_script(solution, test_code)

        result = await execution_env.execute_code(
            full_code,
            language=self.language,
            timeout=self.timeout,
        )
        if not result.success and result.error:
            instance_id = instance.metadata.get("id", "?")
            logger.warning(f"Code execution failed [{instance_id}]: {result.error}")
        return 1.0 if result.success else 0.0


def _build_bcb_execution_script(solution: str, test_code: str) -> str:
    """Build a Python script replicating the old BCB execution harness.

    The old system (bcb_execution/execution.py unsafe_execute):
    1. Sets environment variables (TZ, OMP_NUM_THREADS, TF_CPP_MIN_LOG_LEVEL)
    2. Creates a __test__ module with builtins, sys, os
    3. exec(code + test) in the module
    4. Loads TestCases via unittest.TestLoader
    5. Runs suite via suite.run(test_result)
    6. Passes only if no failures AND no errors
    """
    return (
        "import types, sys, os, builtins, unittest, io, contextlib\n"
        "os.environ['TZ'] = 'UTC'\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
        "os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'\n"
        "module_name = '__test__'\n"
        "new_module = types.ModuleType(module_name)\n"
        "new_module.__dict__.update({\n"
        "    '__builtins__': builtins,\n"
        "    '__file__': f'{module_name}.py',\n"
        "    '__package__': None,\n"
        "    '__doc__': None,\n"
        "    'sys': sys,\n"
        "    'os': os,\n"
        "    'environ': os.environ,\n"
        "})\n"
        f"_code = {solution!r}\n"
        f"_test = {test_code!r}\n"
        "full_code = _code + '\\n' + _test\n"
        "stream = io.StringIO()\n"
        "with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):\n"
        "    exec(compile(full_code, f'{module_name}.py', 'exec'), new_module.__dict__)\n"
        "sys.modules[module_name] = new_module\n"
        "TestCases = getattr(new_module, 'TestCases')\n"
        "loader = unittest.TestLoader()\n"
        "suite = loader.loadTestsFromTestCase(TestCases)\n"
        "test_result = unittest.TestResult()\n"
        "with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):\n"
        "    suite.run(test_result)\n"
        "if test_result.failures or test_result.errors:\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )


@register("bigcodebench")
class BigCodeBench(Task):
    """BigCodeBench code completion task (full subset, complete prompt variant)."""

    data_source = DataSource(path="bigcode/bigcodebench")
    # Python deps from the official BigCodeBench requirements-eval.txt:
    # https://github.com/bigcode-project/bigcodebench/blob/main/Requirements/requirements-eval.txt
    sandbox_env = SandboxEnv(
        "bigcodebench",
        (
            "beautifulsoup4==4.8.2",
            "blake3==0.4.1",
            "chardet==5.2.0",
            "cryptography==38.0.0",
            "datetime==5.5",
            "django==4.2.7",
            "dnspython==2.6.1",
            "docxtpl==0.11.5",
            "faker==20.1.0",
            "flask==3.0.3",
            "flask-login==0.6.3",
            "flask-mail==0.9.1",
            "flask-restful==0.3.10",
            "flask-wtf==1.2.1",
            "folium==0.16.0",
            "gensim==4.3.2",
            "geopandas==0.13.2",
            "geopy==2.4.1",
            "holidays==0.29",
            "keras==2.11.0",
            "Levenshtein==0.25.0",
            "librosa==0.10.1",
            "lxml==4.9.3",
            "matplotlib==3.7.0",
            "mechanize==0.4.9",
            "natsort==7.1.1",
            "networkx==2.6.3",
            "numba==0.55.0",
            "nltk==3.8",
            "numpy==1.21.2",
            "opencv-python-headless==4.9.0.80",
            "openpyxl==3.1.2",
            "pandas==2.0.3",
            "pillow==10.3.0",
            "prettytable==3.10.0",
            "psutil==5.9.5",
            "pycryptodome==3.14.1",
            "pyfakefs==5.4.1",
            "pyquery==1.4.3",
            "pytest==8.2.0",
            "pytesseract==0.3.10",
            "python-dateutil==2.9.0",
            "python-docx==1.1.0",
            "python-http-client==3.3.7",
            "python-Levenshtein-wheels",
            "pytz==2023.3.post1",
            "pyyaml==6.0.1",
            "requests==2.31.0",
            "requests-mock==1.11.0",
            "rsa==4.9",
            "scikit-image==0.18.0",
            "scikit-learn==1.3.1",
            "scipy==1.7.2",
            "seaborn==0.13.2",
            "selenium==4.15",
            "sendgrid==6.11.0",
            "shapely==2.0.4",
            "soundfile==0.12.1",
            "statsmodels==0.14.0",
            "sympy==1.12",
            "tensorflow==2.11.0",
            "textblob==0.18.0",
            "texttable==1.7.0",
            "werkzeug==3.0.1",
            "wikipedia==1.4.0",
            "wordcloud==1.9.3",
            "wordninja==2.0.0",
            "wtforms==3.1.2",
            "xlrd==2.0.1",
            "xlwt==1.3.0",
            "xmltodict==0.13.0",
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
                "code_prompt": doc.get("code_prompt", ""),
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
                # Use raw text directly (no extract_code_before_fence) to match
                # old oe-eval-internal behavior, which prepends complete_prompt
                # to the raw continuation and sanitizes.
                code = output.text
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
