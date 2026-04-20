"""DS-1000 data science code generation task.

DS-1000 contains 1000 data science questions spanning seven Python libraries
with reliable metrics and perturbation-based defenses against memorization.

Paper: https://arxiv.org/abs/2211.11501
Dataset: xlangai/DS-1000
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import DS1000_STOP_SEQUENCES
from olmo_eval.evals.tasks.common import SandboxEnv, Task, register, register_variant

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment


@dataclass(frozen=True, slots=True)
class DS1000Scorer(CodeExecutionScorer):
    """Scorer for DS-1000 that executes the pre-assembled test program directly.

    Unlike other code tasks where metadata["test"] contains static test code
    and extracted_answer contains just the generated code, DS-1000 assembles
    the complete test program (code_context + model continuation) into
    extracted_answer. metadata["test"] is intentionally empty.
    """

    timeout: float = 120.0

    async def ascore(
        self,
        instance: Instance,
        output: LMOutput,
        execution_env: ExecutionEnvironment,
    ) -> float:
        if output.extracted_answer is None:
            return 0.0

        test_code = instance.metadata.get("test", "")
        if test_code:
            full_code = f"{output.extracted_answer}\n\n{test_code}"
        else:
            # DS-1000: extracted_answer IS the complete test program
            # (assembled by _extract_answers from code_context + continuation)
            full_code = output.extracted_answer

        result = await execution_env.execute_code(
            full_code,
            language=self.language,
            timeout=self.timeout,
        )
        return 1.0 if result.success else 0.0


_PY310_URL = (
    "https://github.com/indygreg/python-build-standalone/releases/download/"
    "20240107/cpython-3.10.13+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz"
)

_DS1000_DEPS = (
    "numpy==1.26.4 pandas==1.5.3 matplotlib==3.8.4 scipy==1.12.0"
    " scikit-learn==1.4.0 seaborn==0.13.2 statsmodels==0.14.1"
    " xgboost==2.0.3 gensim==4.3.2"
)


@register("ds1000")
class DS1000(Task):
    """DS-1000 data science code generation task."""

    data_source = DataSource(path="xlangai/DS-1000")
    sandbox_env = SandboxEnv(
        "ds1000",
        (
            "numpy==1.26.4",
            "pandas==1.5.3",
            "matplotlib==3.8.4",
            "scipy==1.12.0",
            "scikit-learn==1.4.0",
            "seaborn==0.13.2",
            "statsmodels==0.14.1",
            "xgboost==2.0.3",
            "gensim==4.3.2",
        ),
        dockerfile_extra=(
            # Install Python 3.10 to match old oe-eval-internal Lambda environment.
            # swe-rex runs on Python 3.11 (/root/python/bin); test code runs on 3.10.
            f"ADD {_PY310_URL} /tmp/python310.tar.gz",
            "RUN mkdir -p /root/python310 && tar xzf /tmp/python310.tar.gz -C /root/python310"
            " && rm /tmp/python310.tar.gz",
            "RUN /root/python310/python/bin/pip install --no-cache-dir uv",
            f"RUN /root/python310/python/bin/python3 -m uv pip install --system --no-cache"
            f" {_DS1000_DEPS}",
            "RUN /root/python310/python/bin/python3 -m uv pip install --system --no-cache"
            " torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu",
            "RUN /root/python310/python/bin/python3 -m uv pip install --system --no-cache"
            " tensorflow-cpu==2.16.1",
        ),
    )
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=5,
        stop_sequences=DS1000_STOP_SEQUENCES,
    )
    fewshot_split: str = "test"

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        prompt = self._format_prompt(doc["prompt"])
        reference_code = doc.get("reference_code", "")
        code_context = doc.get("code_context", "")

        return Instance(
            question=prompt,
            gold_answer=reference_code.rstrip("\n") + "\n```",
            metadata={
                "id": doc.get("metadata", {}).get("problem_id", str(index)),
                "code_context": code_context,
                "test": "",
                "lib": doc.get("metadata", {}).get("library", ""),
            },
        )

    @staticmethod
    def _format_prompt(prompt_text: str) -> str:
        """Process DS-1000 prompt: replace code tags, mask solution lines."""
        text = prompt_text
        text = text.replace("\nBEGIN SOLUTION\n<code>\n", "\n")
        text = text.replace("    ### BEGIN SOLUTION", "")
        text = text.replace("<code>", "```python")
        text = text.replace("</code>", "```")
        text = text + "\n"

        # Mask "put solution in this variable" lines
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "put solution" in line:
                lines[i] = "\n# " + line
                # Remove trailing ``` if present before the masked line
                if i > 0 and lines[i - 1].strip() == "```":
                    lines.pop(i - 1)
        text = "\n".join(lines)
        text = text.rstrip("\n") + "\n"
        return text

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
        return output.text

    # Wrapper that reproduces the old oe-eval-internal Lambda execution
    # pattern: tempdir + swallowed I/O + exec() with empty globals.
    # Delegates to Python 3.10 (/root/python310/python/bin/python3) to match
    # the old Lambda (public.ecr.aws/lambda/python:3.10).
    # The outer python3 (3.11, swe-rex) writes the inner code to a temp file
    # and invokes Python 3.10 to execute it.
    _EXEC_WRAPPER = """\
import subprocess, sys, tempfile, os
_inner = '''
import contextlib, io, os, sys, tempfile

class _WriteOnlyStringIO(io.StringIO):
    def read(self, *a, **k): raise IOError
    def readline(self, *a, **k): raise IOError
    def readlines(self, *a, **k): raise IOError
    def readable(self, *a, **k): return False

class _RedirectStdin(contextlib._RedirectStream):
    _stream = "stdin"

_td = tempfile.mkdtemp()
os.chdir(_td)
_stream = _WriteOnlyStringIO()
with contextlib.redirect_stdout(_stream):
    with contextlib.redirect_stderr(_stream):
        with _RedirectStdin(_stream):
            exec(compile(_TEST_CODE, "<test>", "exec"), {})
'''
_py310 = "/root/python310/python/bin/python3"
if not os.path.exists(_py310):
    _py310 = "python3"  # fallback for local testing
_f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
_f.write(f"_TEST_CODE = {repr(_TEST_CODE)}\\n")
_f.write(_inner)
_f.close()
try:
    _r = subprocess.run([_py310, _f.name], timeout=115)
    sys.exit(_r.returncode)
finally:
    os.unlink(_f.name)
"""

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Assemble test program from code_context and model continuation.

        Wraps each test in the old oe-eval-internal Lambda execution pattern:
        tempdir isolation, swallowed I/O, and exec() with empty globals.
        This ensures identical pass/fail behavior to the old system.
        """
        for response in responses:
            code_context = response.instance.metadata.get("code_context", "")
            for output in response.outputs:
                continuation = self.extract_answer(output)
                if continuation and code_context:
                    # DS-1000 test harness format:
                    # code_context defines test_execution() and optionally test_string()
                    inner_code = (
                        code_context
                        + "\n"
                        + f"code = {repr(continuation)}\n"
                        + "test_execution(code)\n"
                        + ("test_string(code)\n" if "test_string(" in code_context else "\n")
                    )
                    test_program = f"_TEST_CODE = {repr(inner_code)}\n" + self._EXEC_WRAPPER
                    output.extracted_answer = test_program
                else:
                    output.extracted_answer = None


register_variant(
    "ds1000",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(answer_prefix=""),
)

register_variant(
    "ds1000",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricByteAvg(),),
)

register_variant(
    "ds1000",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=DS1000Scorer),),
)

register_variant(
    "ds1000",
    "olmo3base",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(answer_prefix=""),
    metrics=(PassAtKMetric(k=1, scorer=DS1000Scorer),),
)
