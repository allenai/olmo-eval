"""HumanEval Edit — code editing variant of HumanEval infilling.

Instead of FIM tokens, presents the model with an incomplete function
(prefix + suffix concatenated with a TODO marker) and a natural language
instruction to fill in the missing implementation. Scored with the same
test harness as standard HumanEval (code execution + pass@k).

Two tasks from the same dataset:
- humaneval_edit: full-body infill (empty suffix) — "implement this function"
- humaneval_edit_mid: middle infill (has suffix) — "fill in the gap"

Uses the loubnabnl/humaneval_infilling dataset (HumanEval-MultiLineInfilling).
"""

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import ChatFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetric, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import HUMANEVAL_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code, indent_code
from olmo_eval.evals.tasks.common import Task, register, register_variant


EDIT_MARKER = "    # TODO: implement this"


def _build_edit_context(prompt: str, suffix: str) -> str:
    """Build incomplete function: prefix + TODO marker + suffix."""
    context = prompt + EDIT_MARKER
    if suffix.strip():
        context += "\n" + suffix
    return context


def _select_best_mid_variant(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the mid-infill variant with the most balanced split.

    Maximizes min(middle_len, suffix_len) to avoid trivial cases.
    """
    def score(doc: dict[str, Any]) -> int:
        return min(
            len(doc.get("canonical_solution", "").strip()),
            len(doc.get("suffix", "").strip()),
        )

    return max(candidates, key=score)


@register("humaneval_edit")
class HumanEvalEdit(Task):
    """HumanEval Edit — fill in missing code given surrounding context.

    Full-body variant: empty suffix, model writes entire implementation.
    158 problems (one per HumanEval entry point).
    """

    data_source = DataSource(
        path="loubnabnl/humaneval_infilling",
        subset="HumanEval-MultiLineInfilling",
    )
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    )
    fewshot_split: str = "test"

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            seen: set[str] = set()
            for doc in loader.load(source):
                ep = doc.get("entry_point", "")
                if ep in seen:
                    continue
                if doc.get("suffix", "").strip():
                    continue
                seen.add(ep)
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        prompt = doc["prompt"]
        suffix = doc.get("suffix", "")
        entry_point = doc["entry_point"]
        unit_tests = doc["test"] + f"\ncheck({entry_point})"
        incomplete = _build_edit_context(prompt, suffix)

        return Instance(
            question=incomplete,
            gold_answer=doc["canonical_solution"],
            metadata={
                "id": doc["task_id"],
                "entry_point": entry_point,
                "answer_prefix": prompt,
                "suffix": suffix,
                "test": unit_tests,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_code(output.text)

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        for response in responses:
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    code = indent_code(code)
                    full_code = response.instance.metadata["answer_prefix"] + code
                    suffix = response.instance.metadata.get("suffix", "")
                    if suffix.strip():
                        full_code += "\n" + suffix
                    output.extracted_answer = full_code
                else:
                    output.extracted_answer = None


@register("humaneval_edit_mid")
class HumanEvalEditMid(HumanEvalEdit):
    """HumanEval Edit Mid — fill in missing code in the middle of a function.

    Harder: model generates code that fits between existing prefix AND suffix.
    125 problems (one per entry point, best-balanced split selected).
    """

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            candidates: dict[str, list[dict[str, Any]]] = {}
            for doc in loader.load(source):
                if not doc.get("suffix", "").strip():
                    continue
                ep = doc.get("entry_point", "")
                candidates.setdefault(ep, []).append(doc)
            for ep, docs in sorted(candidates.items()):
                best = _select_best_mid_variant(docs)
                self._instances_cache.append(self.process_doc(best))
        yield from self._instances_cache


# =============================================================================
# Variants — humaneval_edit (full body)
# =============================================================================

_EDIT_CHAT_TEMPLATE = """\
The following Python function is incomplete. Replace the `# TODO: implement this` \
comment with the correct implementation. Write only the function body code.

```python
{question}
```"""

register_variant(
    "humaneval_edit",
    "chat",
    formatter=ChatFormatter(
        user_template=_EDIT_CHAT_TEMPLATE,
        assistant_template="{answer}",
    ),
)

register_variant(
    "humaneval_edit",
    "bpb",
    formatter=PPLFormatter(leading_space=True, answer_prefix=" "),
    metrics=(BPBMetric(),),
)

register_variant(
    "humaneval_edit",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.2,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)

register_variant(
    "humaneval_edit",
    "pass_at_10",
    metrics=(PassAtKMetric(k=10, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        num_samples=10,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)

# =============================================================================
# Variants — humaneval_edit_mid (middle infill)
# =============================================================================

_MID_EDIT_CHAT_TEMPLATE = """\
The following Python function has a missing section marked with `# TODO: implement this`. \
The code before and after the marker is correct. Write only the missing code that \
should replace the TODO comment.

```python
{question}
```"""

register_variant(
    "humaneval_edit_mid",
    "chat",
    formatter=ChatFormatter(
        user_template=_MID_EDIT_CHAT_TEMPLATE,
        assistant_template="{answer}",
    ),
)

register_variant(
    "humaneval_edit_mid",
    "bpb",
    formatter=PPLFormatter(leading_space=True, answer_prefix=" "),
    metrics=(BPBMetric(),),
)

register_variant(
    "humaneval_edit_mid",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.2,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)

register_variant(
    "humaneval_edit_mid",
    "pass_at_10",
    metrics=(PassAtKMetric(k=10, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        num_samples=10,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)
