"""BrowseComp with live web search and the reference GPT-4.1 grader.

Questions come from OpenAI's BrowseComp test set, which is downloaded and
decrypted on first use. The answering prompt and grader follow the simple-evals
implementation. The agent searches the open web with Serper and reads pages with
Jina Reader through the harness's tools.

Usage:
    olmo-eval run -m /path/to/checkpoint --harness browsecomp -t browsecomp

Set ``BROWSECOMP_HOME`` to choose where the test set is cached.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.types import Instance, LMRequest, Response
from olmo_eval.data import DataSource, SourceType
from olmo_eval.evals.tasks.browsecomp_common import (
    BrowseCompCalibration,
    BrowseCompMetric,
    BrowseCompScorer,
    decrypt_with_password,
    grade_with_judge,
    run_completed,
    trajectory_statistics,
)
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)

DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
DATASET_FILENAME = "browse_comp_test_set.csv"
EXPECTED_QUESTIONS = 1266
SEARCH_TOOL = "serper_google_webpage_search"
FETCH_TOOL = "jina_fetch_webpage_content"

# Answering prompt from openai/simple-evals browsecomp_eval.py.
QUERY_TEMPLATE = (
    "{Question}\n"
    "\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer}}\n"
    "Exact Answer: {{your succinct, final answer}}\n"
    "Confidence: {{your confidence score between 0% and 100% for your answer}}"
)


def artifact_home() -> Path:
    """Return the directory where the BrowseComp test set is cached."""
    return (
        Path(os.environ.get("BROWSECOMP_HOME", "~/.cache/olmo-eval/browsecomp"))
        .expanduser()
        .resolve()
    )


def dataset_path() -> Path:
    return artifact_home() / DATASET_FILENAME


def download_dataset(destination: Path) -> Path:
    """Fetch the encrypted test set and write it atomically."""
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    try:
        with httpx.stream("GET", DATASET_URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


MEAN_METRICS = tuple(
    BrowseCompMetric(name=name, scorer=BrowseCompScorer(name=name))
    for name in ("accuracy", "search_calls", "fetch_calls")
)


@register("browsecomp")
class BrowseComp(Task):
    """Open-web research questions scored by the reference GPT-4.1 grader."""

    data_source = DataSource(path=DATASET_URL, split="test")
    formatter = ChatFormatter(user_template=QUERY_TEMPLATE.replace("{Question}", "{question}"))
    metrics = (*MEAN_METRICS, BrowseCompCalibration())
    primary_metric = MEAN_METRICS[0]
    judge_model = "gpt-4.1"
    judge_max_tokens = 1024
    strip_thinking = True
    required_secrets = ("OPENAI_API_KEY",)

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Load the cached local copy of the test set, downloading it on first use."""
        from olmo_eval.data import DataLoader

        source = self.config.get_data_source(split=split)
        if source.path == DATASET_URL:
            cached = dataset_path()
            if not cached.is_file():
                logger.info("Downloading the BrowseComp test set to %s", cached)
                download_dataset(cached)
            source = replace(source, path=str(cached), source_type=SourceType.LOCAL)
        for index, doc in enumerate(DataLoader().load(source)):
            yield self.process_doc(doc, index)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        canary = doc["canary"]
        question = decrypt_with_password(doc["problem"], canary)
        answer = decrypt_with_password(doc["answer"], canary)
        if not question or not answer:
            raise ValueError(f"Incomplete BrowseComp record at row {index}")
        return Instance(
            question=question,
            gold_answer=answer,
            metadata={"id": f"browsecomp_{index}", "topic": doc.get("problem_topic")},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is None:
            raise ValueError("BrowseComp requires an answering prompt")
        return self.config.formatter.format(instance)

    async def score_responses(
        self, responses: Sequence[Response], context: Any = None
    ) -> Sequence[Response]:
        """Grade final responses and count tool use across their trajectories."""
        import openai

        if self.config.sampling_params and self.config.sampling_params.num_samples != 1:
            raise ValueError("BrowseComp evaluates one agent trajectory per question")
        async with openai.AsyncOpenAI() as client:
            for response in responses:
                if len(response.outputs) > 1:
                    raise ValueError("BrowseComp evaluates one agent trajectory per question")
                _, counts = trajectory_statistics(response)
                scores = {metric.name: 0.0 for metric in MEAN_METRICS}
                scores.update(
                    search_calls=float(counts.get(SEARCH_TOOL, 0)),
                    fetch_calls=float(counts.get(FETCH_TOOL, 0)),
                )
                response.scores.update(scores)
                output = response.outputs[0] if response.outputs else None
                text = output.text if output is not None else ""
                completed = run_completed(response, output)
                verdict: dict[str, Any] = {
                    "parse_error": True,
                    "error": "Response incomplete or cannot be parsed",
                }
                raw = None
                if completed:
                    verdict, raw = await grade_with_judge(
                        client,
                        question=response.instance.question,
                        response_text=text,
                        correct_answer=response.instance.gold_answer or "",
                        model=self.config.judge_model or "gpt-4.1",
                        max_output_tokens=self.config.judge_max_tokens or 1024,
                        query_id=response.instance.metadata["id"],
                    )
                    scores["accuracy"] = float(verdict["correct"] is True)
                response.scores.update(scores)
                if output is not None:
                    output.metadata["judge_result"] = {
                        **verdict,
                        "raw_response": raw,
                        "judge_model": self.config.judge_model,
                        "is_completed": completed,
                        "tool_call_counts": counts,
                    }
                    for name, score in scores.items():
                        output.metadata[f"score:{name}"] = score
        return responses


register_variant("browsecomp", "mini", limit=10)
