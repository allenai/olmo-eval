"""Replay provider: serves the generations a previous run saved, so a task can be rescored."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import REQUEST_ERROR_KEY
from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX
from olmo_eval.runners.io.writers import _lazy_loader_placeholder

#: Saved per-output fields the replayed output's metadata carries over.
_METADATA_FIELDS = ("finish_reason", "completion_tokens", "prompt_tokens")


def _prompt_key(context: Any, system_prompt: str | None) -> str:
    """The key a saved request and a live request share when the model saw the same prompt."""
    return json.dumps([context, system_prompt], sort_keys=True, default=_lazy_loader_placeholder)


def _request_key(request: LMRequest) -> str:
    context = list(request.messages) if request.messages else request.prompt
    return _prompt_key(context, request.system_prompt)


def _requests_path(predictions_path: Path) -> Path:
    """The requests file the writers place beside a predictions file."""
    name = predictions_path.name.removesuffix(PREDICTIONS_SUFFIX) + REQUESTS_SUFFIX
    return predictions_path.parent.parent.parent / "requests" / predictions_path.parent.name / name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _replayed_output(saved: dict[str, Any]) -> LMOutput:
    # The raw generation, so the task strips reasoning and extracts answers again.
    text = saved.get("original_text") or saved.get("text") or ""
    metadata = {key: saved[key] for key in _METADATA_FIELDS if key in saved}
    return LMOutput(text=text, metadata=metadata)


class ReplayProvider(InferenceProvider):
    """Answers each request with the generations a previous run saved for the same prompt.

    ``predictions`` is a predictions JSONL file or a results directory searched for
    them; each file is paired with the requests file written beside it, which holds the
    prompts. A request whose prompt was not saved, or was saved with a different number
    of samples, fails like a generation error, so the task's failure budget catches a
    replay against the wrong run. Prompts saved more than once by one file are served
    in order; a prompt saved by two files (two configurations of a task) is rejected.
    Requests with images or tools fail too: the saved files keep neither image
    identities nor native tool calls.
    """

    def __init__(self, model_name: str = "replay", predictions: str = "") -> None:
        super().__init__(model_name)
        if not predictions:
            raise ValueError("The replay provider needs provider.predictions=<file or directory>")
        root = Path(predictions)
        files = [root] if root.is_file() else sorted(root.rglob(f"*{PREDICTIONS_SUFFIX}"))
        if not files:
            raise FileNotFoundError(f"No *{PREDICTIONS_SUFFIX} files found under {root}")
        models = sorted({path.parent.name for path in files})
        if len(models) > 1:
            raise ValueError(
                f"{root} holds predictions for several models ({', '.join(models)}); "
                "point provider.predictions at one model's predictions directory or file"
            )

        self._saved: dict[str, deque[list[dict[str, Any]]]] = defaultdict(deque)
        sources: dict[str, Path] = {}
        for path in files:
            requests = {r["doc_id"]: r for r in _read_jsonl(_requests_path(path))}
            for prediction in _read_jsonl(path):
                request = requests[prediction["doc_id"]]["request"]
                key = _prompt_key(request.get("context"), request.get("system_prompt"))
                if sources.setdefault(key, path) != path:
                    raise ValueError(
                        f"{sources[key].name} and {path.name} saved the same prompt; "
                        "point provider.predictions at the predictions file to replay"
                    )
                self._saved[key].append(prediction["model_output"])

    def _replay(self, request: LMRequest, num_samples: int) -> list[LMOutput]:
        queue = self._saved.get(_request_key(request))
        if request.images or request.tools:
            error = "requests with images or tools cannot be replayed"
        elif not queue:
            error = "no saved generation matches this prompt"
        elif len(queue[0]) != num_samples:
            error = (
                f"saved {len(queue[0])} samples for this prompt, the task asks for {num_samples}"
            )
        else:
            return [_replayed_output(saved) for saved in queue.popleft()]
        return [LMOutput(text="", metadata={REQUEST_ERROR_KEY: f"replay: {error}"})]

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        num_samples = self._default_sampling_params(sampling_params).num_samples
        return [self._replay(request, num_samples) for request in requests]

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return self.generate(requests, sampling_params)

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        raise NotImplementedError("The replay provider replays generations, not logprobs")
