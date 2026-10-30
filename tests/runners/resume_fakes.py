"""A provider and judge whose answers depend on the request, for the resume tests.

The mock provider gives every request the same answer, so a run that restored
outputs onto the wrong instance would look correct. Here each output is a function
of a hash of the prompt and the sample index: about half repeat the answer the
fake dataset hides in the prompt (``<<say:...>>``), the rest are wrong, and some
requests fail, either from their own context overflow (final) or as a marked
connection error (which a resumed run tries again). Calls also take a random few
milliseconds, so results arrive out of order. Generations carry per-token logprobs
and their summaries, as vllm_server returns them when asked for logprobs.

Workers are separate processes, so the provider reaches them through
:func:`inference_worker`, which the tests put in place of the real worker target.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from typing import Any

from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, SamplingParams
from olmo_eval.inference.errors import CONTEXT_OVERFLOW_KEY, REQUEST_ERROR_KEY
from olmo_eval.inference.providers.mock import MockProvider

_SAY = re.compile(r"<<say:(.*?)>>", re.DOTALL)

FAILED_REQUEST = "context overflow: maximum context length of the fake model exceeded"
TRANSIENT_REQUEST = "ConnectionError: the fake server reset the connection"


def digest(*parts: Any) -> int:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return int.from_bytes(hashlib.sha256(blob.encode()).digest()[:8], "big")


def _prompt(request: LMRequest) -> str:
    return request.prompt or json.dumps(request.messages, sort_keys=True, default=str)


async def _latency() -> None:
    await asyncio.sleep(random.uniform(0.0, 0.01))


def _token_logprobs(text: str, prompt: str, sample: int) -> list[LogProbEntry]:
    """Per-token logprobs of a generation, one entry per word, as vllm_server returns them."""
    return [
        {"token": token, "logprob": -(digest(prompt, sample, i) % 5000) / 1000.0}
        for i, token in enumerate(re.findall(r"\s*\S+", text))
    ]


class HashProvider(MockProvider):
    """Answers each request according to a hash of its prompt and sample index."""

    def generate(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        results = []
        for request in requests:
            prompt = _prompt(request)
            failure = digest(prompt, "fail") % 9
            if failure == 0:
                metadata = {REQUEST_ERROR_KEY: FAILED_REQUEST, CONTEXT_OVERFLOW_KEY: True}
                results.append([LMOutput(text="", metadata=metadata)])
                continue
            if failure == 1:
                results.append([LMOutput(text="", metadata={REQUEST_ERROR_KEY: TRANSIENT_REQUEST})])
                continue
            said = _SAY.findall(prompt)
            outputs = []
            for sample in range(params.num_samples):
                h = digest(prompt, sample)
                text = said[-1] if said and h % 2 == 0 else f"I am not sure, maybe {h % 97}."
                logprobs = _token_logprobs(text, prompt, sample)
                outputs.append(
                    LMOutput(
                        text=text,
                        logprobs=logprobs,
                        metadata={
                            "finish_reason": "stop" if h % 5 else "length",
                            "prompt_tokens": len(prompt) // 4,
                            "completion_tokens": 1 + h % 50,
                            # As vllm_server sets them when it returns logprobs.
                            "sum_logits": sum(entry["logprob"] for entry in logprobs),
                            "num_tokens": len(logprobs),
                            "num_tokens_all": len(logprobs),
                        },
                    )
                )
            results.append(outputs)
        return results

    async def agenerate(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        await _latency()
        return self.generate(requests, sampling_params)

    def logprobs(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        results = []
        for request in requests:
            outputs = []
            for continuation in request.continuations or ():
                logprob = -(digest(request.prompt, continuation) % 1000) / 100.0
                outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=[{"token": continuation, "logprob": logprob}],
                        metadata={"sum_logits": logprob, "num_tokens": 1, "num_tokens_all": 1},
                    )
                )
            results.append(outputs)
        return results

    async def alogprobs(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        await _latency()
        return self.logprobs(requests, sampling_params)


JUDGE_CALLS: list[str] = []


async def judge(prompt: Any, *, system_prompt: str | None = None) -> str:
    """Grade by a hash of the prompt; some calls fail, and some replies give no verdict."""
    JUDGE_CALLS.append(str(prompt))
    await _latency()
    h = digest(str(prompt), "judge")
    if h % 5 == 0:
        raise ConnectionError("fake judge unavailable")
    if h % 5 == 1:
        return "I cannot decide."
    return "VERDICT: CORRECT" if h % 2 else "VERDICT: INCORRECT"


def inference_worker(*args: Any, **kwargs: Any) -> None:
    """The real worker, with the mock provider kind serving HashProvider."""
    import olmo_eval.inference as inference
    from olmo_eval.runners.asynq.workers import inference_worker as real_worker

    real_create = inference.create_provider

    def create_provider(kind: Any, model_name: str, *a: Any, **kw: Any) -> Any:
        if str(kind) == "mock":
            return HashProvider(model_name)
        return real_create(kind, model_name, *a, **kw)

    inference.create_provider = create_provider
    real_worker(*args, **kwargs)
