"""Functions for building predictions and requests from evaluation responses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from olmo_eval.common.types import SamplingParams


def build_predictions(scored: Sequence[Any]) -> list[dict]:
    """Build per-instance predictions from scored responses.

    Args:
        scored: Sequence of scored Response objects

    Returns:
        List of prediction dicts suitable for JSONL output
    """
    predictions = []
    for idx, resp in enumerate(scored):
        # Build model_output from LMOutput objects
        model_output = []
        for out in resp.outputs:
            # Get values from metadata (set by backend) or compute from logprobs
            meta = out.metadata or {}
            num_bytes = len(out.text.encode("utf-8")) if out.text else 0
            num_chars = len(out.text) if out.text else 0

            # Use metadata values if available (from vLLM backend), else compute
            if "sum_logits" in meta:
                sum_logits = meta["sum_logits"]
                num_tokens = meta.get("num_tokens", len(out.logprobs) if out.logprobs else 0)
            elif out.logprobs:
                sum_logits = sum(t.get("logprob", 0) for t in out.logprobs)
                num_tokens = len(out.logprobs)
            else:
                sum_logits = 0.0
                num_tokens = 0

            out_data: dict[str, Any] = {
                "text": out.text,
                "extracted_answer": out.extracted_answer,
                "sum_logits": sum_logits,
                "num_tokens": num_tokens,
                "num_tokens_all": meta.get("num_tokens_all", num_tokens),
                "is_greedy": meta.get("is_greedy", False),
            }

            # Compute derived metrics (matching oe-eval format)
            if num_tokens > 0:
                out_data["logits_per_token"] = sum_logits / num_tokens
            if num_chars > 0:
                out_data["logits_per_char"] = sum_logits / num_chars
            if num_bytes > 0:
                out_data["bits_per_byte"] = -sum_logits / (num_bytes * math.log(2))

            out_data["num_chars"] = num_chars

            model_output.append(out_data)

        # Get label from metadata or gold_answer
        label = resp.instance.metadata.get("gold_idx", resp.instance.gold_answer)

        # Build instance_metrics in nested format {metric: {scorer: value}}
        # resp.scores is {scorer_name: value}, we need to convert to nested
        # Since instance-level scores are keyed by scorer, we use scorer as inner key
        instance_metrics: dict[str, dict[str, float]] = {}
        for scorer_name, value in resp.scores.items():
            # Each scorer produces scores under its own name
            # We use scorer_name as both outer and inner key for simplicity
            if scorer_name not in instance_metrics:
                instance_metrics[scorer_name] = {}
            instance_metrics[scorer_name][scorer_name] = value

        # Build prediction (doc and context available in requests.jsonl)
        prediction: dict[str, Any] = {
            "doc_id": idx,
            "native_id": resp.instance.metadata.get("id", f"doc_{idx}"),
            "model_output": model_output,
            "instance_metrics": instance_metrics,
            "label": label,
        }

        # Add final_output text for chat/agent tasks
        if resp.outputs and resp.outputs[0].text:
            prediction["final_output"] = resp.outputs[0].text

        # Add trajectory if present (multi-turn/agent tasks)
        if resp.trajectory is not None:
            prediction["trajectory"] = resp.trajectory.to_dict()

        predictions.append(prediction)

    return predictions


def build_requests(
    instances: Sequence[Any],
    requests: Sequence[Any],
    task_name: str,
    sampling_params: SamplingParams | None = None,
) -> list[dict]:
    """Build per-instance request objects in oe-eval compatible format.

    This produces the same format as oe-eval's *-requests.jsonl files, which
    shows exactly what the model saw during evaluation.

    Args:
        instances: Sequence of Instance objects
        requests: Sequence of LMRequest objects (parallel to instances)
        task_name: Name of the task
        sampling_params: Optional sampling parameters

    Returns:
        List of request dicts suitable for JSONL output with oe-eval compatible schema:
        {
            "request_type": "loglikelihood" | "generate_until" | ...,
            "doc": {
                "query": "...",  # The instance question
                "choices": [...],  # For BPB/MC tasks
                # Plus original metadata
            },
            "request": {
                "context": "...",  # Full prompt (few-shot + current)
                "continuation": "...",  # For loglikelihood: the text being scored
                "perplexity_context": "...",  # Usually same as context
                "stop_sequences": [...],
                "generation_kwargs": {...}
            },
            "idx": 0,
            "task_name": "...",
            "doc_id": 0,
            "native_id": "...",
            "label": ...
        }
    """
    from olmo_eval.common.types import RequestType

    request_list = []

    for idx, (instance, request) in enumerate(zip(instances, requests, strict=True)):
        # Build doc from instance
        doc: dict[str, Any] = {
            "query": instance.question,
            **instance.metadata,
        }

        # Add choices for BPB and MC tasks
        if instance.choices:
            doc["choices"] = list(instance.choices)
        elif request.continuations:
            # For BPB tasks without explicit choices, use continuations
            doc["choices"] = list(request.continuations)

        # Build request object based on request type
        request_dict: dict[str, Any] = {}

        if request.request_type == RequestType.LOGLIKELIHOOD:
            # oe-eval's GenerateUntilAndLoglikelihoodRequest format
            request_dict["context"] = request.prompt
            request_dict["perplexity_context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
        elif request.request_type == RequestType.COMPLETION:
            # oe-eval's GenerateUntilRequest format
            request_dict["context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
        elif request.request_type == RequestType.CHAT:
            # Chat format - context is the messages
            request_dict["context"] = list(request.messages)

        # Add generation kwargs
        if sampling_params:
            request_dict["stop_sequences"] = (
                list(sampling_params.stop_sequences) if sampling_params.stop_sequences else []
            )
            request_dict["generation_kwargs"] = {
                "max_gen_toks": sampling_params.max_tokens,
                "do_sample": sampling_params.temperature > 0,
                "temperature": sampling_params.temperature,
            }
            if sampling_params.top_p is not None:
                request_dict["generation_kwargs"]["top_p"] = sampling_params.top_p
        else:
            request_dict["stop_sequences"] = []
            request_dict["generation_kwargs"] = {}

        # Determine request type string (oe-eval naming)
        if request.request_type == RequestType.LOGLIKELIHOOD:
            request_type_str = "loglikelihood"
        elif request.request_type == RequestType.COMPLETION:
            if request.continuations:
                request_type_str = "generate_until_and_loglikelihood"
            else:
                request_type_str = "generate_until"
        elif request.request_type == RequestType.CHAT:
            request_type_str = "generate_until"
        else:
            request_type_str = "unknown"

        # Determine label (ground truth)
        label = instance.metadata.get("gold_idx")
        if label is None and instance.gold_answer is not None:
            label = instance.gold_answer

        request_list.append(
            {
                "request_type": request_type_str,
                "doc": doc,
                "request": request_dict,
                "idx": 0,  # For multi-sample, this would vary
                "task_name": task_name,
                "doc_id": idx,
                "native_id": instance.metadata.get("id", f"doc_{idx}"),
                "label": label,
            }
        )

    return request_list
