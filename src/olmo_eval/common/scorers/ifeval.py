"""Scorer for IFBench / IFEval instruction-following evaluation.

Uses the ``ifbench`` package registry, which covers the original IFEval
(DEFAULT) verifiers, the OOD verifiers used by ``allenai/IFBench_test2``,
and the verifiers used by the multi-turn ``VGraf/ifeval_mt`` slices. The
scorer evaluates a response against per-instance instructions (looked up
in ``instance.metadata["instruction_id_list"]`` / ``"kwargs"``) and writes
both strict and loose pass/fail lists for each instruction into
``output.metadata["ifeval"]``. The four IFEval metrics consume that field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput


def _loose_response_variants(response: str) -> list[str]:
    """Generate the eight response variants used by upstream loose scoring."""
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def _check_one(
    instruction_cls: Any,
    instruction_id: str,
    kwargs: dict[str, Any],
    prompt: str,
    response: str,
) -> bool:
    instruction = instruction_cls(instruction_id)
    cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    arg_keys = instruction.get_instruction_args_keys()
    if "prompt_to_repeat" in arg_keys and not cleaned_kwargs.get("prompt_to_repeat"):
        cleaned_kwargs["prompt_to_repeat"] = prompt
    instruction.build_description(**cleaned_kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    return bool(response.strip()) and bool(instruction.check_following(response))


@dataclass(frozen=True)
class IFEvalScorer(Scorer):
    """Run IFBench/IFEval instruction verifiers against a response.

    The numeric ``score()`` return is the prompt-level loose accuracy (1.0 if
    every instruction passes under at least one loose variant, else 0.0). The
    full strict + loose pass lists are written to ``output.metadata["ifeval"]``
    so the four metric classes can derive prompt/inst × strict/loose figures.
    """

    name: ClassVar[str] = "ifeval"

    def score(self, instance: Instance, output: LMOutput) -> float:
        instruction_ids: list[str] = instance.metadata.get("instruction_id_list", [])
        kwargs_list: list[dict[str, Any]] = instance.metadata.get("kwargs", [])
        prompt: str = instance.metadata.get("prompt", instance.question)
        response: str = output.text or ""

        strict_results: list[bool] = []
        loose_results: list[bool] = []

        if instruction_ids:
            from ifbench import instructions_registry

            registry = instructions_registry.INSTRUCTION_DICT
            loose_variants = _loose_response_variants(response)
            for inst_id, inst_kwargs in zip(instruction_ids, kwargs_list, strict=True):
                instruction_cls = registry[inst_id]
                strict_results.append(
                    _check_one(instruction_cls, inst_id, inst_kwargs, prompt, response)
                )
                loose_pass = any(
                    _check_one(instruction_cls, inst_id, inst_kwargs, prompt, variant)
                    for variant in loose_variants
                )
                loose_results.append(loose_pass)

        if output.metadata is None:
            output.metadata = {}
        output.metadata["ifeval"] = {
            "strict": strict_results,
            "loose": loose_results,
        }

        if not loose_results:
            return 0.0
        return 1.0 if all(loose_results) else 0.0
