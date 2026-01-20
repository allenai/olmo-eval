"""Formatter protocols and implementations."""

from dataclasses import dataclass
from typing import Protocol

from .types import Instance, LMRequest, RequestType


class Formatter(Protocol):
    """Protocol for formatting instances into LM requests."""

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        """Format an instance with optional few-shot examples."""
        ...


@dataclass(slots=True)
class ChatFormatter:
    """Format instances as chat messages."""

    system_prompt: str = ""
    user_template: str = "{question}"
    assistant_template: str = "{answer}"

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for ex in fewshot or []:
            messages.append(
                {
                    "role": "user",
                    "content": self.user_template.format(question=ex.question),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.assistant_template.format(answer=ex.gold_answer or ""),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": self.user_template.format(question=instance.question),
            }
        )
        return LMRequest(request_type=RequestType.CHAT, messages=tuple(messages))


@dataclass(slots=True)
class CompletionFormatter:
    """Format instances as completion prompts."""

    template: str = "{question}"
    fewshot_separator: str = "\n\n"
    answer_prefix: str = ""

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        parts: list[str] = []
        for ex in fewshot or []:
            example = self.template.format(question=ex.question)
            if ex.gold_answer:
                example += self.answer_prefix + ex.gold_answer
            parts.append(example)
        parts.append(self.template.format(question=instance.question) + self.answer_prefix)
        prompt = self.fewshot_separator.join(parts)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)


@dataclass(slots=True)
class MultipleChoiceFormatter:
    """Format multiple choice with continuations for logprob scoring."""

    template: str = "{question}"
    choice_template: str = "{choice}"
    include_choices_in_prompt: bool = True

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        prompt = self.template.format(question=instance.question)
        continuations: tuple[str, ...] = ()
        if instance.choices:
            if self.include_choices_in_prompt:
                # Add labeled choices to the prompt
                choices_text = "\n".join(
                    f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(instance.choices)
                )
                prompt = f"{prompt}\n\n{choices_text}"
            continuations = tuple(self.choice_template.format(choice=c) for c in instance.choices)
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=continuations,
        )


@dataclass(slots=True)
class MCQAChatFormatter:
    """Format multiple choice questions for chat-based CoT generation."""

    system_prompt: str = ""

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        messages: list[dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Format question with choices
        question_text = instance.question
        if instance.choices:
            choices_text = "\n".join(
                f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(instance.choices)
            )
            question_text = f"{question_text}\n\n{choices_text}"

        messages.append({"role": "user", "content": question_text})

        return LMRequest(request_type=RequestType.CHAT, messages=tuple(messages))


@dataclass(slots=True)
class PPLFormatter:
    """Format instances for perplexity/BPB (bits-per-byte) evaluation.

    This formatter creates requests for loglikelihood scoring where the
    gold answer is used as the continuation to evaluate. It does not use
    few-shot examples, as BPB evaluation typically measures raw language
    modeling performance on the target text.

    The instance's gold_answer is treated as the text to compute logprobs over.
    """

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        # For BPB evaluation, we use an empty prompt and the gold answer as continuation
        # This measures the model's logprob of generating the gold text
        if instance.gold_answer is None:
            raise ValueError("PPLFormatter requires instance.gold_answer to be set")

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="",
            continuations=(instance.gold_answer,),
        )
