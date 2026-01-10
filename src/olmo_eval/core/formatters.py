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

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        prompt = self.template.format(question=instance.question)
        continuations: tuple[str, ...] = ()
        if instance.choices:
            continuations = tuple(self.choice_template.format(choice=c) for c in instance.choices)
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=continuations,
        )
