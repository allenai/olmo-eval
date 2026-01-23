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

    Uses the question as context and measures P(answer | question).
    This avoids the first-token logprob issue where vLLM returns None
    for prompt_logprobs[0] when there's no conditioning context.

    For multiple choice tasks:
    - Uses the actual gold answer TEXT (not the letter) via gold_idx
    - Falls back to gold_text metadata or gold_answer
    """

    fewshot_separator: str = "\n\n"
    leading_space: bool = True  # Whether to add leading space to continuation
    # Whether to always prepend separator before the current instance's question.
    # This matches oe-eval's multilingual_mbpp behavior where the prompt always
    # has "\n\n" before the current doc's text (due to: join(...) + "\n\n" + text).
    always_prepend_separator: bool = False
    # Prefix to add before gold_answer when building few-shot examples.
    # In oe-eval, doc_to_target often returns " " + answer, so we can replicate
    # this with answer_prefix=" ".
    answer_prefix: str = ""

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        # Determine the text to compute logprobs over
        gold_text: str | None = None

        # For MC tasks: use the actual choice text, not just the letter
        if instance.choices and "gold_idx" in instance.metadata:
            gold_idx = instance.metadata["gold_idx"]
            if 0 <= gold_idx < len(instance.choices):
                gold_text = instance.choices[gold_idx]

        # Fallback to gold_text from metadata if available
        if gold_text is None and "gold_text" in instance.metadata:
            gold_text = instance.metadata["gold_text"]

        # Final fallback to gold_answer
        if gold_text is None:
            gold_text = instance.gold_answer

        if gold_text is None:
            raise ValueError("PPLFormatter requires a gold answer to be set")

        # Build prompt with few-shot examples
        parts: list[str] = []
        for ex in fewshot or []:
            example = ex.question or ""
            if ex.gold_answer:
                # Concatenate with optional prefix (oe-eval uses " " for humaneval)
                example += self.answer_prefix + ex.gold_answer
            parts.append(example)

        # Add the current instance question
        if instance.question:
            parts.append(instance.question)

        prompt = self.fewshot_separator.join(parts)

        # In oe-eval's multilingual_mbpp, fewshot_context always adds "\n\n" before
        # the current doc's text: join(fewshot) + "\n\n" + doc_text + ...
        # This means even with 0-shot, the prompt starts with "\n\n".
        # The always_prepend_separator option replicates this behavior.
        if self.always_prepend_separator and prompt:
            prompt = self.fewshot_separator + prompt

        # Optionally add leading space when there's context (standard tokenization)
        # For code tasks like MBPP, this should be disabled
        if self.leading_space and prompt and not gold_text.startswith(("\n", " ")):
            gold_text = " " + gold_text

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=(gold_text,),
        )
