"""Formatter base class and implementations."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from .types import Instance, LMRequest, RequestType


@dataclass
class Formatter(ABC):
    """Abstract base class for formatting instances into LM requests.

    Subclasses must define:
        - format(): method to convert an instance to an LMRequest
    """

    @property
    @abstractmethod
    def request_type(self) -> RequestType:
        """The type of request this formatter produces."""
        ...

    @abstractmethod
    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        """Format an instance with optional few-shot examples."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {"type": self.__class__.__name__, **asdict(self)}


@dataclass(slots=True)
class ChatFormatter(Formatter):
    """Format instances as chat messages.

    Attributes:
        system_prompt: System prompt to include. Added both as a system message
            in the messages list and as the system_prompt field on LMRequest.
        user_template: Template for user messages (uses {question}).
        assistant_template: Template for assistant messages (uses {answer}).
    """

    system_prompt: str = ""
    user_template: str = "{question}"
    assistant_template: str = "{answer}"

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

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
        return LMRequest(
            request_type=self.request_type,
            messages=tuple(messages),
            system_prompt=self.system_prompt if self.system_prompt else None,
        )


@dataclass(slots=True)
class CompletionFormatter(Formatter):
    """Format instances as completion prompts."""

    template: str = "{question}"
    fewshot_separator: str = "\n\n"
    answer_prefix: str = ""
    fewshot_answer_key: str | None = None

    @property
    def request_type(self) -> RequestType:
        return RequestType.COMPLETION

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        parts: list[str] = []
        for ex in fewshot or []:
            example = self.template.format(question=ex.question)
            # Use fewshot_answer_key from metadata if specified, otherwise use gold_answer
            if self.fewshot_answer_key and self.fewshot_answer_key in ex.metadata:
                answer = ex.metadata[self.fewshot_answer_key]
            else:
                answer = ex.gold_answer
            if answer:
                example += self.answer_prefix + str(answer)
            parts.append(example)
        parts.append(self.template.format(question=instance.question) + self.answer_prefix)
        prompt = self.fewshot_separator.join(parts)
        return LMRequest(request_type=self.request_type, prompt=prompt)


@dataclass(slots=True)
class MultipleChoiceFormatter(Formatter):
    """Format multiple choice with continuations for logprob scoring."""

    template: str = "{question}"
    choice_template: str = "{choice}"
    include_choices_in_prompt: bool = True

    @property
    def request_type(self) -> RequestType:
        return RequestType.LOGLIKELIHOOD

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
            request_type=self.request_type,
            prompt=prompt,
            continuations=continuations,
        )


@dataclass(slots=True)
class MCQAChatFormatter(Formatter):
    """Format multiple choice questions for chat-based CoT generation."""

    system_prompt: str = ""

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

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

        return LMRequest(
            request_type=self.request_type,
            messages=tuple(messages),
            system_prompt=self.system_prompt if self.system_prompt else None,
        )


@dataclass(slots=True)
class PPLFormatter(Formatter):
    """Format instances for perplexity/BPB (bits-per-byte) evaluation.

    Uses the question as context and measures P(answer | question).
    This avoids the first-token logprob issue where vLLM returns None
    for prompt_logprobs[0] when there's no conditioning context.

    For multiple choice tasks:
    - Uses the actual gold answer TEXT (not the letter) via gold_idx
    - Falls back to gold_text metadata or gold_answer
    """

    fewshot_separator: str = "\n\n"
    leading_space: bool = True
    # This matches oe-eval's multilingual_mbpp behavior where the prompt always
    # has "\n\n" before the current doc's text (due to: join(...) + "\n\n" + text).
    always_prepend_separator: bool = False
    answer_prefix: str = ""

    @property
    def request_type(self) -> RequestType:
        return RequestType.LOGLIKELIHOOD

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
                # Concatenate with optional prefix
                example += self.answer_prefix + ex.gold_answer
            parts.append(example)

        # Add the current instance question
        if instance.question:
            parts.append(instance.question)

        prompt = self.fewshot_separator.join(parts)

        if self.always_prepend_separator and prompt:
            prompt = self.fewshot_separator + prompt

        # Optionally add leading space when there's context (standard tokenization)
        # For code tasks like MBPP, this should be disabled
        if self.leading_space and prompt and not gold_text.startswith(("\n", " ")):
            gold_text = " " + gold_text

        return LMRequest(
            request_type=self.request_type,
            prompt=prompt,
            continuations=(gold_text,),
        )
