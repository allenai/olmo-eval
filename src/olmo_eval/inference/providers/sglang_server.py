"""Provider for a running SGLang server, or an sglang_router in front of several."""

from __future__ import annotations

from typing import Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams

from .vllm_server import VLLMServerProvider


class SGLangServerProvider(VLLMServerProvider):
    """Provider that talks to an SGLang server it does not manage, over HTTP.

    SGLang serves the same OpenAI-compatible chat and completion API as vLLM, so requests,
    retries, concurrency and context fitting are VLLMServerProvider's. What differs:

    - Prompts are counted with the model's own tokenizer and chat template, the ones the
      server loads, because sglang_router does not forward /tokenize.
    - Prompt truncation (truncate_prompt_tokens, truncation_side) is refused: SGLang has no
      such request field and would ignore it.
    - Log-likelihood scoring is not supported yet: SGLang returns no prompt_logprobs.

    Example:
        provider = SGLangServerProvider("Qwen/Qwen3-0.6B", base_url="http://localhost:30000/v1")
    """

    def __init__(self, model_name: str, base_url: str | None = None, **kwargs: Any) -> None:
        if not base_url:
            raise ValueError(
                "provider.kind=sglang_server connects to a running SGLang server: "
                "set provider.base_url"
            )
        super().__init__(model_name, base_url=base_url, **kwargs)

    def _get_tokenizer(self, *, require_local: bool = False) -> Any:
        # The remote tokenizer calls /tokenize, which sglang_router does not forward.
        return super()._get_tokenizer(require_local=True)

    async def _count_tokens(self, payload: dict[str, Any]) -> int | None:
        tokenizer = self._get_tokenizer()
        if "messages" not in payload:
            return len(tokenizer.encode(payload["prompt"]))
        prompt = tokenizer.apply_chat_template(
            payload["messages"],
            tools=payload.get("tools"),
            add_generation_prompt=True,
            tokenize=False,
            **payload.get("chat_template_kwargs", {}),
        )
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    def _get_prompt_truncation(self, params: SamplingParams) -> tuple[int | None, str | None]:
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            raise ValueError(
                "SGLang has no prompt truncation, so sglang_server cannot honour "
                "truncate_prompt_tokens or truncation_side"
            )
        return None, None

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        raise NotImplementedError(
            "sglang_server does not score continuations yet: SGLang returns no prompt_logprobs"
        )
