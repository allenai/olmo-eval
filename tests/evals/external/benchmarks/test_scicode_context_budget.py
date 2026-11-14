"""SciCode budget fitting across sequential HTTP-backed requests."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import httpx

from olmo_eval.cli.utils import parse_key_value_args
from olmo_eval.common.types import SamplingParams
from olmo_eval.evals.external.benchmarks.scicode import eval as scicode_eval
from olmo_eval.evals.external.benchmarks.scicode import loader as scicode_loader
from olmo_eval.inference.providers.vllm_server import VLLMServerProvider


def problem(steps: int) -> scicode_loader.SciCodeProblem:
    return scicode_loader.SciCodeProblem(
        problem_id="99",
        problem_name="budget",
        required_dependencies="import numpy as np",
        sub_steps=[
            {
                "step_description_prompt": f"Step {i}",
                "function_header": f"def step_{i}():",
                "return_line": "    return None",
            }
            for i in range(steps)
        ],
    )


class TestConfigForwarding(unittest.IsolatedAsyncioTestCase):
    async def test_external_cli_boolean_reaches_sampling_and_result_metadata(self) -> None:
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                args = parse_key_value_args(
                    [f"fit_max_tokens_to_context={str(enabled).lower()}"], coerce_types=True
                )
                self.assertIs(args["fit_max_tokens_to_context"], enabled)
                provider = mock.Mock(spec=VLLMServerProvider)
                provider.chat_template_kwargs = None
                provider.model_name = "model"
                evaluator = scicode_eval.SciCodeExternalEval()
                with (
                    mock.patch.object(scicode_loader, "load_problems", return_value=[problem(1)]),
                    mock.patch.object(
                        evaluator,
                        "_run_problem",
                        new_callable=mock.AsyncMock,
                        return_value={"total": 1, "passed": 1, "all_passed": True},
                    ) as run,
                ):
                    result = await evaluator.execute(
                        provider,
                        {
                            **args,
                            "h5py_host_path": "/test.h5",
                            "max_tokens": 32768,
                            "temperature": 1.0,
                        },
                    )
                sampling = run.call_args.kwargs["sampling_params"]
                self.assertIs(sampling.fit_max_tokens_to_context, enabled)
                self.assertEqual(sampling.max_tokens, 32768)
                self.assertEqual(sampling.temperature, 1.0)
                self.assertIs(result.metadata["fit_max_tokens_to_context"], enabled)
        self.assertFalse(scicode_eval.SciCodeConfig().fit_max_tokens_to_context)


class TestSequentialHTTPBudget(unittest.IsolatedAsyncioTestCase):
    async def run_cascade(
        self,
        prompt_lengths: list[int],
        *,
        context_known: bool = True,
        count_known: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], mock.AsyncMock]:
        prompts: list[dict[str, Any]] = []
        generations: list[dict[str, Any]] = []
        provider = VLLMServerProvider("model", base_url="http://budget.invalid/v1", max_retries=0)
        provider.chat_template_kwargs = {"enable_thinking": False}
        client_class, async_client_class = httpx.Client, httpx.AsyncClient
        counts = iter(prompt_lengths)

        def transport(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                model = {"id": "model", **({"max_model_len": 34816} if context_known else {})}
                return httpx.Response(200, json={"data": [model]})
            self.assertEqual(request.url.path, "/tokenize")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "model")
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            prompts.append(payload)
            if not count_known:
                return httpx.Response(404, json={"error": "tokenization unavailable"})
            return httpx.Response(200, json={"count": next(counts)})

        async def generate(**kwargs: Any) -> Any:
            generations.append(kwargs)
            prompt_length = prompt_lengths[len(generations) - 1]
            if kwargs["max_tokens"] + prompt_length > 34816:
                raise ValueError("This model's maximum context length is 34816 tokens.")
            code = (
                f"```python\ndef step_{len(generations)}():\n"
                f"    return 'MARKER_{len(generations)}'\n```"
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=code, tool_calls=None))]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=generate))
        )
        evaluator = scicode_eval.SciCodeExternalEval()
        verify = mock.AsyncMock(return_value=[True] * len(prompt_lengths))
        with (
            mock.patch.object(provider, "_get_or_create_client", return_value=fake_client),
            mock.patch.object(
                httpx,
                "Client",
                side_effect=lambda **kw: client_class(
                    transport=httpx.MockTransport(transport), **kw
                ),
            ),
            mock.patch.object(
                httpx,
                "AsyncClient",
                side_effect=lambda **kw: async_client_class(
                    transport=httpx.MockTransport(transport), **kw
                ),
            ),
            mock.patch.object(evaluator, "_verify", verify),
        ):
            await evaluator._run_problem(
                problem=problem(len(prompt_lengths)),
                provider=provider,
                sampling_params=SamplingParams(
                    max_tokens=32768, temperature=1.0, fit_max_tokens_to_context=True
                ),
                sc_args=scicode_eval.SciCodeConfig(),
                container_runtime="podman",
            )
        return prompts, generations, verify

    async def test_each_substep_counts_its_actual_growing_prompt(self) -> None:
        prompts, generations, verify = await self.run_cascade([100, 2276, 9000])
        self.assertEqual([call["max_tokens"] for call in generations], [32768, 32538, 25814])
        self.assertNotIn("MARKER_1", prompts[0]["messages"][0]["content"])
        self.assertIn("MARKER_1", prompts[1]["messages"][0]["content"])
        self.assertIn("MARKER_2", prompts[2]["messages"][0]["content"])
        verify.assert_awaited_once()

    async def test_unknown_context_does_not_hide_an_overflow_as_empty_code(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "maximum context length"):
            await self.run_cascade([2276], context_known=False)

    async def test_unknown_token_count_does_not_hide_an_overflow_as_empty_code(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "maximum context length"):
            await self.run_cascade([2276], count_known=False)

    async def test_a_prompt_that_fills_the_window_still_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "maximum context length"):
            await self.run_cascade([34816])
