"""Tests for how the vLLM server startup budget is configured.

``DEFAULT_STARTUP_TIMEOUT`` is 300s, which is not enough for a 35B model under
disk contention. A launch raises it with
``-o provider.kwargs.startup_timeout=<seconds>``; these tests pin every hop that
value takes, because each one can be broken without any visible symptom other
than runs quietly reverting to the 300s default.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestStartupTimeoutIsConfigurable:
    """The startup budget must be settable from the launch site."""

    def test_provider_forwards_startup_timeout_through_kwargs_passthrough(self):
        """The provider must not swallow ``startup_timeout``.

        ``VLLMServerProvider`` deliberately does not declare the parameter; it
        falls through ``**server_kwargs`` into ``VLLMServerProcess``, which does.
        Adding an explicit ``startup_timeout`` parameter to the provider without
        forwarding it would silently pin every run back to 300s, so pin the
        passthrough itself rather than the parameter list.
        """
        from inspect import Parameter, signature

        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        params = signature(VLLMServerProvider.__init__).parameters
        assert "startup_timeout" not in params
        assert any(p.kind is Parameter.VAR_KEYWORD for p in params.values())

        mock_server = MagicMock()
        mock_server.base_url = "http://127.0.0.1:8000/v1"

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils.VLLMServerProcess",
                return_value=mock_server,
            ) as mock_server_cls,
            patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"),
        ):
            VLLMServerProvider("org/model", startup_timeout=1200)

        assert mock_server_cls.call_args.kwargs["startup_timeout"] == 1200

    def test_server_process_consumes_it_instead_of_passing_it_to_vllm(self):
        """The kwarg must be consumed by the wrapper, not handed to vLLM.

        Every kwarg ``VLLMServerProcess`` does not name becomes a
        ``--kebab-case`` CLI argument, and vLLM exits on unknown flags. Being a
        named parameter is exactly what makes this one configuration rather than
        a broken launch.
        """
        from olmo_eval.inference.providers.vllm_server_utils import (
            DEFAULT_STARTUP_TIMEOUT,
            VLLMServerProcess,
            _build_server_command,
        )

        assert "--startup-timeout" not in _build_server_command("org/model", 8000)

        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
            return_value=23456,
        ):
            server = VLLMServerProcess(model_name="org/model", port=8000, startup_timeout=1200)
            default_server = VLLMServerProcess(model_name="org/model", port=8000)

        assert server.startup_timeout == 1200
        assert "startup_timeout" not in server.server_kwargs
        # The default is deliberately left alone; raising it is the launch's job.
        assert default_server.startup_timeout == DEFAULT_STARTUP_TIMEOUT

    def test_server_process_passes_the_budget_to_the_readiness_wait(self):
        """The configured value, not the module default, must reach the wait loop."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
                return_value=23456,
            ),
            patch("subprocess.Popen", return_value=mock_process),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(True, None, None),
            ) as mock_wait,
            patch("atexit.register"),
        ):
            VLLMServerProcess(model_name="org/model", port=8000, startup_timeout=1200).start()

        assert mock_wait.call_args.kwargs["timeout"] == 1200

    def test_dotlist_override_arrives_as_a_number(self):
        """``-o provider.kwargs.startup_timeout=1200`` must arrive as an int.

        ``_wait_for_server`` compares the value against elapsed seconds, so a
        string would raise ``TypeError`` on the first iteration instead of
        extending the budget.
        """
        from olmo_eval.cli.beaker.launch import _apply_harness_overrides
        from olmo_eval.harness.config import HarnessConfig

        harness_config = _apply_harness_overrides(
            HarnessConfig(name="test"),
            ["provider.kwargs.startup_timeout=1200"],
        )

        value = dict(harness_config.provider.kwargs)["startup_timeout"]
        assert value == 1200
        assert isinstance(value, int)
