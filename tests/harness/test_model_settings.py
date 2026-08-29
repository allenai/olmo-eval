"""Tests for pinning request-level model settings on a harness run.

gpt-5.6-sol rejects function tools and a reasoning effort together on
/v1/chat/completions, so a run with tools has to be able to turn reasoning off.
Nothing on the branch could express that: an env var carried by two weeks of
recorded runs read nothing, and worked only because the server default agreed.
These pin the mechanism that replaced it, end to end -- the dict an operator
types, the ModelSettings it becomes, and the Agent it reaches.
"""

import pytest

from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.presets import get_harness_preset
from olmo_eval.harness.scaffolds.openai_agents import (
    OpenAIAgentsScaffold,
    build_model_settings,
)

pytest.importorskip("agents")


class _ProviderStub:
    """Just enough provider for agent construction; makes no network calls."""

    model_name = "gpt-5.6-sol"

    def get_openai_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key="test-key", base_url="http://localhost:1/v1")


def _agent_for(scaffold_kwargs: dict) -> object:
    config = HarnessConfig(
        name="test",
        scaffold="openai_agents",
        scaffold_kwargs=scaffold_kwargs,
    )
    return OpenAIAgentsScaffold()._create_agent(_ProviderStub(), config)


class TestScaffoldRegistration:
    def test_the_scaffold_is_still_what_openai_agents_resolves_to(self):
        # Adding a module-level helper immediately above the class put it
        # between @register_scaffold and the class it decorates, so the registry
        # returned the helper and every worker died calling it with no
        # arguments. Direct imports of both names still worked, which is why
        # only a real run caught it.
        from olmo_eval.harness.scaffolds import get_scaffold

        assert isinstance(get_scaffold("openai_agents"), OpenAIAgentsScaffold)


class TestBuildModelSettings:
    @pytest.mark.parametrize("spec", [None, {}])
    def test_nothing_configured_yields_nothing(self, spec):
        # Returning an empty ModelSettings would replace the SDK's defaults with
        # this function's idea of them.
        assert build_model_settings(spec) is None

    def test_reasoning_effort_becomes_the_sdk_reasoning_object(self):
        settings = build_model_settings({"reasoning_effort": "none"})

        assert settings.reasoning.effort == "none"

    def test_reasoning_effort_does_not_go_through_extra_args(self):
        # OpenAIChatCompletionsModel splats extra_args into the same call that
        # already passes reasoning_effort= explicitly; the two would collide.
        settings = build_model_settings({"reasoning_effort": "none"})

        assert not settings.extra_args
        assert not settings.extra_body

    def test_other_model_settings_fields_pass_through(self):
        settings = build_model_settings({"temperature": 0.0, "max_tokens": 512})

        assert settings.temperature == 0.0
        assert settings.max_tokens == 512

    def test_an_unknown_key_raises_rather_than_being_dropped(self):
        # The failure this whole mechanism replaced was a knob that read nothing.
        with pytest.raises(ValueError, match="Unknown model_settings key"):
            build_model_settings({"reasoning_efort": "none"})

    def test_the_error_names_the_valid_keys(self):
        with pytest.raises(ValueError, match="reasoning_effort"):
            build_model_settings({"nonsense": 1})


class TestAgentConstruction:
    def test_the_setting_reaches_the_agent(self):
        agent = _agent_for({"model_settings": {"reasoning_effort": "none"}})

        assert agent.model_settings.reasoning.effort == "none"

    def test_absent_setting_leaves_the_sdk_defaults(self):
        agent = _agent_for({})

        assert agent.model_settings.reasoning is None

    def test_an_unrelated_scaffold_kwarg_is_not_treated_as_settings(self):
        agent = _agent_for({"enable_compaction": False})

        assert agent.model_settings.reasoning is None


class TestOverridePath:
    """The -o syntax an operator actually types has to reach scaffold_kwargs."""

    def _apply(self, overrides: list[str]) -> HarnessConfig:
        from olmo_eval.cli.beaker.launch import _apply_harness_overrides

        return _apply_harness_overrides(get_harness_preset("arxiv_paper_search_agent"), overrides)

    def test_nested_override_reaches_scaffold_kwargs(self):
        config = self._apply(["scaffold_kwargs.model_settings.reasoning_effort=none"])

        assert config.scaffold_kwargs["model_settings"] == {"reasoning_effort": "none"}

    def test_the_literal_none_stays_a_string(self):
        # "none" is a valid reasoning effort; coercing it to Python None would
        # silently mean "unset" and put the run straight back into the 400.
        config = self._apply(["scaffold_kwargs.model_settings.reasoning_effort=none"])

        assert config.scaffold_kwargs["model_settings"]["reasoning_effort"] == "none"

    def test_the_override_survives_into_a_usable_setting(self):
        config = self._apply(["scaffold_kwargs.model_settings.reasoning_effort=none"])

        settings = build_model_settings(config.scaffold_kwargs["model_settings"])

        assert settings.reasoning.effort == "none"

    def test_the_preset_does_not_pin_reasoning_itself(self):
        # Other backbones on this preset must keep their reasoning.
        preset = get_harness_preset("arxiv_paper_search_agent")

        assert "model_settings" not in preset.scaffold_kwargs

    def test_the_preset_documents_the_flag(self):
        docstring = get_harness_preset.__doc__ or ""
        from olmo_eval.harness.presets import HarnessPresets

        preset_doc = HarnessPresets.__dict__["arxiv_paper_search_agent"]._factory.__doc__ or ""

        assert "scaffold_kwargs.model_settings.reasoning_effort=none" in preset_doc + docstring
