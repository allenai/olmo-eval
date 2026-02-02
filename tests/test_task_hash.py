"""Tests for task hash consistency and prefix matching."""

from olmo_eval.runners.common import compute_task_hash


class TestTaskConfigToDict:
    """Tests for TaskConfig.to_dict() and hash consistency."""

    def test_task_config_to_dict_includes_required_fields(self):
        """TaskConfig.to_dict() should include all config fields."""
        from olmo_eval.evals.tasks.core.base import TaskConfig

        config = TaskConfig(
            name="test_task",
            num_fewshot=5,
            fewshot_seed=42,
            limit=100,
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "test_task"
        assert config_dict["num_fewshot"] == 5
        assert config_dict["fewshot_seed"] == 42
        assert config_dict["limit"] == 100
        assert "split" in config_dict
        assert "data_source" in config_dict
        assert "metrics" in config_dict

    def test_same_task_config_same_hash(self):
        """Same TaskConfig should always produce same hash."""
        from olmo_eval.evals.tasks.core.base import TaskConfig

        config1 = TaskConfig(
            name="humaneval",
            num_fewshot=0,
            limit=50,
        )

        config2 = TaskConfig(
            name="humaneval",
            num_fewshot=0,
            limit=50,
        )

        hash1 = compute_task_hash(config1.to_dict())
        hash2 = compute_task_hash(config2.to_dict())

        assert hash1 == hash2, "Same config should produce same hash"

    def test_different_limit_different_hash(self):
        """Different limit should produce different hash."""
        from olmo_eval.evals.tasks.core.base import TaskConfig

        config1 = TaskConfig(name="test", limit=50)
        config2 = TaskConfig(name="test", limit=100)

        hash1 = compute_task_hash(config1.to_dict())
        hash2 = compute_task_hash(config2.to_dict())

        assert hash1 != hash2, "Different limit should produce different hash"

    def test_different_num_fewshot_different_hash(self):
        """Different num_fewshot should produce different hash."""
        from olmo_eval.evals.tasks.core.base import TaskConfig

        config1 = TaskConfig(name="test", num_fewshot=0)
        config2 = TaskConfig(name="test", num_fewshot=5)

        hash1 = compute_task_hash(config1.to_dict())
        hash2 = compute_task_hash(config2.to_dict())

        assert hash1 != hash2, "Different num_fewshot should produce different hash"


class TestAgentTaskConfigToDict:
    """Tests for AgentTaskConfig.to_dict() and hash consistency."""

    def test_agent_task_config_to_dict_includes_agent_settings(self):
        """AgentTaskConfig.to_dict() should include agent-specific fields."""
        from olmo_eval.core.formatters import ChatFormatter
        from olmo_eval.evals.tasks.core.agent_task import AgentTaskConfig

        config = AgentTaskConfig(
            name="test_agent",
            max_turns=15,
            max_concurrency=8,
            formatter=ChatFormatter(system_prompt="Test prompt"),
        )

        config_dict = config.to_dict()

        assert "agent_settings" in config_dict
        assert config_dict["agent_settings"]["max_turns"] == 15
        assert config_dict["agent_settings"]["max_concurrency"] == 8
        assert config_dict["formatter"]["system_prompt"] == "Test prompt"

    def test_same_agent_config_same_hash(self):
        """Same AgentTaskConfig should always produce same hash."""
        from olmo_eval.evals.tasks.core.agent_task import AgentTaskConfig

        config1 = AgentTaskConfig(
            name="simpleqa_agent",
            max_turns=10,
            max_concurrency=4,
        )

        config2 = AgentTaskConfig(
            name="simpleqa_agent",
            max_turns=10,
            max_concurrency=4,
        )

        hash1 = compute_task_hash(config1.to_dict())
        hash2 = compute_task_hash(config2.to_dict())

        assert hash1 == hash2, "Same config should produce same hash"

    def test_different_max_turns_different_hash(self):
        """Different max_turns should produce different hash."""
        from olmo_eval.evals.tasks.core.agent_task import AgentTaskConfig

        config1 = AgentTaskConfig(name="test", max_turns=10)
        config2 = AgentTaskConfig(name="test", max_turns=20)

        hash1 = compute_task_hash(config1.to_dict())
        hash2 = compute_task_hash(config2.to_dict())

        assert hash1 != hash2, "Different max_turns should produce different hash"

    def test_config_does_not_include_runtime_fields(self):
        """Config dict should not include runtime fields like model_url."""
        from olmo_eval.evals.tasks.core.agent_task import AgentTaskConfig

        config = AgentTaskConfig(name="test", max_turns=10)
        config_dict = config.to_dict()

        # These runtime fields should NOT be in the config
        assert "model_url" not in config_dict.get("agent_settings", {})
        assert "model" not in config_dict.get("agent_settings", {})
