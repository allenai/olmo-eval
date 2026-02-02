"""Integration tests for SimpleQA agent task."""

import pytest

from olmo_eval.core.types import SEARCH_TOOLS, Instance
from olmo_eval.evals.tasks import get_task, list_tasks
from olmo_eval.evals.tasks.core import AgentTask


class TestSimpleQAAgentRegistration:
    """Tests for simpleqa_agent task registration."""

    def test_simpleqa_agent_registered(self):
        """Test that simpleqa_agent is registered."""
        assert "simpleqa_agent" in list_tasks()

    def test_get_simpleqa_agent(self):
        """Test getting simpleqa_agent task."""
        task = get_task("simpleqa_agent")
        assert task.config.name == "simpleqa_agent"

    def test_simpleqa_agent_is_agent_task(self):
        """Test that simpleqa_agent is an AgentTask."""
        task = get_task("simpleqa_agent")
        assert isinstance(task, AgentTask)

    def test_simpleqa_agent_config(self):
        """Test simpleqa_agent configuration."""
        task = get_task("simpleqa_agent")
        assert task.config.max_turns == 10
        assert task.config.max_concurrency == 1
        assert task.system_prompt is not None
        assert "helpful assistant" in task.system_prompt

    def test_simpleqa_agent_required_secrets(self):
        """Test that simpleqa_agent has required secrets configured."""
        task = get_task("simpleqa_agent")
        assert "OPENAI_API_KEY" in task.config.required_secrets
        assert "S2_API_KEY" in task.config.required_secrets
        assert "SERPER_API_KEY" in task.config.required_secrets

    def test_simpleqa_agent_tools_in_config(self):
        """Test that simpleqa_agent has tools configured."""
        task = get_task("simpleqa_agent")
        assert task.config.tools is not None
        assert len(task.config.tools) == 3
        tool_names = [t.name for t in task.config.tools]
        assert "semantic_scholar_snippet_search" in tool_names
        assert "serper_google_webpage_search" in tool_names
        assert "serper_fetch_webpage_content" in tool_names


class TestSimpleQAAgentProcessDoc:
    """Tests for simpleqa_agent process_doc method."""

    @pytest.fixture
    def task(self):
        """Create a simpleqa_agent task."""
        return get_task("simpleqa_agent")

    def test_process_doc(self, task):
        """Test the process_doc method directly."""
        doc = {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "id": "test_123",
        }
        instance = task.process_doc(doc, index=0)

        assert instance is not None
        assert instance.question == "What is the capital of France?"
        assert instance.gold_answer == "Paris"
        assert instance.metadata["id"] == "test_123"
        assert instance.metadata["index"] == 0
        assert instance.tools == SEARCH_TOOLS

    def test_process_doc_alternative_fields(self, task):
        """Test process_doc with alternative field names."""
        doc = {
            "problem": "What is 2+2?",
            "ground_truth": "4",
        }
        instance = task.process_doc(doc, index=5)

        assert instance is not None
        assert instance.question == "What is 2+2?"
        assert instance.gold_answer == "4"
        assert instance.metadata["index"] == 5

    def test_process_doc_messages_format(self, task):
        """Test process_doc with messages format (actual dataset format)."""
        doc = {
            "messages": [{"role": "user", "content": "Who won the 2020 election?"}],
            "ground_truth": "Joe Biden",
        }
        instance = task.process_doc(doc, index=10)

        assert instance is not None
        assert instance.question == "Who won the 2020 election?"
        assert instance.gold_answer == "Joe Biden"
        assert instance.metadata["index"] == 10
        assert instance.tools == SEARCH_TOOLS

    def test_process_doc_empty_question(self, task):
        """Test that process_doc returns None for empty questions."""
        doc = {"question": "", "answer": "something"}
        instance = task.process_doc(doc, index=0)
        assert instance is None

    def test_process_doc_empty_messages(self, task):
        """Test that process_doc returns None for empty messages."""
        doc = {"messages": [], "ground_truth": "something"}
        instance = task.process_doc(doc, index=0)
        assert instance is None


def _can_load_simpleqa_dataset() -> bool:
    """Check if we can load instances from the simpleqa dataset."""
    try:
        task = get_task("simpleqa_agent")
        instances = list(task.instances)
        return len(instances) > 0
    except Exception:
        return False


@pytest.mark.skipif(
    not _can_load_simpleqa_dataset(),
    reason="Cannot load instances from allenai/simpleqa_full dataset",
)
class TestSimpleQAAgentInstances:
    """Integration tests for simpleqa_agent instance loading from actual dataset."""

    @pytest.fixture
    def task(self):
        """Create a simpleqa_agent task."""
        return get_task("simpleqa_agent")

    def test_can_load_instances(self, task):
        """Test that instances can be loaded from the dataset."""
        instances = list(task.instances)
        assert len(instances) > 0, "Should load at least one instance from dataset"

    def test_instance_structure(self, task):
        """Test that instances have the correct structure."""
        instances = list(task.instances)
        instance = instances[0]

        assert isinstance(instance, Instance)
        assert instance.question, "Instance should have a question"
        assert instance.gold_answer, "Instance should have a gold_answer"
        assert instance.tools == SEARCH_TOOLS, "Instance should have search tools"

    def test_instance_metadata(self, task):
        """Test that instances have correct metadata."""
        instances = list(task.instances)
        instance = instances[0]

        assert "id" in instance.metadata
        assert "index" in instance.metadata
        assert instance.metadata["dataset"] == "simpleqa"

    def test_instance_tools(self, task):
        """Test that instances have the correct tools configured."""
        instances = list(task.instances)
        instance = instances[0]

        assert instance.tools is not None
        assert len(instance.tools) > 0
        # Tools are ToolSchema objects with a name attribute
        tool_names = [t.name for t in instance.tools]
        assert any("semantic_scholar" in name or "serper" in name for name in tool_names)

    def test_instances_cached(self, task):
        """Test that instances are cached after first load."""
        instances1 = list(task.instances)
        instances2 = list(task.instances)

        assert instances1 == instances2
        assert task._instances_cache is not None
