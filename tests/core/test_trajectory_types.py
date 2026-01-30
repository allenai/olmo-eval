"""Tests for olmo_eval.core.trajectory_types module."""

from olmo_eval.core.types import AgentTrajectory, AgentTurn, ToolCall, ToolResult


class TestAgentTurn:
    """Tests for AgentTurn dataclass."""

    def test_create_assistant_turn(self):
        """Test creating an assistant turn."""
        turn = AgentTurn.assistant(content="Hello")
        assert turn.role == "assistant"
        assert turn.content == "Hello"
        assert turn.tool_calls == ()

    def test_create_assistant_with_tool_calls(self):
        """Test assistant turn with tool calls."""
        calls = [ToolCall.create("1", "search", {"query": "test"})]
        turn = AgentTurn.assistant(content="Let me search", tool_calls=calls)
        assert turn.has_tool_calls
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].function.name == "search"

    def test_create_tool_turn(self):
        """Test creating a tool result turn."""
        results = [ToolResult(tool_call_id="1", content="Search results")]
        turn = AgentTurn.tool(results=results)
        assert turn.role == "tool"
        assert turn.has_tool_results
        assert len(turn.tool_results) == 1

    def test_create_user_turn(self):
        """Test creating a user turn."""
        turn = AgentTurn.user(content="What's the weather?")
        assert turn.role == "user"
        assert turn.content == "What's the weather?"

    def test_has_tool_calls_false(self):
        """Test has_tool_calls when no calls."""
        turn = AgentTurn.assistant(content="Just text")
        assert not turn.has_tool_calls

    def test_has_tool_results_false(self):
        """Test has_tool_results when no results."""
        turn = AgentTurn.assistant(content="Just text")
        assert not turn.has_tool_results

    def test_timestamp_and_token_count(self):
        """Test optional timestamp and token count."""
        turn = AgentTurn.assistant(content="Hello", timestamp_ms=1234567890, token_count=10)
        assert turn.timestamp_ms == 1234567890
        assert turn.token_count == 10


class TestAgentTrajectory:
    """Tests for AgentTrajectory dataclass."""

    def test_empty_trajectory(self):
        """Test empty trajectory."""
        traj = AgentTrajectory()
        assert traj.num_turns == 0
        assert traj.total_tokens == 0
        assert traj.total_tool_calls == 0

    def test_from_turns(self):
        """Test creating from list of turns."""
        turns = [
            AgentTurn.user(content="Hello"),
            AgentTurn.assistant(content="Hi there"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.num_turns == 2

    def test_total_tokens(self):
        """Test token counting."""
        turns = [
            AgentTurn.assistant(content="Hello", token_count=5),
            AgentTurn.assistant(content="World", token_count=10),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.total_tokens == 15

    def test_total_tool_calls(self):
        """Test tool call counting."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "search", {}),
                    ToolCall.create("2", "calc", {}),
                ]
            ),
            AgentTurn.assistant(tool_calls=[ToolCall.create("3", "search", {})]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.total_tool_calls == 3

    def test_tool_call_sequence(self):
        """Test flattened tool call sequence."""
        turns = [
            AgentTurn.assistant(tool_calls=[ToolCall.create("1", "first", {})]),
            AgentTurn.assistant(tool_calls=[ToolCall.create("2", "second", {})]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        sequence = traj.tool_call_sequence
        assert len(sequence) == 2
        assert sequence[0].function.name == "first"
        assert sequence[1].function.name == "second"

    def test_unique_tools_used(self):
        """Test getting unique tools."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "search", {}),
                    ToolCall.create("2", "search", {}),
                    ToolCall.create("3", "calc", {}),
                ]
            ),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.unique_tools_used == {"search", "calc"}

    def test_tool_calls_by_name(self):
        """Test filtering tool calls by name."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "search", {"q": "a"}),
                    ToolCall.create("2", "calc", {}),
                    ToolCall.create("3", "search", {"q": "b"}),
                ]
            ),
        ]
        traj = AgentTrajectory.from_turns(turns)
        search_calls = traj.tool_calls_by_name("search")
        assert len(search_calls) == 2

    def test_tool_call_names(self):
        """Test getting tool names in order."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "a", {}),
                    ToolCall.create("2", "b", {}),
                ]
            ),
            AgentTurn.assistant(tool_calls=[ToolCall.create("3", "c", {})]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.tool_call_names() == ["a", "b", "c"]

    def test_assistant_turns(self):
        """Test getting assistant turns only."""
        turns = [
            AgentTurn.user(content="Hello"),
            AgentTurn.assistant(content="Hi"),
            AgentTurn.user(content="Bye"),
            AgentTurn.assistant(content="Goodbye"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assistant = traj.assistant_turns
        assert len(assistant) == 2
        assert all(t.role == "assistant" for t in assistant)

    def test_tool_turns(self):
        """Test getting tool turns only."""
        turns = [
            AgentTurn.assistant(tool_calls=[ToolCall.create("1", "test", {})]),
            AgentTurn.tool([ToolResult(tool_call_id="1", content="result")]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        tool_turns = traj.tool_turns
        assert len(tool_turns) == 1
        assert tool_turns[0].role == "tool"

    def test_get_turn(self):
        """Test getting turn by index."""
        turns = [
            AgentTurn.user(content="First"),
            AgentTurn.assistant(content="Second"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.get_turn(0).content == "First"
        assert traj.get_turn(1).content == "Second"
        assert traj.get_turn(99) is None
        assert traj.get_turn(-1) is None

    def test_with_final_answer(self):
        """Test setting final answer."""
        traj = AgentTrajectory()
        new_traj = traj.with_final_answer("The answer is 42")
        assert new_traj.final_answer == "The answer is 42"
        assert traj.final_answer is None  # Original unchanged

    def test_with_state(self):
        """Test setting state snapshot."""
        traj = AgentTrajectory()
        new_traj = traj.with_state({"count": 10})
        assert new_traj.state_snapshot == {"count": 10}
        assert traj.state_snapshot == {}  # Original unchanged

    def test_to_messages(self):
        """Test converting to OpenAI message format."""
        call = ToolCall.create("1", "search", {"q": "test"})
        result = ToolResult(tool_call_id="1", content="results")
        turns = [
            AgentTurn.user(content="Search for test"),
            AgentTurn.assistant(content="Let me search", tool_calls=[call]),
            AgentTurn.tool([result]),
            AgentTurn.assistant(content="I found results"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        messages = traj.to_messages()

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "assistant"


class TestAgentTrajectoryToolResults:
    """Tests for tool result handling in AgentTrajectory."""

    def test_tool_result_sequence(self):
        """Test getting tool results in order."""
        turns = [
            AgentTurn.tool([ToolResult(tool_call_id="1", content="first")]),
            AgentTurn.tool([ToolResult(tool_call_id="2", content="second")]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        results = traj.tool_result_sequence
        assert len(results) == 2
        assert results[0].content == "first"
        assert results[1].content == "second"
