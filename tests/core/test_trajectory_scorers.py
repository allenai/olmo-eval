"""Tests for olmo_eval.core.trajectory_scorers module."""

from olmo_eval.common.scorers import (
    TrajectoryCombinedScorer,
    TrajectoryEfficiencyScorer,
    TrajectoryResponseScorer,
    TrajectoryStateScorer,
)
from olmo_eval.common.types import (
    AgentTrajectory,
    AgentTurn,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    ToolCall,
)


def make_response(
    instance: Instance,
    output: LMOutput | None = None,
    trajectory: AgentTrajectory | None = None,
) -> Response:
    """Helper to create Response objects."""
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT),
        outputs=[output] if output else [],
        trajectory=trajectory,
    )


class TestTrajectoryResponseScorer:
    """Tests for TrajectoryResponseScorer."""

    def test_subsequence_match(self):
        """Test subsequence matching."""
        scorer = TrajectoryResponseScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "c"}),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "a", {}),
                ToolCall.create("2", "b", {}),
                ToolCall.create("3", "c", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_exact_match_required(self):
        """Test exact match mode."""
        scorer = TrajectoryResponseScorer(require_exact_match=True)
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "b"}),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "a", {}),
                ToolCall.create("2", "b", {}),
                ToolCall.create("3", "c", {}),  # Extra call
            ],
        )

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_score_response_with_trajectory(self):
        """Test score_response using trajectory."""
        scorer = TrajectoryResponseScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "search"},),
        )
        trajectory = AgentTrajectory.from_turns(
            [AgentTurn.assistant(tool_calls=[ToolCall.create("1", "search", {})])]
        )
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 1.0

    def test_no_requirements(self):
        """Test when no required trajectory."""
        scorer = TrajectoryResponseScorer()
        instance = Instance(question="Task")
        output = LMOutput(text="Done")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_empty_required_trajectory(self):
        """Test with empty required trajectory."""
        scorer = TrajectoryResponseScorer()
        instance = Instance(question="Task", required_trajectory=())
        output = LMOutput(text="Done")

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = TrajectoryResponseScorer()
        assert scorer.name == "trajectory_response"


class TestTrajectoryStateScorer:
    """Tests for TrajectoryStateScorer."""

    def test_state_match(self):
        """Test state matching via score_response."""
        scorer = TrajectoryStateScorer()
        instance = Instance(
            question="Task",
            expected_final_state={"count": 10, "done": True},
        )
        trajectory = AgentTrajectory().with_state({"count": 10, "done": True})
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 1.0

    def test_state_mismatch(self):
        """Test state mismatch."""
        scorer = TrajectoryStateScorer()
        instance = Instance(
            question="Task",
            expected_final_state={"count": 10},
        )
        trajectory = AgentTrajectory().with_state({"count": 5})
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 0.0

    def test_custom_comparator(self):
        """Test with custom state comparator."""

        def loose_compare(expected, actual):
            return set(expected.keys()) == set(actual.keys())

        scorer = TrajectoryStateScorer(state_comparator=loose_compare)
        instance = Instance(
            question="Task",
            expected_final_state={"a": 1, "b": 2},
        )
        trajectory = AgentTrajectory().with_state({"a": 999, "b": 888})
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 1.0

    def test_no_trajectory(self):
        """Test when no trajectory available."""
        scorer = TrajectoryStateScorer()
        instance = Instance(
            question="Task",
            expected_final_state={"done": True},
        )
        response = make_response(instance)

        score = scorer.score_response(response)
        assert score == 0.0

    def test_score_fallback(self):
        """Test score() fallback returns 0."""
        scorer = TrajectoryStateScorer()
        instance = Instance(question="Task", expected_final_state={"done": True})
        output = LMOutput(text="Done")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = TrajectoryStateScorer()
        assert scorer.name == "trajectory_state"


class TestTrajectoryEfficiencyScorer:
    """Tests for TrajectoryEfficiencyScorer."""

    def test_perfect_efficiency(self):
        """Test perfect efficiency (minimal steps)."""
        scorer = TrajectoryEfficiencyScorer(minimal_steps=3)
        instance = Instance(question="Task")
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "a", {}),
                ToolCall.create("2", "b", {}),
                ToolCall.create("3", "c", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_low_efficiency(self):
        """Test low efficiency (many extra steps)."""
        scorer = TrajectoryEfficiencyScorer(minimal_steps=2)
        instance = Instance(question="Task")
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "a", {}),
                ToolCall.create("2", "b", {}),
                ToolCall.create("3", "c", {}),
                ToolCall.create("4", "d", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 0.5

    def test_uses_required_trajectory_length(self):
        """Test using required_trajectory for minimal steps."""
        scorer = TrajectoryEfficiencyScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "b"}),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "a", {}),
                ToolCall.create("2", "b", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_score_response_with_trajectory(self):
        """Test score_response using trajectory."""
        scorer = TrajectoryEfficiencyScorer(minimal_steps=2)
        instance = Instance(question="Task")
        trajectory = AgentTrajectory.from_turns(
            [
                AgentTurn.assistant(
                    tool_calls=[
                        ToolCall.create("1", "a", {}),
                        ToolCall.create("2", "b", {}),
                    ]
                ),
            ]
        )
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 1.0

    def test_no_tool_calls(self):
        """Test with no tool calls."""
        scorer = TrajectoryEfficiencyScorer()
        instance = Instance(question="Task")
        output = LMOutput(text="Done")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_capped_at_one(self):
        """Test efficiency is capped at 1.0."""
        scorer = TrajectoryEfficiencyScorer(minimal_steps=10)
        instance = Instance(question="Task")
        output = LMOutput(
            text="",
            tool_calls=[ToolCall.create("1", "a", {})],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = TrajectoryEfficiencyScorer()
        assert scorer.name == "trajectory_efficiency"


class TestTrajectoryCombinedScorer:
    """Tests for TrajectoryCombinedScorer."""

    def test_both_pass(self):
        """Test when both response and state pass."""
        scorer = TrajectoryCombinedScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "search"},),
            expected_final_state={"found": True},
        )
        trajectory = AgentTrajectory.from_turns(
            [AgentTurn.assistant(tool_calls=[ToolCall.create("1", "search", {})])]
        ).with_state({"found": True})
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 1.0

    def test_response_fails(self):
        """Test when response check fails."""
        scorer = TrajectoryCombinedScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "search"},),
            expected_final_state={"found": True},
        )
        trajectory = AgentTrajectory.from_turns(
            [AgentTurn.assistant(tool_calls=[ToolCall.create("1", "wrong", {})])]
        ).with_state({"found": True})
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 0.0

    def test_state_fails(self):
        """Test when state check fails."""
        scorer = TrajectoryCombinedScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "search"},),
            expected_final_state={"found": True},
        )
        trajectory = AgentTrajectory.from_turns(
            [AgentTurn.assistant(tool_calls=[ToolCall.create("1", "search", {})])]
        ).with_state({"found": False})
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 0.0

    def test_no_state_requirement(self):
        """Test when no state requirement (only response checked)."""
        scorer = TrajectoryCombinedScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "search"},),
        )
        trajectory = AgentTrajectory.from_turns(
            [AgentTurn.assistant(tool_calls=[ToolCall.create("1", "search", {})])]
        )
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 1.0

    def test_exact_sequence_mode(self):
        """Test with require_exact_sequence."""
        scorer = TrajectoryCombinedScorer(require_exact_sequence=True)
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "b"}),
        )
        trajectory = AgentTrajectory.from_turns(
            [
                AgentTurn.assistant(
                    tool_calls=[
                        ToolCall.create("1", "a", {}),
                        ToolCall.create("2", "b", {}),
                        ToolCall.create("3", "c", {}),  # Extra
                    ]
                )
            ]
        )
        response = make_response(instance, trajectory=trajectory)

        score = scorer.score_response(response)
        assert score == 0.0

    def test_score_fallback(self):
        """Test score() fallback returns 0."""
        scorer = TrajectoryCombinedScorer()
        instance = Instance(question="Task")
        output = LMOutput(text="Done")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = TrajectoryCombinedScorer()
        assert scorer.name == "trajectory_combined"
