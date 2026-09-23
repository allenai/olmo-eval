"""Tests for APTBench task registration, answer extraction, and scoring."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import aptbench
from olmo_eval.evals.tasks.aptbench import (
    APTBENCH_CONTEXT_BUDGETS,
    APTBENCH_SUBTASKS,
    LabelSetMatchScorer,
    aptbench_task_names,
    extract_bracketed,
    extract_choice_a_to_d,
    extract_choice_a_to_e,
    extract_correct_wrong,
    extract_first_command,
    extract_letter,
    extract_parenthesized,
    split_template,
)
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


_ALL_TASKS = aptbench_task_names()


def _instance(gold: str) -> Instance:
    return Instance(question="q", gold_answer=gold, metadata={"id": "x"})


def _output(task, text: str) -> LMOutput:
    output = LMOutput(text=text)
    output.extracted_answer = task.extract_answer(output)
    return output


def _response(task, text: str, gold: str) -> Response:
    instance = _instance(gold)
    output = _output(task, text)
    scorer = task.metrics[0].scorer
    response = Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="p"),
        outputs=[output],
    )
    response.scores[scorer.name] = scorer.score(instance, output)
    return response


class TestAPTBenchRegistration:
    """Tests for APTBench task registration."""

    def test_all_subtasks_registered(self):
        assert len(_ALL_TASKS) == 24
        registered = set(list_tasks())
        assert set(_ALL_TASKS) <= registered

    @pytest.mark.parametrize("category", ["env_setup", "issue_fix", "deepresearch", "tool"])
    def test_category_filter(self, category):
        names = aptbench_task_names(category)
        assert names
        assert all(name.startswith(f"aptbench_{category}_") for name in names)

    @pytest.mark.parametrize("subtask", APTBENCH_SUBTASKS, ids=lambda s: s.task_name)
    def test_sampling_params(self, subtask):
        params = get_task(subtask.task_name).config.sampling_params
        assert params.max_tokens == subtask.max_tokens
        assert params.temperature == 0.0
        assert params.stop_sequences is None
        assert params.truncate_prompt_tokens is None

    @pytest.mark.parametrize("budget", APTBENCH_CONTEXT_BUDGETS)
    def test_context_variants(self, budget):
        base = get_task("aptbench_issue_fix_plan").config.sampling_params
        variant = get_task(f"aptbench_issue_fix_plan:ctx{budget}k").config.sampling_params
        assert variant.truncate_prompt_tokens == budget * 1024 - 64
        assert variant.max_tokens == base.max_tokens
        assert variant.temperature == base.temperature

    def test_citation_tasks_use_label_set_scorer(self):
        for subtask in APTBENCH_SUBTASKS:
            scorer = get_task(subtask.task_name).metrics[0].scorer
            assert isinstance(scorer, LabelSetMatchScorer) == subtask.set_match
            assert subtask.set_match == ("citation" in subtask.name)


class TestExtractors:
    """Tests for the ported upstream answer extractors."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("B) foo", "B"),
            ("**C**)", "C"),
            ("d\n", "d"),
            ("xB)", None),
            ("B", None),
            ("", None),
        ],
    )
    def test_extract_letter(self, text, expected):
        assert extract_letter(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("e) yes", "E"), ("b: ok", "B"), ("answer c", "C"), ("F)", None)],
    )
    def test_extract_choice_a_to_e(self, text, expected):
        assert extract_choice_a_to_e(text) == expected

    @pytest.mark.parametrize(("text", "expected"), [("d)", "D"), ("E)", None)])
    def test_extract_choice_a_to_d(self, text, expected):
        assert extract_choice_a_to_d(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("CORRECT.", "correct"), ("it is wrong", "wrong"), ("incorrect", None)],
    )
    def test_extract_correct_wrong(self, text, expected):
        assert extract_correct_wrong(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("pip install x; ls\nnext", "pip install x"),
            ("ls\n", "ls"),
            # Upstream only cuts at ";" when the output spans several lines.
            ("pip install x; ls", "pip install x; ls"),
        ],
    )
    def test_extract_first_command(self, text, expected):
        assert extract_first_command(text) == expected

    def test_extract_bracketed(self):
        assert extract_bracketed("Paris] more") == "Paris"
        assert extract_bracketed("Paris") is None

    def test_extract_parenthesized(self):
        assert extract_parenthesized("A,C) rest") == "A,C"
        assert extract_parenthesized("A,C") is None


class TestExtractAnswer:
    """Tests for APTBenchTask.extract_answer."""

    def test_strips_output_and_prediction(self):
        task = get_task("aptbench_env_setup_action")
        assert _output(task, "  pip install -e . \nls").extracted_answer == "pip install -e ."

    def test_none_when_no_match(self):
        task = get_task("aptbench_issue_fix_plan")
        assert _output(task, "no letter here").extracted_answer is None


class TestLabelSetMatchScorer:
    """Tests for citation label-set scoring."""

    @pytest.mark.parametrize(
        ("pred", "gold", "expected"),
        [
            ("A,C", "A,C", 1.0),
            ("C,A", "A,C", 1.0),
            ("A,C,C", "A,C", 1.0),
            ("A", "A,C", 0.0),
            ("A,B,C", "A,C", 0.0),
            ("A, C", "A,C", 0.0),
            ("a,c", "A,C", 0.0),
            (None, "A,C", 0.0),
            ("", "A,C", 0.0),
        ],
    )
    def test_score(self, pred, gold, expected):
        output = LMOutput(text="", extracted_answer=pred)
        assert LabelSetMatchScorer().score(_instance(gold), output) == expected

    def test_missing_gold(self):
        output = LMOutput(text="", extracted_answer="A")
        instance = Instance(question="q", gold_answer=None, metadata={"id": "x"})
        assert LabelSetMatchScorer().score(instance, output) == 0.0


class TestExactMatchScoring:
    """Tests for case-sensitive exact-match scoring."""

    def test_case_sensitive(self):
        task = get_task("aptbench_issue_fix_plan")
        scorer = task.metrics[0].scorer
        assert scorer.score(_instance("B"), _output(task, "B)")) == 1.0
        assert scorer.score(_instance("B"), _output(task, "b)")) == 0.0


class TestNonEmptyAccuracyMetric:
    """Tests that empty generations are excluded, as upstream does."""

    def test_empty_generations_excluded(self):
        task = get_task("aptbench_issue_fix_plan")
        metric = task.metrics[0]
        responses = [
            _response(task, "B)", "B"),
            _response(task, "A)", "B"),
            _response(task, "", "B"),
            _response(task, " \n", "B"),
        ]
        assert metric.compute(responses) == pytest.approx(1 / 3)
        assert [metric.compute_instance(r) for r in responses] == [1.0, 0.0, None, 0.0]

    def test_all_empty(self):
        task = get_task("aptbench_issue_fix_plan")
        assert task.metrics[0].compute([_response(task, "", "B")]) == 0.0


class TestSplitTemplate:
    """Tests for splitting templates into fixed prefix and instance section."""

    def test_split_at_first_placeholder_paragraph(self):
        template = "Example 1\n\nExample 2\n\nIssue:\n$ISSUE$\n\nChoices:\n$CHOICES$\nAnswer: ("
        prefix, instance = split_template(template, ("$ISSUE$", "$CHOICES$"))
        assert prefix == "Example 1\n\nExample 2\n\n"
        assert instance == "Issue:\n$ISSUE$\n\nChoices:\n$CHOICES$\nAnswer: ("
        assert prefix + instance == template

    def test_placeholder_in_first_paragraph(self):
        prefix, instance = split_template("Q: $QUERY$\nA:", ("$QUERY$",))
        assert prefix == ""
        assert instance == "Q: $QUERY$\nA:"

    def test_no_placeholder_raises(self):
        with pytest.raises(ValueError):
            split_template("no placeholders", ("$ISSUE$",))


class TestPromptConstruction:
    """Tests for process_doc and format_request using a stub template."""

    _TEMPLATE = "Shot 1\n\nShot 2\n\nIssue:\n$ISSUE$\nChoices:\n$CHOICES$\nThe answer is ("

    @pytest.fixture
    def task(self, monkeypatch):
        monkeypatch.setattr(aptbench, "_fetch_text", lambda _path: self._TEMPLATE)
        return get_task("aptbench_issue_fix_locate")

    def test_prompt_matches_upstream_fill(self, task):
        doc = {"issue_statement": "  crash on start \n", "choices": "(A) x\n(B) y\n", "answer": "B"}
        instance = task.process_doc(doc, index=7)
        assert instance is not None

        expected = self._TEMPLATE
        for placeholder, key in (("$ISSUE$", "issue_statement"), ("$CHOICES$", "choices")):
            expected = expected.replace(placeholder, doc[key].strip())

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert request.prompt == expected
        assert "Shot 1" not in instance.question
        assert instance.gold_answer == "B"
        assert instance.metadata["id"] == "aptbench_issue_fix_locate_7"

    def test_list_gold_joined(self, monkeypatch):
        template = "Shot\n\n$ARTICLE$ $CHOICES$ $WEB_PAGE$ ("
        monkeypatch.setattr(aptbench, "_fetch_text", lambda _path: template)
        task = get_task("aptbench_deepresearch_openend_citation_en")
        doc = {"article": "a", "choices": "c", "url_content": "w", "answer": ["A", "C"]}
        instance = task.process_doc(doc)
        assert instance is not None
        assert instance.gold_answer == "A,C"
