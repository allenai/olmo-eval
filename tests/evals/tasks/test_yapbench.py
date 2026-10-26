"""Tests for the YapBench task."""

import logging

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers import ScoringIncompleteError
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response, Split
from olmo_eval.data import DataLoader
from olmo_eval.evals.tasks import yapbench
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.yapbench import (
    YAP_RAW,
    YAP_STRIPPED,
    YAPBENCH_REPO,
    YAPBENCH_REVISION,
    CategoryMedianYapMetric,
    YapIndexMetric,
    YapScorer,
    strip_markdown,
    visible_answer,
)
from olmo_eval.runners.io.builders import build_predictions

ROWS = [
    {
        "id": "A-0001",
        "category": "A",
        "prompt": "",
        "baseline": "Please rephrase.",
        "baseline_type": "clarification",
        "domain": "nonsense",
        "notes": "Empty input",
    },
    {
        "id": "B-0001",
        "category": "B",
        "prompt": "What is the capital of France?",
        "baseline": "Paris",
        "baseline_type": "word",
        "domain": "geography",
        "notes": "Classic factual question with single correct answer",
    },
    {
        "id": "C-0001",
        "category": "C",
        "prompt": "Count the lines in data.txt from the shell.",
        "baseline": "wc -l < data.txt",
        "baseline_type": "command",
        "domain": "shell",
        "notes": "",
    },
]


@pytest.fixture
def task():
    return get_task("yapbench")


@pytest.fixture
def loaded_sources(monkeypatch):
    """Serve the fixture rows, recording each data source loaded."""
    sources = []

    def load(self, source, streaming=False):
        sources.append(source)
        return iter(ROWS)

    monkeypatch.setattr(DataLoader, "load", load)
    return sources


def _scored(yap: float | None, category: str, scorer=YAP_STRIPPED) -> Response:
    """A response whose YapScore has already been computed."""
    return Response(
        instance=Instance(question="q", metadata={"category": category}),
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text="a")],
        scores={} if yap is None else {scorer.name: yap},
    )


async def _yap(text: str, baseline: str = "Paris", strip: bool = True) -> float:
    instance = Instance(question="q", metadata={"baseline": baseline})
    return await YapScorer(strip=strip).ascore_with_context(
        instance, LMOutput(text=text), ScoringContext()
    )


class TestRegistration:
    def test_is_registered(self, task):
        assert task_exists("yapbench")
        assert task.request_type == RequestType.CHAT

    def test_data_source_is_pinned(self, task):
        source = task.config.get_data_source()

        assert task.config.split == Split.TRAIN
        assert (source.path, source.revision, source.split) == (
            YAPBENCH_REPO,
            YAPBENCH_REVISION,
            "train",
        )

    def test_sampling_is_greedy_without_a_token_limit(self, task):
        params = task.config.sampling_params

        assert (params.temperature, params.max_tokens) == (0.0, None)

    def test_metrics_for_both_length_measures(self, task):
        metrics = [(m.name, m.scorer().name) for m in task.config.metrics]
        names = ["yap_index", *(f"category__{c}__median_yap" for c in "ABC")]

        assert metrics == [
            *((name, "yap_stripped") for name in names),
            *((name, "yap_raw") for name in names),
        ]
        primary = task.config.get_primary_metric()
        assert (primary.name, primary.scorer().name) == ("yap_index", "yap_stripped")

    def test_every_metric_is_lower_is_better_in_characters(self, task):
        for metric in task.config.metrics:
            assert metric.higher_is_better() is False
            assert metric.display_format() == "raw"
            assert metric.supports_scorer_fallback() is False


class TestInstances:
    def test_rows_become_instances(self, task, loaded_sources):
        instances = list(task.instances)

        assert [i.question for i in instances] == [row["prompt"] for row in ROWS]
        assert instances[1].gold_answer == "Paris"
        assert instances[1].metadata == {
            "id": "B-0001",
            "category": "B",
            "baseline": "Paris",
            "baseline_type": "word",
            "domain": "geography",
        }
        assert loaded_sources[0].revision == YAPBENCH_REVISION

    @pytest.mark.parametrize("prompt", ["", " ", "\n\n", "\t\t"])
    def test_a_blank_prompt_is_sent_verbatim_with_no_system_prompt(self, task, prompt):
        request = task.format_request(task.process_doc({**ROWS[0], "prompt": prompt}))

        assert request.messages == ({"role": "user", "content": prompt},)
        assert request.system_prompt is None

    def test_an_unknown_category_is_rejected(self, task):
        with pytest.raises(ValueError, match="unknown category 'D'"):
            task.process_doc({**ROWS[0], "id": "D-0001", "category": "D"})


class TestAnswerLength:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The capital of France is **Paris**.", "The capital of France is Paris."),
            ("It is *Paris* or __Paris__ or _Paris_.", "It is Paris or Paris or Paris."),
            ("# Answer\nParis", "Answer\nParis"),
            ("- one\n* two\n+ three", "one\ntwo\nthree"),
            ("1. one\n2. two", "one\ntwo"),
            ("> quoted", "quoted"),
            ("See [the docs](https://example.com).", "See the docs."),
            ("![map](map.png)", "!map"),
            ("  **Paris**\n", "  Paris\n"),
            ("```bash\nwc -l < data.txt\n```", "wc -l < data.txt\n"),
            ("Use `wc -l`.", "Use wc -l."),
            ("Above\n---\nBelow", "Above\n\nBelow"),
            ("~~Lyon~~ Paris", "Lyon Paris"),
            ("snake_case_name stays", "snake_case_name stays"),
            ("2 * 3 * 4", "2  3  4"),
            ("a*b*c", "a*b*c"),
            ("", ""),
        ],
    )
    def test_markdown_is_stripped(self, text, expected):
        assert strip_markdown(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("  Paris \n", "  Paris \n"),
            ("<think>Maybe Lyon</think>\n\nParis\n", "Paris\n"),
            ("Maybe Lyon</think>Paris", "Paris"),
            ("<think>Maybe Lyon", None),
            ("<think>a</think> b <think>c", None),
            ("", ""),
        ],
    )
    def test_visible_answer(self, text, expected):
        assert visible_answer(text) == expected


class TestScorer:
    @pytest.mark.anyio
    async def test_yap_score_is_the_excess_over_the_baseline(self):
        assert await _yap("Paris") == 0.0
        assert await _yap("Paris.") == 1.0
        assert await _yap("Par") == 0.0
        assert await _yap("The capital of France is Paris.") == 26.0

    @pytest.mark.anyio
    async def test_raw_length_counts_markdown(self):
        text = "The capital of France is **Paris**."

        assert await _yap(text, strip=True) == 26.0
        assert await _yap(text, strip=False) == 30.0

    @pytest.mark.anyio
    async def test_characters_are_code_points(self):
        assert await _yap("H₂O, water", baseline="H₂O") == 7.0

    @pytest.mark.anyio
    async def test_reasoning_is_not_counted(self):
        assert await _yap("<think>" + "x" * 500 + "</think>Paris") == 0.0

    @pytest.mark.anyio
    async def test_an_unclosed_trace_cannot_be_measured(self):
        with pytest.raises(ScoringIncompleteError, match="never closed"):
            await _yap("<think>still deciding")

    @pytest.mark.anyio
    async def test_a_trace_opened_by_the_template_and_cut_off_cannot_be_measured(self):
        instance = Instance(question="q", metadata={"baseline": "Paris"})
        cut_off = LMOutput(text="Maybe Lyon", metadata={"finish_reason": "length"})
        finished = LMOutput(text="Maybe Lyon</think>Paris", metadata={"finish_reason": "length"})

        with pytest.raises(ScoringIncompleteError, match="never closed"):
            await YapScorer().ascore_with_context(instance, cut_off, ScoringContext())
        assert await YapScorer().ascore_with_context(instance, finished, ScoringContext()) == 0.0

    @pytest.mark.anyio
    async def test_whitespace_left_by_stripping_is_counted(self):
        assert await _yap("```\nParis\n```") == 1.0

    @pytest.mark.anyio
    @pytest.mark.parametrize("strip", [True, False])
    async def test_whitespace_around_the_answer_is_counted(self, strip):
        assert await _yap("\nParis\n", strip=strip) == 2.0
        assert await _yap("<think>Maybe Lyon</think>\n\nParis\n", strip=strip) == 1.0

    @pytest.mark.anyio
    async def test_stripped_length_matches_the_leaderboard_viewer(self):
        # The Space's viewer reports len(strip_markdown(response)) minus the baseline length
        assert await _yap("\n**Paris**.\n") == 3.0


class TestMetrics:
    @pytest.fixture
    def responses(self):
        return [
            _scored(10.0, "A"),
            _scored(20.0, "A"),
            _scored(90.0, "A"),
            _scored(0.0, "B"),
            _scored(4.0, "B"),
            _scored(None, "B"),
            _scored(30.0, "C"),
        ]

    def test_category_medians(self, responses):
        medians = [
            CategoryMedianYapMetric(name=f"category__{c}__median_yap").compute(responses)
            for c in "ABC"
        ]

        assert medians == [20.0, 2.0, 30.0]

    def test_yap_index_averages_the_category_medians(self, responses):
        assert YapIndexMetric().compute(responses) == pytest.approx((20 + 2 + 30) / 3)

    def test_metrics_read_their_own_scorer(self, responses):
        raw = [_scored(100.0, c, scorer=YAP_RAW) for c in "ABC"]

        assert YapIndexMetric(scorer=YAP_RAW).compute(responses + raw) == 100.0
        assert YapIndexMetric().compute(raw) == 0.0

    def test_an_empty_category_is_left_out_with_a_warning(self, caplog):
        responses = [_scored(10.0, "A"), _scored(20.0, "B")]

        with caplog.at_level(logging.WARNING, logger=yapbench.__name__):
            index = YapIndexMetric().compute(responses)
            median = CategoryMedianYapMetric(name="category__C__median_yap").compute(responses)

        assert (index, median) == (15.0, 0.0)
        assert len([r for r in caplog.records if r.name == yapbench.__name__]) == 2

    def test_per_instance_values(self):
        response = _scored(7.0, "B")
        b = CategoryMedianYapMetric(name="category__B__median_yap")
        a = CategoryMedianYapMetric(name="category__A__median_yap")

        assert (b.compute_instance(response), a.compute_instance(response)) == (7.0, None)
        assert YapIndexMetric().compute_instance(response) is None

    @pytest.mark.parametrize(
        "name", ["median_yap", "category__D__median_yap", "group__A__median_yap"]
    )
    def test_malformed_category_names_are_rejected(self, name):
        with pytest.raises(ValueError, match="category__<A|B|C>__<metric>"):
            CategoryMedianYapMetric(name=name)


class TestEndToEnd:
    @pytest.mark.anyio
    async def test_scores_and_aggregates_a_run(self, task, caplog):
        def response(row, text):
            instance = task.process_doc(row)
            return Response(
                instance=instance, request=task.format_request(instance), outputs=[LMOutput(text)]
            )

        responses = [
            response(ROWS[0], "Could you **rephrase** that?"),
            response(ROWS[1], "<think>France...</think>The capital of France is Paris."),
            response(ROWS[2], "<think>wc or awk"),
        ]

        with caplog.at_level(logging.WARNING):
            await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        assert responses[0].scores == {"yap_stripped": 8.0, "yap_raw": 12.0}
        assert responses[1].scores == {"yap_stripped": 26.0, "yap_raw": 26.0}
        assert responses[2].scores == {}
        assert responses[2].outputs[0].metadata["scoring_errors"]["yap_stripped"]
        assert metrics["yap_index"] == {"yap_stripped": 17.0, "yap_raw": 19.0}
        assert metrics["category__C__median_yap"] == {"yap_stripped": 0.0, "yap_raw": 0.0}

        predictions = build_predictions(responses, metrics=task.config.metrics)
        assert predictions[1]["model_output"][0]["extracted_answer"] == (
            "The capital of France is Paris."
        )
