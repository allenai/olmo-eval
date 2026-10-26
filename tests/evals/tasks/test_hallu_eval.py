"""Tests for the HaluEval tasks."""

import hashlib
import logging
from dataclasses import replace

import pytest
from click.testing import CliRunner

from olmo_eval.cli import tasks as tasks_command
from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import LMOutput, RequestType, Response, Split
from olmo_eval.data import DataLoader
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.constants.halueval import (
    DIALOGUE_INSTRUCTION,
    DIALOGUE_SYSTEM_PROMPT,
    QA_INSTRUCTION,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_INSTRUCTION,
    SUMMARIZATION_SYSTEM_PROMPT,
)
from olmo_eval.evals.tasks.hallu_eval import (
    HALUEVAL_REPO,
    HALUEVAL_REVISION,
    extract_halueval_judgement,
)
from olmo_eval.runners.io.builders import build_predictions
from olmo_eval.runners.processing.utils import compute_task_hash

SPECS = (
    "halueval_qa",
    "halueval_qa:knowledge",
    "halueval_dialogue",
    "halueval_dialogue:knowledge",
    "halueval_summarization",
)

QA_ROWS = [
    {
        "knowledge": "Arthur's Magazine (1844–1846) was an American literary periodical.First "
        "for Women is a woman's magazine published by Bauer Media Group in the USA.",
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "right_answer": "Arthur's Magazine",
        "hallucinated_answer": "First for Women was started first.",
    },
    {
        "knowledge": "The Oberoi Group is a hotel company with its head office in Delhi.",
        "question": "The Oberoi family is part of a hotel company that has a head office in "
        "what city?",
        "right_answer": "Delhi",
        "hallucinated_answer": "Mumbai",
    },
]
DIALOGUE_ROWS = [
    {
        "knowledge": "Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac",
        "dialogue_history": "[Human]: Do you like Iron Man [Assistant]: Sure do! [Human]: Did "
        "you know he also was in Zodiac? ",
        "right_response": "Yes, Robert Downey Jr. starred in Zodiac with Jake Gyllenhaal.",
        "hallucinated_response": "Yes, Robert Downey Jr. starred in Zodiac with Tom Hanks.",
    },
]
SUMMARIZATION_ROWS = [
    {
        "document": "The panther chameleon was found on Monday by a dog walker in Marl Park.",
        "right_summary": "A chameleon was found in a park by a dog walker.",
        "hallucinated_summary": "A chameleon was found in a Cardiff park by a police officer.",
    },
]
ROWS = {"qa": QA_ROWS, "dialogue": DIALOGUE_ROWS, "summarization": SUMMARIZATION_ROWS}


@pytest.fixture
def loaded_sources(monkeypatch):
    """Serve the fixture rows of the requested subset, recording each data source loaded."""
    sources = []

    def load(self, source, streaming=False):
        sources.append(source)
        return iter(ROWS[source.data_files.partition("/")[0]])

    monkeypatch.setattr(DataLoader, "load", load)
    return sources


def _scored_response(task, instance, text: str) -> Response:
    return Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text=text)],
    )


class TestRegistration:
    @pytest.mark.parametrize("spec", SPECS)
    def test_specs_are_registered(self, spec):
        assert task_exists(spec)
        assert get_task(spec).request_type == RequestType.CHAT

    def test_summarization_has_no_knowledge_variant(self):
        assert not task_exists("halueval_summarization:knowledge")

    def test_cli_lists_the_tasks_and_variants(self):
        result = CliRunner().invoke(tasks_command, ["-f", "halueval"])

        assert result.exit_code == 0
        for name in ("halueval_qa", "halueval_dialogue", "halueval_summarization", "knowledge"):
            assert name in result.output

    @pytest.mark.parametrize(
        "spec,subset",
        [
            ("halueval_qa", "qa"),
            ("halueval_dialogue", "dialogue"),
            ("halueval_summarization", "summarization"),
        ],
    )
    def test_data_file_is_pinned(self, spec, subset):
        config = get_task(spec).config
        source = config.get_data_source()

        assert config.split == Split.TRAIN
        assert source.path == HALUEVAL_REPO
        assert source.revision == HALUEVAL_REVISION
        assert source.data_files == f"{subset}/data-00000-of-00001.parquet"
        assert source.split == "train"

    def test_sampling_is_greedy_without_a_token_limit(self):
        params = get_task("halueval_qa").config.sampling_params

        assert params.temperature == 0.0
        assert params.max_tokens is None

    @pytest.mark.parametrize("spec", SPECS)
    def test_metrics(self, spec):
        config = get_task(spec).config
        metrics = {metric.name: metric for metric in config.metrics}

        assert set(metrics) == {
            "accuracy",
            "hallucinated_accuracy",
            "right_accuracy",
            "parse_failure_rate",
        }
        assert config.get_primary_metric().name == "accuracy"
        assert all(metric.display_format() == "percentage" for metric in metrics.values())
        assert not metrics["parse_failure_rate"].higher_is_better()
        assert metrics["right_accuracy"].higher_is_better()

    def test_every_spec_has_its_own_task_hash(self):
        hashes = {compute_task_hash(get_task(spec).config.to_dict()) for spec in SPECS}
        assert len(hashes) == len(SPECS)

    @pytest.mark.parametrize("spec", SPECS)
    def test_prompt_and_extraction_reach_the_task_config(self, spec):
        serialized = get_task(spec).config.to_dict()

        assert serialized["formatter"]["type"] == "HaluEvalFormatter"
        assert ("#Knowledge#" in serialized["formatter"]["user_template"]) == spec.endswith(
            ":knowledge"
        )
        assert serialized["answer_extractor"] == "extract_halueval_judgement"


class TestReferencePrompts:
    @pytest.mark.parametrize(
        "instruction,sha256",
        [
            (QA_INSTRUCTION, "6eaf9fa20795cff9e1f73f4b044cc632e3468312b38f3174ff7026e1202c501b"),
            (
                DIALOGUE_INSTRUCTION,
                "d9c3312948083f349f87df15f94d5725c10aad7c9c37108cf93996d528411469",
            ),
            (
                SUMMARIZATION_INSTRUCTION,
                "611129e0d60ebd0db63377b32fef3117a60db486b8ab2eb6b49f9dd1b4d92b4c",
            ),
        ],
    )
    def test_instructions_match_the_reference_files(self, instruction, sha256):
        assert hashlib.sha256(instruction.encode("utf-8")).hexdigest() == sha256

    @pytest.mark.parametrize(
        "system_prompt,sha256",
        [
            (
                QA_SYSTEM_PROMPT,
                "ecd0f1818a0830f0f85b3564578df53cf9d5ccaaa35241a82b8c45282be6ce2d",
            ),
            (
                DIALOGUE_SYSTEM_PROMPT,
                "2b96f7a9c59b75c7eea7248bfc0f56cb2fb0c8292dd43c1ebd3967e8e911481e",
            ),
            (
                SUMMARIZATION_SYSTEM_PROMPT,
                "191cc85edc9bfb9ee8d15c48f5dd723ecdf440174b8849efdd27a55021c5d51f",
            ),
        ],
    )
    def test_system_prompts_match_the_reference_chat_path(self, system_prompt, sha256):
        assert hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() == sha256

    @pytest.mark.parametrize(
        "spec,system_prompt,expected_user",
        [
            (
                "halueval_qa",
                QA_SYSTEM_PROMPT,
                lambda row, output: (
                    QA_INSTRUCTION
                    + "\n\n#Question#: "
                    + row["question"]
                    + "\n#Answer#: "
                    + output
                    + "\n#Your Judgement#: "
                ),
            ),
            (
                "halueval_dialogue",
                DIALOGUE_SYSTEM_PROMPT,
                lambda row, output: (
                    DIALOGUE_INSTRUCTION
                    + "\n\n#Dialogue History#: "
                    + row["dialogue_history"]
                    + "\n#Response#: "
                    + output
                    + "\n#Your Judgement#: "
                ),
            ),
            (
                "halueval_summarization",
                SUMMARIZATION_SYSTEM_PROMPT,
                lambda row, output: (
                    SUMMARIZATION_INSTRUCTION
                    + "\n\n#Document#: "
                    + row["document"]
                    + "\n#Summary#: "
                    + output
                    + "\n#Your Judgement#: "
                ),
            ),
        ],
    )
    def test_requests_match_the_reference_chat_messages(
        self, loaded_sources, spec, system_prompt, expected_user
    ):
        task = get_task(spec)
        right, hallucinated = list(task.instances)[:2]
        row = ROWS[spec.removeprefix("halueval_")][0]
        name = task.output_name

        for instance, output in (
            (right, row[f"right_{name}"]),
            (hallucinated, row[f"hallucinated_{name}"]),
        ):
            request = task.format_request(instance)
            assert request.messages == (
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": expected_user(row, output)},
            )
            assert request.system_prompt == system_prompt

    @pytest.mark.parametrize(
        "spec,system_prompt,instruction,fields",
        [
            (
                "halueval_qa:knowledge",
                QA_SYSTEM_PROMPT,
                QA_INSTRUCTION,
                lambda row: (
                    "#Question#: " + row["question"] + "\n#Answer#: " + row["hallucinated_answer"]
                ),
            ),
            (
                "halueval_dialogue:knowledge",
                DIALOGUE_SYSTEM_PROMPT,
                DIALOGUE_INSTRUCTION,
                lambda row: (
                    "#Dialogue History#: "
                    + row["dialogue_history"]
                    + "\n#Response#: "
                    + row["hallucinated_response"]
                ),
            ),
        ],
    )
    def test_knowledge_variants_put_the_knowledge_before_the_sample(
        self, loaded_sources, spec, system_prompt, instruction, fields
    ):
        task = get_task(spec)
        hallucinated = list(task.instances)[1]
        row = ROWS[spec.removeprefix("halueval_").removesuffix(":knowledge")][0]

        system, user = task.format_request(hallucinated).messages

        assert system == {"role": "system", "content": system_prompt}
        assert user["content"] == (
            instruction
            + "\n\n#Knowledge#: "
            + row["knowledge"]
            + "\n"
            + fields(row)
            + "\n#Your Judgement#: "
        )

    def test_values_with_braces_are_sent_verbatim(self):
        task = get_task("halueval_qa:knowledge")
        instance = task.process_doc(
            {"knowledge": "{k}", "question": "What is {x}?", "answer": "{0}", "hallucination": "no"}
        )

        user = task.format_request(instance).messages[-1]["content"]

        assert user.endswith(
            "#Knowledge#: {k}\n#Question#: What is {x}?\n#Answer#: {0}\n#Your Judgement#: "
        )

    def test_fewshot_examples_are_rejected(self):
        task = get_task("halueval_qa")
        instance = task.process_doc({"question": "q", "answer": "a", "hallucination": "no"})

        with pytest.raises(ValueError, match="num_fewshot=0"):
            task.config.formatter.format(instance, [instance])

    def test_a_fewshot_override_is_rejected(self):
        task = get_task("halueval_qa")
        task.config = replace(task.config, num_fewshot=3)
        instance = task.process_doc({"question": "q", "answer": "a", "hallucination": "no"})

        with pytest.raises(ValueError, match="num_fewshot=0"):
            task.format_request(instance)


class TestInstances:
    @pytest.mark.parametrize(
        "spec,subset,input_column,output_name",
        [
            ("halueval_qa", "qa", "question", "answer"),
            ("halueval_dialogue", "dialogue", "dialogue_history", "response"),
            ("halueval_summarization", "summarization", "document", "summary"),
        ],
    )
    def test_each_row_gives_its_right_then_its_hallucinated_output(
        self, loaded_sources, spec, subset, input_column, output_name
    ):
        instances = list(get_task(spec).instances)
        rows = ROWS[subset]

        assert len(instances) == 2 * len(rows)
        for index, row in enumerate(rows):
            right, hallucinated = instances[2 * index : 2 * index + 2]
            assert right.question == hallucinated.question == row[input_column]
            assert right.gold_answer == "No"
            assert hallucinated.gold_answer == "Yes"
            assert right.metadata["fields"][output_name] == row[f"right_{output_name}"]
            assert (
                hallucinated.metadata["fields"][output_name] == row[f"hallucinated_{output_name}"]
            )
            assert right.metadata["fields"].get("knowledge") == row.get("knowledge")
            assert (right.metadata["index"], hallucinated.metadata["index"]) == (index, index)
            assert (right.metadata["hallucinated"], hallucinated.metadata["hallucinated"]) == (
                False,
                True,
            )

    def test_instance_ids_are_unique(self, loaded_sources):
        ids = [instance.metadata["id"] for instance in get_task("halueval_qa").instances]

        assert ids == ["0-right", "0-hallucinated", "1-right", "1-hallucinated"]

    def test_loads_the_pinned_data_file_once(self, loaded_sources):
        task = get_task("halueval_dialogue:knowledge")
        list(task.instances)
        list(task.instances)

        assert len(loaded_sources) == 1
        assert loaded_sources[0].data_files == "dialogue/data-00000-of-00001.parquet"
        assert loaded_sources[0].revision == HALUEVAL_REVISION

    def test_an_unknown_hallucination_label_is_rejected(self):
        task = get_task("halueval_qa")

        with pytest.raises(ValueError, match="unknown hallucination label 'maybe'"):
            task.process_doc({"question": "q", "answer": "a", "hallucination": "maybe"}, 4)


class TestJudgementExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Yes", "Yes"),
            ("No", "No"),
            ("Yes.", "Yes"),
            ("  No.\n", "No"),
            ("Yes, the answer is fabricated.", "Yes"),
            ("#Your Judgement#: No", "No"),
            ("<think>Could it be Yes? No...</think>\n\nYes", "Yes"),
            # Substring matching is kept from the reference
            ("Nolan directed it", "No"),
            ("The answer does Not hold", "No"),
        ],
    )
    def test_judgements_are_read(self, text, expected):
        assert extract_halueval_judgement(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "yes",
            "no",
            "I cannot tell.",
            "Yes and No",
            "Yes. Not every detail checks out.",
            "<think>The answer is Yes",
            "<think>Yes</think>\n\n",
        ],
    )
    def test_no_judgement(self, text):
        assert extract_halueval_judgement(text) == ""


class TestEndToEnd:
    @pytest.mark.anyio
    async def test_scores_and_aggregates_a_run(self, loaded_sources):
        task = get_task("halueval_qa")
        right_0, hallucinated_0, right_1, hallucinated_1 = task.instances
        responses = [
            _scored_response(task, right_0, "No"),
            _scored_response(task, hallucinated_0, "<think>No? Yes.</think>\n\nYes."),
            _scored_response(task, right_1, "Yes"),
            _scored_response(task, hallucinated_1, "Yes or No, hard to say"),
        ]

        await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        assert [r.outputs[0].extracted_answer for r in responses] == ["No", "Yes", "Yes", ""]
        assert [r.scores["exact_match"] for r in responses] == [1.0, 1.0, 0.0, 0.0]
        assert metrics["accuracy"]["exact_match"] == pytest.approx(0.5)
        assert metrics["hallucinated_accuracy"]["exact_match"] == pytest.approx(0.5)
        assert metrics["right_accuracy"]["exact_match"] == pytest.approx(0.5)
        assert metrics["parse_failure_rate"]["exact_match"] == pytest.approx(0.25)

    @pytest.mark.anyio
    async def test_judgements_and_per_instance_metrics_reach_the_predictions(self, loaded_sources):
        task = get_task("halueval_qa")
        right, hallucinated = list(task.instances)[:2]
        responses = [
            _scored_response(task, right, "No"),
            _scored_response(task, hallucinated, "I cannot tell."),
        ]

        await task.score_responses(responses, ScoringContext())
        predictions = build_predictions(responses, metrics=task.config.metrics)

        assert [p["native_id"] for p in predictions] == ["0-right", "0-hallucinated"]
        assert [p["label"] for p in predictions] == ["No", "Yes"]
        assert [p["model_output"][0]["extracted_answer"] for p in predictions] == ["No", ""]
        right_metrics, hallucinated_metrics = (p["instance_metrics"] for p in predictions)
        assert right_metrics["parse_failure_rate"]["exact_match"] == 0.0
        assert hallucinated_metrics["parse_failure_rate"]["exact_match"] == 1.0
        assert right_metrics["right_accuracy"]["exact_match"] == 1.0
        assert "hallucinated_accuracy" not in right_metrics
        assert hallucinated_metrics["hallucinated_accuracy"]["exact_match"] == 0.0
        assert "right_accuracy" not in hallucinated_metrics

    @pytest.mark.anyio
    async def test_a_label_with_no_instances_scores_zero_with_a_warning(
        self, loaded_sources, caplog
    ):
        task = get_task("halueval_summarization")
        right = next(iter(task.instances))
        responses = [_scored_response(task, right, "No")]

        await task.score_responses(responses, ScoringContext())
        with caplog.at_level(logging.WARNING):
            metrics = task.compute_metrics(responses)

        assert metrics["right_accuracy"]["exact_match"] == 1.0
        assert metrics["hallucinated_accuracy"]["exact_match"] == 0.0
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "hallucinated_accuracy" in warnings[0]
