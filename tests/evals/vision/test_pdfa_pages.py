"""The PDFA pages tasks: instances and requests from the examples file, and scoring."""

from __future__ import annotations

import json

import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks import pdfa_pages
from olmo_eval.evals.vision.scoring.pdfa_pages import (
    RESULT_KEY,
    PageNumberScorer,
    PageOrderScorer,
    TranscriptionScorer,
    pair_accuracy,
    parse_order,
    parse_page_number,
    transcription_scores,
)

ROWS = [
    {
        "id": "pdfa_find_page-0000",
        "source": "pdfa_find_page",
        "task": "find_page",
        "key": "0000001",
        "shard": 7,
        "pdf_off": 512,
        "pdf_size": 2048,
        "pages": [4, 5, 6],
        "image_labels": ["Page 5:", "Page 6:", "Page 7:"],
        "target_page": 5,
        "prompt": 'Question: Which page? "Annual report 6 of the board"\nAnswer:',
        "answer": "Page 6",
        "render_longest_dim": 1536,
    },
    {
        "id": "pdfa_find_page_long-0000",
        "source": "pdfa_find_page_long",
        "task": "find_page",
        "key": "0000002",
        "shard": 7,
        "pdf_off": 4096,
        "pdf_size": 1024,
        "pages": [10, 11],
        "image_labels": ["Page 11:", "Page 12:"],
        "target_page": 10,
        "prompt": 'Question: Which page? "Revenue grew in 2011 and 111 stores"\nAnswer:',
        "answer": "Page 11",
        "render_longest_dim": 1536,
    },
    {
        "id": "pdfa_page_order-0000",
        "source": "pdfa_page_order",
        "task": "page_order",
        "key": "0000001",
        "shard": 7,
        "pdf_off": 512,
        "pdf_size": 2048,
        "pages": [2, 0, 1],
        "image_labels": ["Image 1", "Image 2", "Image 3"],
        "target_page": None,
        "prompt": "Question: The pages of this document were shuffled.\nAnswer:",
        "answer": "2, 3, 1",
        "render_longest_dim": 1536,
    },
    {
        "id": "pdfa_page_ocr-0000",
        "source": "pdfa_page_ocr",
        "task": "page_ocr",
        "key": "0000003",
        "shard": 12,
        "pdf_off": 0,
        "pdf_size": 100,
        "pages": [3],
        "image_labels": None,
        "target_page": 3,
        "prompt": "Transcribe this page to Markdown.",
        "answer": "# Title\n\nBody text.",
        "render_longest_dim": 1536,
    },
]


@pytest.fixture
def build(tmp_path, monkeypatch):
    (tmp_path / "heldout").mkdir()
    with open(tmp_path / "heldout" / "examples.jsonl", "w") as f:
        for row in ROWS:
            f.write(json.dumps(row) + "\n")
    monkeypatch.setenv("PDFA_PAGES_DIR", str(tmp_path))
    return tmp_path


def _response(instance: Instance, text: str) -> Response:
    request = LMRequest(request_type=RequestType.CHAT, prompt=instance.question)
    return Response(instance=instance, request=request, outputs=[LMOutput(text=text)])


# ---------------------------------------------------------------------------
# Instances and requests
# ---------------------------------------------------------------------------


def test_each_task_reads_its_own_rows(build) -> None:
    assert len(list(get_task("pdfa_pages_find").instances)) == 2
    assert len(list(get_task("pdfa_pages_order").instances)) == 1
    assert len(list(get_task("pdfa_pages_ocr").instances)) == 1
    assert list(get_task("pdfa_pages_transcribe").instances) == []


def test_request_carries_the_prompt_and_the_page_labels(build) -> None:
    task = get_task("pdfa_pages_find")
    instance = next(iter(task.instances))
    request = task.format_request(instance)
    assert request.messages == ({"role": "user", "content": ROWS[0]["prompt"]},)
    assert request.image_labels == ("Page 5:", "Page 6:", "Page 7:")
    assert instance.gold_answer == "Page 6"


def test_pages_are_rendered_lazily_from_the_documents_tar(build) -> None:
    task = get_task("pdfa_pages_find")
    instance = next(iter(task.instances))
    (loader,) = task.format_request(instance).images
    # Nothing is read while requests are built; the loader names the PDF's bytes and pages.
    assert loader.func is pdfa_pages._render_pages
    tar_path = str(build / "data" / "pdfa" / "pdfa-0007.tar")
    assert loader.args == (tar_path, 512, 2048, (4, 5, 6), 1536)


def test_single_page_request_has_no_labels(build) -> None:
    task = get_task("pdfa_pages_ocr")
    request = task.format_request(next(iter(task.instances)))
    assert request.image_labels is None


def test_find_metadata_marks_runs_and_quotes_that_give_the_answer_away(build) -> None:
    short, long = get_task("pdfa_pages_find").instances
    assert (short.metadata["page_number"], short.metadata["run"]) == (6, "short")
    assert short.metadata["quote"] == "leaky"
    # "2011" and "111" contain the digits of 11 but are other numbers.
    assert (long.metadata["page_number"], long.metadata["run"]) == (11, "long")
    assert long.metadata["quote"] == "clean"


def test_missing_build_names_the_variable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PDFA_PAGES_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="PDFA_PAGES_DIR"):
        list(get_task("pdfa_pages_find").instances)


def test_render_pages_reads_the_byte_range_and_keeps_the_page_order(tmp_path) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    for width, height in ((200, 100), (100, 400)):
        pdf.new_page(width, height)
    pdf_path = tmp_path / "doc.pdf"
    pdf.save(str(pdf_path))
    pdf.close()
    pdf_bytes = pdf_path.read_bytes()
    tar_path = tmp_path / "pdfa-0000.tar"
    tar_path.write_bytes(b"x" * 512 + pdf_bytes + b"y" * 512)

    second, first = pdfa_pages._render_pages(str(tar_path), 512, len(pdf_bytes), (1, 0), 64)
    assert (first.mode, first.size) == ("RGB", (64, 32))
    assert second.size == (16, 64)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, number", [("Page 42", 42), ("42", 42), (" page 7 of 9", 7), ("no idea", None)]
)
def test_parse_page_number(text: str, number: int | None) -> None:
    assert parse_page_number(text) == number


def test_parse_order_reads_the_first_line_with_numbers() -> None:
    assert parse_order("2, 3, 1") == [2, 3, 1]
    assert parse_order("Order:\n2 3 1\n4, 5") == [2, 3, 1]
    assert parse_order("unknown") == []


def test_pair_accuracy() -> None:
    assert pair_accuracy([2, 3, 1], [2, 3, 1]) == 1.0
    assert pair_accuracy([2, 3, 1], [1, 3, 2]) == 0.0
    # (2, 3) is in order; the two pairs with the missing 1 count as wrong.
    assert pair_accuracy([2, 3, 1], [2, 3]) == pytest.approx(1 / 3)
    # A repeated item keeps its first place.
    assert pair_accuracy([1, 2], [1, 2, 1]) == 1.0
    assert pair_accuracy([1, 2], []) == 0.0


def test_transcription_ignores_markup_case_and_spacing() -> None:
    scores = transcription_scores("# Title\n\n**Body** text.", "title\nbody   text.")
    assert scores == {"word_similarity": 1.0, "word_f1": 1.0}


def test_transcription_scores_count_word_edits() -> None:
    scores = transcription_scores("one two three four", "one two four")
    assert scores["word_similarity"] == pytest.approx(0.75)
    assert scores["word_f1"] == pytest.approx(6 / 7)
    # Word order matters to the similarity but not to the bag of words.
    swapped = transcription_scores("one two", "two one")
    assert swapped == {"word_similarity": 0.0, "word_f1": 1.0}


def test_transcription_compares_only_the_first_words() -> None:
    scores = transcription_scores("a b c d", "a b x y z", max_words=2)
    assert scores == {"word_similarity": 1.0, "word_f1": 1.0}
    assert transcription_scores("a b", "") == {"word_similarity": 0.0, "word_f1": 0.0}


def test_scorers_store_their_result_on_the_output(build) -> None:
    find = next(iter(get_task("pdfa_pages_find").instances))
    output = LMOutput(text="Page 6")
    assert PageNumberScorer().score(find, output) == 1.0
    assert output.metadata[RESULT_KEY] == {"accuracy": 1.0, "chance": pytest.approx(1 / 3)}
    assert PageNumberScorer().score(find, LMOutput(text="Page 7")) == 0.0

    order = next(iter(get_task("pdfa_pages_order").instances))
    output = LMOutput(text="2, 1, 3")
    assert PageOrderScorer().score(order, output) == 0.0
    # Images 2 and 3 and images 2 and 1 are in order; 3 and 1 are swapped.
    assert output.metadata[RESULT_KEY] == {"accuracy": 0.0, "pairs": pytest.approx(2 / 3)}
    assert PageOrderScorer().score(order, LMOutput(text="2, 3, 1")) == 1.0

    ocr = next(iter(get_task("pdfa_pages_ocr").instances))
    assert TranscriptionScorer().score(ocr, LMOutput(text="Title\n\nBody text.")) == 1.0


def test_metrics_average_over_their_subset_and_count_unscored_examples_as_zero(build) -> None:
    task = get_task("pdfa_pages_find")
    short, long = task.instances
    responses = [_response(short, "Page 6"), _response(long, "Page 3")]
    task._apply_scorers(responses)
    values = {metric.name: metric.compute(responses) for metric in task.config.metrics}
    assert values == {
        "accuracy": 0.5,
        "short_accuracy": 1.0,
        "long_accuracy": 0.0,
        "clean_quote_accuracy": 0.0,
        "chance_accuracy": pytest.approx((1 / 3 + 1 / 2) / 2),
    }

    metric = next(m for m in task.config.metrics if m.name == "short_accuracy")
    assert metric.compute_instance(responses[0]) == 1.0
    assert metric.compute_instance(responses[1]) is None
    assert metric.supports_pairwise_scorer_fallback() is False
    unscored = [_response(short, "Page 6"), responses[0]]
    assert metric.compute(unscored) == 0.5
