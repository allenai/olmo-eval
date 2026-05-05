from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task


def test_naturalqs_olmo3base_uses_fixed_fewshot() -> None:
    task = get_task("naturalqs:gen:olmo3base")

    assert task.config.limit == 10_000
    assert task.config.fewshot_source == "naturalqs_fixed"
    assert task.config.num_fewshot == 5


def test_naturalqs_format_request_matches_open_qa_prompt() -> None:
    task = get_task("naturalqs:gen:olmo3base")
    instance = Instance(
        question="Question: who wrote frankenstein?\nAnswer:",
        gold_answer="Mary Shelley",
        metadata={"all_answers": ["Mary Shelley"], "answers": [("Mary Shelley",)]},
    )

    request = task.format_request(instance)

    assert request.request_type == RequestType.COMPLETION
    assert request.prompt.count("Question:") == 6
    assert "Question: who wrote frankenstein?\nAnswer:" in request.prompt


def test_naturalqs_extract_answer_strips_whitespace() -> None:
    task = get_task("naturalqs")

    assert task.extract_answer(LMOutput(text="  North \n")) == "North"
