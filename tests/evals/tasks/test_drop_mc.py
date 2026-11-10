from olmo_eval.evals.tasks.common import get_task


def test_drop_mc_v3_fewshot_examples_are_multiple_choice() -> None:
    task = get_task("drop:mc:olmo3base:v3")
    shots = task.get_fewshot()

    assert len(shots) == 5
    assert all(shot.metadata["passage"] and shot.choices for shot in shots)
    assert all(shot.metadata["mc_answer"] in "ABCD" for shot in shots)
