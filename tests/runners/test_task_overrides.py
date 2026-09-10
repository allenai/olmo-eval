from olmo_eval.runners.asynq.runner import AsyncEvalRunner


def test_nested_sampling_params_are_extracted_as_sampling_overrides() -> None:
    runner = AsyncEvalRunner(
        task_overrides={
            "task:variant": {
                "limit": 16,
                "sampling_params": {"max_tokens": 16384, "temperature": 0.6},
            }
        }
    )

    task_overrides, sampling_overrides = runner._build_task_overrides("task:variant")

    assert task_overrides == {"limit": 16}
    assert sampling_overrides == {"max_tokens": 16384, "temperature": 0.6}
