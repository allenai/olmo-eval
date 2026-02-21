from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "hellaswag",
    ("hellaswag:rc", "hellaswag:mc"),
)

make_suite(
    "hellaswag:olmo3base",
    ("hellaswag:rc:olmo3base", "hellaswag:mc:olmo3base"),
)
