from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "arc:mc:olmo3base",
    ("arc_easy:mc:olmo3base", "arc_challenge:mc:olmo3base"),
)
