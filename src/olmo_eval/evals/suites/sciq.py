from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "sciq",
    ("sciq:rc", "sciq:mc"),
)

make_suite(
    "sciq:olmo3base",
    ("sciq:rc:olmo3base", "sciq:mc:olmo3base"),
)
