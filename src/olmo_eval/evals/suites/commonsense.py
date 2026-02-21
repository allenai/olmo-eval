from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "piqa",
    ("piqa:rc", "piqa:mc"),
)

make_suite(
    "piqa:olmo3base",
    ("piqa:rc:olmo3base", "piqa:mc:olmo3base"),
)

make_suite(
    "csqa",
    ("csqa:rc", "csqa:mc"),
)

make_suite(
    "csqa:olmo3base",
    ("csqa:rc:olmo3base", "csqa:mc:olmo3base"),
)
