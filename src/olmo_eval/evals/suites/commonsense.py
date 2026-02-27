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

make_suite(
    "csqa:xlarge",
    ("csqa:rc:xlarge", "csqa:mc:xlarge"),
)

make_suite(
    "csqa:olmes:full",
    ("csqa:rc:olmes:full", "csqa:mc:olmes:full"),
)

make_suite(
    "socialiqa",
    ("socialiqa:rc", "socialiqa:mc"),
)

make_suite(
    "socialiqa:olmo3base",
    ("socialiqa:rc:olmo3base", "socialiqa:mc:olmo3base"),
)

make_suite(
    "socialiqa:xlarge",
    ("socialiqa:rc:xlarge", "socialiqa:mc:xlarge"),
)

make_suite(
    "socialiqa:olmes:full",
    ("socialiqa:rc:olmes:full", "socialiqa:mc:olmes:full"),
)
