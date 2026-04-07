from olmo_eval.evals.suites.biology import _LAB_BENCH_TASKS
from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "lab_bench:olmo3base",
    tuple(f"{t}:olmo3base" for t in _LAB_BENCH_TASKS),
    description="LAB-Bench with RC cloze format (3-shot, logprob MC)",
)

make_suite(
    "arc:mc:olmo3base",
    ("arc_easy:mc:olmo3base", "arc_challenge:mc:olmo3base"),
)


make_suite(
    "medmcqa:olmo3base",
    ("medmcqa:rc:olmo3base", "medmcqa:mc:olmo3base"),
)


make_suite(
    "medqa_en:olmo3base",
    ("medqa_en:rc:olmo3base", "medqa_en:mc:olmo3base"),
)


make_suite(
    "piqa:olmo3base",
    ("piqa:rc:olmo3base", "piqa:mc:olmo3base"),
)


make_suite(
    "csqa:olmo3base",
    ("csqa:rc:olmo3base", "csqa:mc:olmo3base"),
)


make_suite(
    "socialiqa:olmo3base",
    ("socialiqa:rc:olmo3base", "socialiqa:mc:olmo3base"),
)

make_suite(
    "coqa:gen:olmo3base",
    ("coqa:gen:olmo3base",),
)


make_suite(
    "hellaswag:olmo3base",
    ("hellaswag:rc:olmo3base", "hellaswag:mc:olmo3base"),
)


make_suite(
    "jeopardy:gen:olmo3base",
    ("jeopardy:gen:olmo3base",),
)


make_suite(
    "qasper_yesno:olmo3base",
    ("qasper_yesno:rc:olmo3base",),
)


make_suite(
    "sciq:olmo3base",
    ("sciq:rc:olmo3base", "sciq:mc:olmo3base"),
)


make_suite(
    "sciriff_yesno:olmo3base",
    ("sciriff_yesno:rc:olmo3base",),
)
