from olmo_eval.evals.suites.biology import _LAB_BENCH_TASKS
from olmo_eval.evals.suites.registry import make_suite
from olmo_eval.evals.tasks.basic_skills import BASIC_SKILLS_SUBTASKS

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
    "medmcqa:rc_mc:olmo3base",
    ("medmcqa:rc:olmo3base", "medmcqa:mc:olmo3base"),
)


make_suite(
    "medqa_en:rc_mc:olmo3base",
    ("medqa_en:rc:olmo3base", "medqa_en:mc:olmo3base"),
)


make_suite(
    "piqa:rc_mc:olmo3base",
    ("piqa:rc:olmo3base", "piqa:mc:olmo3base"),
)


make_suite(
    "csqa:rc_mc:olmo3base",
    ("csqa:rc:olmo3base", "csqa:mc:olmo3base"),
)


make_suite(
    "socialiqa:rc_mc:olmo3base",
    ("socialiqa:rc:olmo3base", "socialiqa:mc:olmo3base"),
)

make_suite(
    "coqa:gen_only:olmo3base",
    ("coqa:gen:olmo3base",),
)

make_suite(
    "hellaswag:rc_mc:olmo3base",
    ("hellaswag:rc:olmo3base", "hellaswag:mc:olmo3base"),
)


make_suite(
    "jeopardy:gen_only:olmo3base",
    ("jeopardy:gen:olmo3base",),
)


make_suite(
    "qasper_yesno:rc_only:olmo3base",
    ("qasper_yesno:rc:olmo3base",),
)


make_suite(
    "sciq:rc_mc:olmo3base",
    ("sciq:rc:olmo3base", "sciq:mc:olmo3base"),
)


make_suite(
    "sciriff_yesno:rc_only:olmo3base",
    ("sciriff_yesno:rc:olmo3base",),
)


make_suite(
    "squad:rc_mc:olmo3base",
    ("squad:mc:olmo3base", "squad:rc:olmo3base"),
)


make_suite(
    "winogrande:rc_mc:olmo3base",
    ("winogrande:rc:olmo3base", "winogrande:mc:olmo3base"),
)


make_suite(
    "naturalqs:olmo3base",
    ("naturalqs:mc:olmo3base", "naturalqs:rc:olmo3base"),
)


make_suite(
    "basic_skills:rc:olmo3base",
    tuple(f"basic_skills_{s}:rc::olmo3base" for s in BASIC_SKILLS_SUBTASKS),
)
