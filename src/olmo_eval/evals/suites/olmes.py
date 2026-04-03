from olmo_eval.evals.suites.registry import make_suite
from olmo_eval.evals.tasks.basic_skills import BASIC_SKILLS_SUBTASKS

make_suite(
    "arc:rc:olmes:full",
    ("arc_easy:rc:olmes:full", "arc_challenge:rc:olmes:full"),
)

make_suite(
    "basic_skills:rc::olmes",
    tuple(f"basic_skills_{s}:rc::olmes" for s in BASIC_SKILLS_SUBTASKS),
)
