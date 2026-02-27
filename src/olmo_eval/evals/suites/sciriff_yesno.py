from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "sciriff_yesno",
    ("sciriff_yesno:rc",),
)

make_suite(
    "sciriff_yesno:olmo3base",
    ("sciriff_yesno:rc:olmo3base",),
)
