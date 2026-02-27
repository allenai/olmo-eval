from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "qasper_yesno",
    ("qasper_yesno:rc",),
)

make_suite(
    "qasper_yesno:olmo3base",
    ("qasper_yesno:rc:olmo3base",),
)
