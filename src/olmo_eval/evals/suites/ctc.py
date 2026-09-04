"""Suites for the CTC long-context family: one flag instead of 22 (or 187).

Naming:

* ``ctc:figure``   -- every task at its 2k-32k ladder (the paper grid; 108 runs)
* ``ctc:xlong``    -- every task's rungs ABOVE 32k (64k-1M where built; 62 runs)
* ``ctc:r32k`` ... -- every task that has that rung, at that rung (one context-length column)
* ``ctc:nq`` ...   -- one task's full ladder, every rung (one row)
* ``ctc``          -- everything (170 runs; know what you are asking for)

Aggregation is DISPLAY_ONLY throughout: the tasks carry heterogeneous metrics (f1, pair f1,
kendall tau, ce_pos_recall, partial credit), and averaging those into one number would be
exactly the kind of quiet nonsense the per-task metric declarations exist to prevent.
"""

from __future__ import annotations

from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register
from olmo_eval.evals.tasks.ctc_suite import ROSTER, RUNG_TOKENS

_DISPLAY = AggregationStrategy.DISPLAY_ONLY

_BASE = tuple(r for r in RUNG_TOKENS if RUNG_TOKENS[r] <= 32768)
_XLONG = tuple(r for r in RUNG_TOKENS if RUNG_TOKENS[r] > 32768)

_FIGURE = register(
    Suite(
        name="ctc:figure",
        tasks=tuple(
            f"{name}:{rung}" for name, row in ROSTER.items() for rung in row.rungs if rung in _BASE
        ),
        aggregation=_DISPLAY,
        description="All 22 CTC tasks over the 2k-32k figure ladder.",
    )
)

_XLONG_SUITE = register(
    Suite(
        name="ctc:xlong",
        tasks=tuple(
            f"{name}:{rung}" for name, row in ROSTER.items() for rung in row.rungs if rung in _XLONG
        ),
        aggregation=_DISPLAY,
        description="Every task's rungs above 32k (64k-1M where the source corpus allows).",
    )
)

for _rung in RUNG_TOKENS:
    register(
        Suite(
            name=f"ctc:{_rung}",
            tasks=tuple(f"{name}:{_rung}" for name, row in ROSTER.items() if _rung in row.rungs),
            aggregation=_DISPLAY,
            description=f"Every CTC task that has a {_rung} rung, at {_rung}.",
        )
    )

for _name, _row in ROSTER.items():
    register(
        Suite(
            name=f"ctc:{_name.removeprefix('ctc_')}",
            tasks=tuple(f"{_name}:{rung}" for rung in _row.rungs),
            aggregation=_DISPLAY,
            description=f"The full {_name} ladder ({len(_row.rungs)} rungs).",
        )
    )

register(
    Suite(
        name="ctc",
        tasks=(_FIGURE, _XLONG_SUITE),  # Suite objects: string entries would be read as task names
        aggregation=_DISPLAY,
        description="The entire CTC suite: 22 tasks x every built rung.",
    )
)
