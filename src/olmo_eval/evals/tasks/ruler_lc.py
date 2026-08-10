"""RULER long-context (ruler-lc) tasks.

Long-context companion to ruler.py: same task types, scorers, and prompting
behavior, but reads pre-generated data from local disk (see ruler_lc_loader.py)
instead of the allenai/ruler_data HuggingFace release, and extends context
sizes beyond the original 131072-token cap. Data is generated with
https://github.com/jopetty/RULER via scripts/generate-data.sh.
"""

from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data.ruler_lc_loader import get_ruler_lc_data_root
from olmo_eval.data.ruler_lc_tasks import RULER_LC_TASKS
from olmo_eval.evals.tasks.common.registry import register
from olmo_eval.evals.tasks.ruler import RulerTask, make_ruler_task_class


class RulerLcTask(RulerTask):
    """RULER task variant backed by the local ruler-lc dataset.

    ruler-lc records always carry a preformatted ``input`` plus a trailing
    ``answer_prefix`` that must be concatenated with no separator, unlike
    ruler.py's chat/system-template fallback path for records without a
    preformatted ``input``.
    """

    _tasks_registry: dict[str, dict[str, Any]] = RULER_LC_TASKS
    _name_prefix: str = "ruler_lc_"

    def _get_data_root(self) -> str:
        return get_ruler_lc_data_root()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("input")
        if not isinstance(question, str) or not question:
            return None

        answer = doc.get("outputs")
        metadata: dict = {
            "id": doc.get("index", index),
            "task_type": self.task_type,
            "context_size": self.context_size,
            "tag": self.ruler_config["tag"],
        }
        if isinstance(answer, list):
            metadata["all_gold_answers"] = answer

        return Instance(
            question=f"{question}{doc.get('answer_prefix', '')}",
            gold_answer=answer,
            metadata=metadata,
        )


# Dynamically register all ruler-lc tasks
for _task_name, _task_config in RULER_LC_TASKS.items():
    _cls = make_ruler_task_class(
        _task_name, _task_config, base_cls=RulerLcTask, class_prefix="RulerLc"
    )
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"ruler_lc_{_task_name}")(_cls)
