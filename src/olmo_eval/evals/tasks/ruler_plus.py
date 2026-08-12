"""RULER-plus (ruler-plus) tasks.

Long-context companion to ruler.py: same task types, scorers, and prompting
behavior, but downloads pre-generated data from the allenai/ruler-plus
HuggingFace dataset repo (see ruler_plus_loader.py) instead of the
allenai/ruler_data HuggingFace release, and extends context sizes beyond the
original 131072-token cap. Data is generated with https://github.com/jopetty/RULER
via scripts/generate-data.sh.
"""

from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data.ruler_plus_loader import get_ruler_plus_data_root
from olmo_eval.data.ruler_plus_tasks import RULER_PLUS_TASKS
from olmo_eval.evals.tasks.common.registry import register
from olmo_eval.evals.tasks.ruler import RulerTask, make_ruler_task_class


class RulerPlusTask(RulerTask):
    """RULER task variant backed by the ruler-plus dataset.

    ruler-plus records always carry a preformatted ``input`` plus a trailing
    ``answer_prefix`` that must be concatenated with no separator, unlike
    ruler.py's chat/system-template fallback path for records without a
    preformatted ``input``.
    """

    _tasks_registry: dict[str, dict[str, Any]] = RULER_PLUS_TASKS
    _name_prefix: str = "ruler_plus_"

    def _get_data_root(self) -> str:
        return get_ruler_plus_data_root()

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


# Number of samples to draw per task/context-size condition.
_RULER_PLUS_LIMIT = 512

# Dynamically register all ruler-plus tasks
for _task_name, _task_config in RULER_PLUS_TASKS.items():
    _cls = make_ruler_task_class(
        _task_name,
        _task_config,
        base_cls=RulerPlusTask,
        class_prefix="RulerPlus",
        limit=_RULER_PLUS_LIMIT,
    )
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"ruler_plus_{_task_name}")(_cls)
