"""Task framework for evaluation."""

from .base import Task, TaskConfig
from .registry import clear_registry, get_task, list_tasks, register

__all__ = [
    "Task",
    "TaskConfig",
    "register",
    "get_task",
    "list_tasks",
    "clear_registry",
]

# Import task modules to trigger registration (must be after exports to avoid circular import)
from . import agi_eval as _agi_eval  # noqa: F401, E402
from . import aime as _aime  # noqa: F401, E402
from . import arc as _arc  # noqa: F401, E402
from . import basic_skills as _basic_skills  # noqa: F401, E402
from . import bbh as _bbh  # noqa: F401, E402
from . import code as _code  # noqa: F401, E402
from . import core_tasks as _core_tasks  # noqa: F401, E402
from . import deepmind_math as _deepmind_math  # noqa: F401, E402
from . import fim as _fim  # noqa: F401, E402
from . import gpqa as _gpqa  # noqa: F401, E402
from . import gsm as _gsm  # noqa: F401, E402
from . import hellaswag as _hellaswag  # noqa: F401, E402
from . import medmcqa as _medmcqa  # noqa: F401, E402
from . import minerva as _minerva  # noqa: F401, E402
from . import mmlu as _mmlu  # noqa: F401, E402
from . import qa as _qa  # noqa: F401, E402
from . import wikitext as _wikitext  # noqa: F401, E402
