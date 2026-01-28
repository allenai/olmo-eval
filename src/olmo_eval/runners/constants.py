"""Shared constants and exceptions for evaluation runners."""

from olmo_eval.core.types import SamplingParams
from olmo_eval.evals.tasks.core.base import TaskConfig


class ValidationError(Exception):
    """Raised when validation of runner inputs fails."""

    pass


# Keys derived from the dataclass OVERRIDE_KEYS
TASKCONFIG_KEYS = TaskConfig.OVERRIDE_KEYS
SAMPLING_KEYS = SamplingParams.OVERRIDE_KEYS
