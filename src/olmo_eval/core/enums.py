"""Type-safe enums for configuration constants."""

from enum import Enum


class Split(str, Enum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class MetricName(str, Enum):
    """Standard metric identifiers."""

    ACCURACY = "accuracy"
    ACC_PER_CHAR = "acc_per_char"
    ACC_PER_TOKEN = "acc_per_token"
    EXACT_MATCH = "exact_match"
    PASS_AT_1 = "pass_at_1"
    PASS_AT_K = "pass_at_k"
    F1 = "f1"
