"""Literal type definitions for constrained string values."""

from typing import Literal

ProviderLiteral = Literal["vllm", "hf", "mock", "litellm"]
DtypeLiteral = Literal["auto", "float16", "bfloat16", "float32"]
PriorityLiteral = Literal["low", "normal", "high", "urgent"]
LoadFormatLiteral = Literal[
    "auto", "pt", "safetensors", "runai_streamer", "tensorizer", "bitsandbytes"
]
