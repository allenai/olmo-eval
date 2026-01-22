from dataclasses import dataclass
from typing import Any

# =============================================================================
# Stop Sequences for Code Generation
# =============================================================================

HUMANEVAL_STOP_SEQUENCES: tuple[str, ...] = (
    "\nclass",
    "\nif",
    "\nprint",
    "\n#",
    "\n```",
    "\n```\n\n",
    "<|eot_id|>",
)
"""Stop sequences for HumanEval and similar code generation tasks."""

MBPP_STOP_SEQUENCES: tuple[str, ...] = (
    "\nclass",
    "\nassert",
    '\n"""',
    "\nprint",
    "\nif",
    "\n```",
    "\n#",
    "\n<|/",
    "<|eot_id|>",
)
"""Stop sequences for MBPP code generation tasks."""

# Generic code stop sequences for any Python code generation
CODE_STOP_SEQUENCES: tuple[str, ...] = (
    "\nclass",
    "\ndef",
    "\nif __name__",
    "\n#",
    "\n```",
    "<|eot_id|>",
    "<|endoftext|>",
    "<|im_end|>",
    "</s>",
)
"""Generic stop sequences for Python code generation."""


# =============================================================================
# FIM (Fill-in-the-Middle) Token Configurations
# =============================================================================


@dataclass(frozen=True, slots=True)
class FIMConfig:
    """Configuration for Fill-in-the-Middle code completion.

    Attributes:
        lead_token: Token marking the prefix context.
        center_token: Token marking the suffix context (hole to fill).
        end_token: Token marking the middle content (completion).
        stop_sequences: Sequences that signal generation should stop.
    """

    lead_token: str
    center_token: str
    end_token: str
    stop_sequences: tuple[str, ...]

    def to_context_kwargs(self) -> dict[str, str]:
        """Return context formatting kwargs for the formatter."""
        return {
            "lead_token": self.lead_token,
            "center_token": self.center_token,
            "end_token": self.end_token,
        }

    def to_generation_kwargs(self) -> dict[str, Any]:
        """Return generation kwargs including stop sequences."""
        return {"stop_sequences": list(self.stop_sequences)}


SANTACODER_FIM = FIMConfig(
    lead_token="<fim-prefix>",
    center_token="<fim-suffix>",
    end_token="<fim-middle>",
    stop_sequences=("<|eot_id|>", "<|endoftext|>", "<|filename|>", "<file_sep>"),
)
"""SantaCoder FIM token configuration."""

STARCODER_FIM = FIMConfig(
    lead_token="<fim_prefix>",
    center_token="<fim_suffix>",
    end_token="<fim_middle>",
    stop_sequences=("<|eot_id|>", "<|endoftext|>", "<|filename|>", "<file_sep>"),
)
"""StarCoder FIM token configuration."""

DEEPSEEK_CODER_FIM = FIMConfig(
    lead_token="<｜fim▁begin｜>",
    center_token="<｜fim▁hole｜>",
    end_token="<｜fim▁end｜>",
    stop_sequences=("<|eot_id|>", "<|endoftext|>", "<|EOT|>"),
)
"""DeepSeek Coder FIM token configuration."""

OLMO_FIM = FIMConfig(
    lead_token="<|fim_prefix|>",
    center_token="<|fim_suffix|>",
    end_token="<|fim_middle|>",
    stop_sequences=("<|endoftext|>", "<|filename|>", "<|file_sep|>"),
)
"""OLMo FIM token configuration."""


FIM_CONFIGS: dict[str, FIMConfig] = {
    "santacoder": SANTACODER_FIM,
    "starcoder": STARCODER_FIM,
    "deepseek": DEEPSEEK_CODER_FIM,
    "olmo": OLMO_FIM,
}
"""Mapping of model family names to their FIM configurations."""


# =============================================================================
# Code Generation Tasks
# =============================================================================

ALL_CODEX_TASKS: tuple[str, ...] = (
    "humaneval:temp0.8",
    "humanevalplus:temp0.8",
    "mbpp::none",
    "mbppplus::none",
    "bigcodebench::none",
    "bigcodebench_hard::none",
)
"""Standard code generation benchmark tasks."""

STARCODER_CODEX_TASKS: tuple[str, ...] = (
    "humaneval::starcoder_pass@1",
    "humaneval::starcoder_pass@10",
    "mbpp::starcoder_pass@1",
    "mbpp::starcoder_pass@10",
)
"""StarCoder-specific code generation tasks with pass@k metrics."""

STARCODER_PASS_AT_1_TASKS: tuple[str, ...] = (
    "humaneval::starcoder_pass@1",
    "mbpp::starcoder_pass@1",
)
"""StarCoder tasks evaluated with pass@1 metric only."""

FIM_TASKS: tuple[str, ...] = (
    "humanevalfim_single",
    "humanevalfim_multi",
    "humanevalfim_random",
)
"""Fill-in-the-middle code completion tasks."""

CRUX_EVAL_TASKS: tuple[str, ...] = (
    "cruxeval_input:pass@5",
    "cruxeval_output:pass@5",
)
"""CRUXEval code understanding tasks."""


# =============================================================================
# Multilingual Code Tasks
# =============================================================================

MULTILINGUAL_MBPP_TASKS: tuple[str, ...] = (
    "mt_mbpp_bash",
    "mt_mbpp_c",
    "mt_mbpp_cpp",
    "mt_mbpp_csharp",
    "mt_mbpp_go",
    "mt_mbpp_haskell",
    "mt_mbpp_java",
    "mt_mbpp_javascript",
    "mt_mbpp_matlab",
    "mt_mbpp_php",
    "mt_mbpp_python",
    "mt_mbpp_r",
    "mt_mbpp_ruby",
    "mt_mbpp_rust",
    "mt_mbpp_scala",
    "mt_mbpp_swift",
    "mt_mbpp_typescript",
)
"""Multilingual MBPP tasks across 17 programming languages."""

MULTILINGUAL_MBPP_TASKS_V2: tuple[str, ...] = (
    "mt_mbpp_v2fix_bash",
    "mt_mbpp_v2fix_c",
    "mt_mbpp_v2fix_cpp",
    "mt_mbpp_v2fix_csharp",
    "mt_mbpp_v2fix_go",
    "mt_mbpp_v2fix_haskell",
    "mt_mbpp_v2fix_java",
    "mt_mbpp_v2fix_javascript",
    "mt_mbpp_v2fix_matlab",
    "mt_mbpp_v2fix_php",
    "mt_mbpp_v2fix_python",
    "mt_mbpp_v2fix_r",
    "mt_mbpp_v2fix_ruby",
    "mt_mbpp_v2fix_rust",
    "mt_mbpp_v2fix_scala",
    "mt_mbpp_v2fix_swift",
    "mt_mbpp_v2fix_typescript",
)
"""Multilingual MBPP v2 tasks with bug fixes."""

MULTIPL_E_HE_TASKS: tuple[str, ...] = (
    "multipl_e_humaneval:cpp::olmo3",
    "multipl_e_humaneval:java::olmo3",
    "multipl_e_humaneval:js::olmo3",
    "multipl_e_humaneval:php::olmo3",
    "multipl_e_humaneval:rs::olmo3",
    "multipl_e_humaneval:sh::olmo3",
)
"""MultiPL-E HumanEval multilingual code generation tasks."""

MULTIPL_E_MBPP_TASKS: tuple[str, ...] = (
    "multipl_e_mbpp:cpp::olmo3",
    "multipl_e_mbpp:java::olmo3",
    "multipl_e_mbpp:js::olmo3",
    "multipl_e_mbpp:php::olmo3",
    "multipl_e_mbpp:rs::olmo3",
)
"""MultiPL-E MBPP multilingual code generation tasks."""
