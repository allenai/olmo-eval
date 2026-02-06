"""Harness abstraction for configuring model capabilities.

The Harness is the primary abstraction for "a model configured with specific capabilities".
It owns the model runtime configuration (tools, system prompt, execution behavior)
and provides both single-turn and multi-turn interfaces.

Key Components:
- Tool: Unified schema + implementation for tools
- HarnessConfig: Immutable configuration describing model capabilities
- Harness: Wraps a provider with config, provides generate() and run()
- AgentBackend: Pluggable execution backends (internal loop, OpenAI Agents SDK)

Example:
    from olmo_eval.core.harness import (
        Harness,
        HarnessConfig,
        Tool,
        tool,
        get_harness_preset,
    )
    from olmo_eval.inference import VLLMProvider

    # Define a custom tool
    @tool(description="Search the web for information")
    async def web_search(query: str) -> str:
        return await search_api(query)

    # Create harness with tools
    provider = VLLMProvider("llama3.1-8b")
    config = HarnessConfig(
        name="search_agent",
        tool_names=("web_search",),
        system_prompt="You have access to search tools.",
        max_turns=10,
    )
    harness = Harness(provider, config)

    # Single-turn generation with tools
    outputs = harness.generate([request])

    # Multi-turn agent execution
    result = await harness.run(request)
    print(result.trajectory)

    # Or use a preset
    config = get_harness_preset("search")
    harness = Harness(provider, config)
"""

from .backend import (
    BACKEND_REGISTRY,
    AgentBackend,
    InternalBackend,
    OpenAIAgentsBackend,
    get_backend,
    list_backends,
    register_backend,
)
from .config import HarnessConfig, harness_config
from .harness import Harness, create_harness
from .presets import (
    HARNESS_PRESETS,
    get_harness_preset,
    list_harness_presets,
    register_harness_preset,
)
from .registry import (
    TOOL_REGISTRY,
    clear_registry,
    ensure_tools_registered,
    get_tool,
    get_tools,
    list_tools,
    register_tool,
    registered_tool,
)
from .result import HarnessResult
from .tool import Tool, tool

__all__ = [
    # Main classes
    "Harness",
    "HarnessConfig",
    "HarnessResult",
    "Tool",
    # Factory functions
    "create_harness",
    "harness_config",
    # Tool decorators and registry
    "tool",
    "registered_tool",
    "register_tool",
    "get_tool",
    "get_tools",
    "list_tools",
    "clear_registry",
    "ensure_tools_registered",
    "TOOL_REGISTRY",
    # Backends
    "AgentBackend",
    "InternalBackend",
    "OpenAIAgentsBackend",
    "BACKEND_REGISTRY",
    "get_backend",
    "list_backends",
    "register_backend",
    # Presets
    "HARNESS_PRESETS",
    "get_harness_preset",
    "list_harness_presets",
    "register_harness_preset",
]
