"""Harness abstraction for configuring model capabilities.

The Harness is the primary abstraction for "a model provider configured with specific capabilities".
It owns the model runtime configuration (tools, system prompt, execution behavior)
and provides both single-turn and multi-turn interfaces.

Key Components:
- Tool: Unified schema + implementation for tools
- ProviderConfig: Configuration for creating an InferenceProvider
- HarnessConfig: Immutable configuration describing model capabilities
- Harness: Wraps a provider with config, provides generate() and run()
- Backend: Pluggable execution backends

Example:
    from olmo_eval.harness import (
        Harness,
        HarnessConfig,
        ProviderConfig,
        Tool,
        tool,
        get_harness_preset,
    )

    # Define a custom tool
    @tool(description="Search the web for information")
    async def web_search(query: str) -> str:
        return await search_api(query)

    # Create harness with provider config
    config = HarnessConfig(
        name="search_agent",
        provider=ProviderConfig(
            kind="vllm",
            model="llama3.1-8b",
        ),
        tool_names=("web_search",),
        system_prompt="You have access to search tools.",
        max_turns=10,
    )
    harness = Harness(config)

    # Single-turn generation with tools
    outputs = harness.generate([request])

    # Multi-turn agent execution
    result = await harness.run(request)
    print(result.trajectory)

    # Or use a preset
    config = get_harness_preset("search")
    harness = Harness(config)
"""

from .backends import (
    BACKEND_REGISTRY,
    Backend,
    OpenAIAgentsBackend,
    get_backend,
    get_backend_extras,
    list_backends,
    register_backend,
)
from .config import HarnessConfig, ProviderConfig, harness_config
from .harness import Harness, create_harness
from .presets import (
    get_harness_preset,
    list_harness_presets,
    register_harness_preset,
)
from .result import HarnessResult
from .tools import (
    TOOL_REGISTRY,
    Tool,
    clear_registry,
    ensure_tools_registered,
    get_tool,
    get_tools,
    list_tools,
    register_tool,
    registered_tool,
    tool,
)

__all__ = [
    # Main classes
    "Harness",
    "HarnessConfig",
    "HarnessResult",
    "ProviderConfig",
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
    "Backend",
    "OpenAIAgentsBackend",
    "BACKEND_REGISTRY",
    "get_backend",
    "get_backend_extras",
    "list_backends",
    "register_backend",
    # Presets
    "get_harness_preset",
    "list_harness_presets",
    "register_harness_preset",
]
