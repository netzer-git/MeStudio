"""Core orchestration components."""

from mestudio.core.config import Settings, get_settings
from mestudio.core.llm_client import (
    LMStudioClient,
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
)
from mestudio.core.models import (
    Message,
    ToolCall,
    FunctionCall,
    LLMResponse,
    StreamChunk,
    TokenUsage,
    SessionMetadata,
)

# Lazy import orchestrator to avoid circular imports
def __getattr__(name):
    if name in (
        "Orchestrator",
        "OrchestratorConfig", 
        "TurnResult",
        "OutputHandler",
        "ConsoleOutputHandler",
        "NullOutputHandler",
        "ORCHESTRATOR_SYSTEM_PROMPT",
    ):
        from mestudio.core.orchestrator import (
            Orchestrator,
            OrchestratorConfig,
            TurnResult,
            OutputHandler,
            ConsoleOutputHandler,
            NullOutputHandler,
            ORCHESTRATOR_SYSTEM_PROMPT,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # LLM Client
    "LMStudioClient",
    "LLMClientError",
    "LLMConnectionError",
    "LLMTimeoutError",
    # Models
    "Message",
    "ToolCall",
    "FunctionCall",
    "LLMResponse",
    "StreamChunk",
    "TokenUsage",
    "SessionMetadata",
    # Orchestrator (lazy loaded)
    "Orchestrator",
    "OrchestratorConfig",
    "TurnResult",
    "OutputHandler",
    "ConsoleOutputHandler",
    "NullOutputHandler",
    "ORCHESTRATOR_SYSTEM_PROMPT",
]
