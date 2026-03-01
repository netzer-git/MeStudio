"""Tool registration and dispatch system."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Coroutine, get_type_hints

from loguru import logger

from mestudio.context.token_counter import get_token_counter
from mestudio.utils.logging import log_tool_call, log_tool_registered


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """Definition of a registered tool."""

    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable[..., Coroutine[Any, Any, str]]
    max_result_tokens: int = 8000
    timeout: float = 30.0

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Singleton registry for all agent tools.
    
    Manages tool registration, schema generation, and dispatch.
    """

    _instance: ToolRegistry | None = None

    # Read-only tools whose results can be cached within a turn
    _CACHEABLE_TOOLS = {
        "get_environment_info", "list_drives", "list_directory",
        "find_files", "search_files", "read_file", "context_status",
        "get_plan", "list_sessions",
    }
    _CACHE_TTL_SECONDS = 300  # Cache results for 5 minutes

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._token_counter = get_token_counter()
            cls._instance._default_timeout = 30.0
            cls._instance._result_cache = {}  # {cache_key: (result, timestamp)}
            cls._instance._call_history = []  # Track recent calls for loop detection
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def clear_cache(self) -> None:
        """Clear the tool result cache."""
        self._result_cache.clear()

    def reset_call_history(self) -> None:
        """Reset call history for a new conversation/task."""
        self._call_history = []

    def _get_cache_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Generate a cache key for tool call."""
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]
        return f"{tool_name}:{args_hash}"

    def _get_cached_result(self, cache_key: str) -> str | None:
        """Get cached result if valid."""
        if cache_key in self._result_cache:
            result, timestamp = self._result_cache[cache_key]
            if time.time() - timestamp < self._CACHE_TTL_SECONDS:
                return result
            # Expired, remove it
            del self._result_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: str) -> None:
        """Cache a tool result."""
        self._result_cache[cache_key] = (result, time.time())
        # Limit cache size
        if len(self._result_cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(
                self._result_cache.keys(),
                key=lambda k: self._result_cache[k][1]
            )
            for key in sorted_keys[:20]:
                del self._result_cache[key]

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        """Get all registered tools."""
        return self._tools.copy()

    def register(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        handler: Callable[..., Coroutine[Any, Any, str]],
        max_result_tokens: int = 8000,
        timeout: float | None = None,
    ) -> None:
        """Register a tool.
        
        Args:
            name: Unique tool name.
            description: Description shown to the LLM.
            parameters: List of parameter definitions.
            handler: Async function to execute the tool.
            max_result_tokens: Max tokens for tool result (truncates if exceeded).
            timeout: Execution timeout in seconds (default: 30s).
        """
        if name in self._tools:
            logger.warning(f"Overwriting existing tool: {name}")

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            max_result_tokens=max_result_tokens,
            timeout=timeout or self._default_timeout,
        )
        log_tool_registered(name, description)
        logger.debug(f"Registered tool: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool.
        
        Args:
            name: Tool name to remove.
        
        Returns:
            True if removed, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Generate OpenAI-compatible tool definitions.
        
        Returns:
            List of tool schemas for the OpenAI API.
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str,
    ) -> str:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments (dict or JSON string).
        
        Returns:
            Tool result as string, or error message.
        """
        # Get tool
        tool = self._tools.get(tool_name)
        if not tool:
            error = f"Error: Unknown tool '{tool_name}'"
            logger.error(error)
            return error

        # Parse arguments if JSON string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                error = f"Error: Invalid JSON arguments for '{tool_name}': {e}"
                logger.error(error)
                return error

        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Arguments: {arguments}")
        
        # Strip invalid arguments (empty keys, unknown params)
        if isinstance(arguments, dict):
            # First remove empty string keys and None values
            arguments = {k: v for k, v in arguments.items() if k and k.strip()}
            
            # Then strip unknown arguments (models sometimes add extra params)
            if arguments:
                import inspect
                sig = inspect.signature(tool.handler)
                valid_params = set(sig.parameters.keys())
                filtered_args = {k: v for k, v in arguments.items() if k in valid_params}
                if len(filtered_args) < len(arguments):
                    removed = set(arguments.keys()) - set(filtered_args.keys())
                    logger.debug(f"Stripped unknown arguments: {removed}")
                arguments = filtered_args
        
        # Track calls for loop detection (before cache check)
        call_signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        self._call_history.append((call_signature, time.time()))
        # Keep only last 20 calls from the last 5 minutes
        cutoff = time.time() - 300
        self._call_history = [(sig, ts) for sig, ts in self._call_history[-20:] if ts > cutoff]
        
        # Detect repetitive patterns - warn if same call made 3+ times
        recent_identical = sum(1 for sig, _ in self._call_history if sig == call_signature)
        loop_warning = ""
        if recent_identical >= 3:
            loop_warning = f"\n\n[WARNING: This exact tool call has been made {recent_identical} times recently. Consider using the results you already have or trying a different approach.]"
        
        # Check cache for read-only tools
        cache_key = None
        if tool_name in self._CACHEABLE_TOOLS:
            cache_key = self._get_cache_key(tool_name, arguments)
            cached = self._get_cached_result(cache_key)
            if cached is not None:
                logger.debug(f"Tool {tool_name} returning cached result")
                return cached + "\n[cached]" + loop_warning
        
        start_time = time.perf_counter()
        args_keys = list(arguments.keys()) if isinstance(arguments, dict) else []

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.handler(**arguments),
                timeout=tool.timeout,
            )

            # Truncate if needed
            result = self.truncate_result(result, tool.max_result_tokens)

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_tool_call(
                tool_name=tool_name,
                args_keys=args_keys,
                duration_ms=duration_ms,
                success=True,
                result_length=len(result),
            )
            logger.debug(f"Tool {tool_name} completed: {len(result)} chars")
            
            # Cache successful results for cacheable tools
            if cache_key is not None:
                self._cache_result(cache_key, result)
            
            return result + loop_warning

        except asyncio.TimeoutError:
            error = f"Error: Tool '{tool_name}' timed out after {tool.timeout}s"
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_tool_call(
                tool_name=tool_name,
                args_keys=args_keys,
                duration_ms=duration_ms,
                success=False,
                result_length=0,
                error="timeout",
            )
            logger.error(error)
            return error

        except TypeError as e:
            # Missing/invalid arguments - provide helpful usage hint
            import inspect
            sig = inspect.signature(tool.handler)
            params = []
            for name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty:
                    params.append(f"{name} (required)")
                else:
                    params.append(f"{name}={param.default!r}")
            usage_hint = f"Expected: {', '.join(params)}" if params else ""
            
            error = f"Error: Invalid arguments for '{tool_name}': {e}"
            if usage_hint:
                error += f"\n{usage_hint}"
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_tool_call(
                tool_name=tool_name,
                args_keys=args_keys,
                duration_ms=duration_ms,
                success=False,
                result_length=0,
                error=str(e),
            )
            logger.error(error)
            return error

        except Exception as e:
            # Catch all other errors gracefully
            error = f"Error in '{tool_name}': {type(e).__name__}: {e}"
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_tool_call(
                tool_name=tool_name,
                args_keys=args_keys,
                duration_ms=duration_ms,
                success=False,
                result_length=0,
                error=str(e),
            )
            logger.error(error)
            return error

    def truncate_result(self, result: str, max_tokens: int) -> str:
        """Truncate a tool result to fit within token budget.
        
        Args:
            result: The tool result string.
            max_tokens: Maximum allowed tokens.
        
        Returns:
            Truncated result (if needed) with marker.
        """
        tokens = self._token_counter.count_tokens(result)
        if tokens <= max_tokens:
            return result

        return self._token_counter.truncate_to_tokens(result, max_tokens)


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return ToolRegistry()


# Type mapping for Python types to JSON schema types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(
    name: str,
    description: str,
    max_result_tokens: int = 8000,
    timeout: float | None = None,
) -> Callable:
    """Decorator to register a function as a tool.
    
    The function's docstring and type hints are used to generate
    parameter descriptions and schemas.
    
    Args:
        name: Tool name.
        description: Tool description for the LLM.
        max_result_tokens: Max tokens for result.
        timeout: Execution timeout.
    
    Example:
        @tool(name="read_file", description="Read contents of a file")
        async def read_file(
            path: str,
            start_line: int | None = None,
            end_line: int | None = None,
        ) -> str:
            '''
            Args:
                path: Path to the file to read.
                start_line: First line to read (1-indexed).
                end_line: Last line to read (1-indexed).
            '''
            ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, str]]) -> Callable:
        # Extract parameters from function signature
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        docstring = func.__doc__ or ""
        
        # Parse docstring for parameter descriptions
        param_docs = _parse_docstring_params(docstring)
        
        parameters: list[ToolParameter] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            
            # Get type
            hint = hints.get(param_name, str)
            # Handle Optional/Union types
            origin = getattr(hint, "__origin__", None)
            if origin is type(None) or str(hint).startswith("typing.Optional"):
                # Optional type - extract the inner type
                args = getattr(hint, "__args__", (str,))
                hint = args[0] if args and args[0] is not type(None) else str
            
            json_type = _TYPE_MAP.get(hint, "string")
            
            # Get description from docstring
            param_desc = param_docs.get(param_name, f"The {param_name} parameter")
            
            # Check if required
            required = param.default is inspect.Parameter.empty
            default = None if required else param.default
            
            parameters.append(ToolParameter(
                name=param_name,
                type=json_type,
                description=param_desc,
                required=required,
                default=default,
            ))
        
        # Register the tool
        registry = get_registry()
        registry.register(
            name=name,
            description=description,
            parameters=parameters,
            handler=func,
            max_result_tokens=max_result_tokens,
            timeout=timeout,
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        # Attach metadata
        wrapper._tool_name = name
        wrapper._tool_definition = registry.get(name)
        
        return wrapper
    
    return decorator


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Parse parameter descriptions from a Google-style docstring.
    
    Args:
        docstring: The function docstring.
    
    Returns:
        Dict mapping parameter names to descriptions.
    """
    params = {}
    
    if not docstring:
        return params
    
    lines = docstring.split("\n")
    in_args_section = False
    current_param = None
    current_desc_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check for Args section
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue
        
        # Check for end of Args section
        if in_args_section and stripped and not stripped.startswith(" ") and ":" in stripped:
            if stripped.lower().startswith(("returns:", "raises:", "yields:", "examples:")):
                in_args_section = False
                # Save last param
                if current_param:
                    params[current_param] = " ".join(current_desc_lines).strip()
                continue
        
        if in_args_section:
            # Check for new parameter
            if ": " in stripped and not stripped.startswith(" "):
                # Save previous param
                if current_param:
                    params[current_param] = " ".join(current_desc_lines).strip()
                
                # Parse new param
                parts = stripped.split(": ", 1)
                current_param = parts[0].split("(")[0].strip()
                current_desc_lines = [parts[1]] if len(parts) > 1 else []
            elif current_param and stripped:
                # Continuation of description
                current_desc_lines.append(stripped)
    
    # Save last param
    if current_param:
        params[current_param] = " ".join(current_desc_lines).strip()
    
    return params
