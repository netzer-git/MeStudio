"""Rich + prompt_toolkit CLI interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mestudio.cli.renderers import (
    ContextStatusRenderer,
    DiffRenderer,
    ErrorRenderer,
    HelpRenderer,
    PlanRenderer,
    StreamingMarkdownRenderer,
    ToolCallRenderer,
)
from mestudio.cli.theme import (
    ASSISTANT_BORDER,
    ERROR_BORDER,
    USER_BORDER,
    USER_STYLE,
)


class CLIOutputHandler:
    """Output handler that renders to the CLI.
    
    Implements the OutputHandler protocol expected by Orchestrator.
    """

    def __init__(self, interface: "CLIInterface"):
        self.interface = interface
        self._markdown_renderer: StreamingMarkdownRenderer | None = None
        self._tool_renderer: ToolCallRenderer

    async def on_text_chunk(self, text: str) -> None:
        """Handle a streaming text chunk."""
        if self._markdown_renderer is None:
            self._markdown_renderer = StreamingMarkdownRenderer(self.interface.console)
            self._markdown_renderer.start()
        self._markdown_renderer.update(text)

    async def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Handle tool execution start."""
        # Finish any streaming text first
        self._finish_streaming()
        
        # Generate a simple ID for tracking
        call_id = f"{name}_{id(arguments)}"
        self.interface.tool_renderer.start(call_id, name, arguments)

    async def on_tool_result(self, name: str, result: str, success: bool) -> None:
        """Handle tool execution result."""
        call_id = f"{name}_{id(result)}"  # Won't match but that's ok
        # Just show result without trying to match ID
        self.interface.tool_renderer.finish("", name, result, success)

    async def on_compaction(self, level: Any) -> None:
        """Handle context compaction event."""
        self._finish_streaming()
        level_name = level.name if hasattr(level, "name") else str(level)
        self.interface.console.print(
            f"[yellow]Context compaction triggered: {level_name}[/yellow]"
        )

    async def on_error(self, error: str) -> None:
        """Handle error event."""
        self._finish_streaming()
        self.interface.error_renderer.render(error)

    def _finish_streaming(self) -> None:
        """Finish any in-progress markdown streaming."""
        if self._markdown_renderer is not None:
            self._markdown_renderer.finish()
            self._markdown_renderer = None

    def finish(self) -> None:
        """Clean up after a complete response."""
        self._finish_streaming()


class CLIInterface:
    """Main CLI interface using Rich and prompt_toolkit."""

    # Available slash commands
    COMMANDS = {
        "/help": "Show this help message",
        "/plan": "Show current plan",
        "/context": "Show context/token usage status",
        "/save [label]": "Save current session",
        "/load": "List and load a saved session",
        "/compact": "Force context compaction",
        "/clear": "Clear conversation (keep system prompt)",
        "/quit": "Exit the agent",
        "/exit": "Exit the agent",
    }

    def __init__(
        self,
        console: Console | None = None,
        history_file: str = ".mestudio_history",
    ):
        self.console = console or Console()
        
        # Set up prompt_toolkit session with history
        history_path = Path(history_file)
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            multiline=False,  # Single line by default, use Meta+Enter for multiline
            key_bindings=self._create_key_bindings(),
        )
        
        # Renderers
        self.markdown_renderer = StreamingMarkdownRenderer(self.console)
        self.tool_renderer = ToolCallRenderer(self.console)
        self.diff_renderer = DiffRenderer(self.console)
        self.plan_renderer = PlanRenderer(self.console)
        self.context_renderer = ContextStatusRenderer(self.console)
        self.error_renderer = ErrorRenderer(self.console)
        self.help_renderer = HelpRenderer(self.console)
        
        # Command handlers (set by main.py)
        self._command_handlers: dict[str, Callable[..., Coroutine[Any, Any, None]]] = {}

    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings."""
        bindings = KeyBindings()
        
        # Meta+Enter for newline (multi-line input)
        @bindings.add("escape", "enter")
        def _(event: Any) -> None:
            event.current_buffer.insert_text("\n")
        
        return bindings

    def create_output_handler(self) -> CLIOutputHandler:
        """Create an output handler for the orchestrator."""
        return CLIOutputHandler(self)

    def register_command(
        self,
        command: str,
        handler: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """Register a handler for a slash command."""
        self._command_handlers[command] = handler

    async def prompt_user(self) -> str:
        """Get input from the user."""
        try:
            # Run prompt_toolkit in executor to not block async
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.prompt(
                    [("class:prompt", "> ")],
                    style=None,
                ),
            )
            return user_input.strip()
        except EOFError:
            return "/quit"
        except KeyboardInterrupt:
            return ""

    async def handle_input(self, user_input: str) -> tuple[bool, str | None]:
        """Process user input, handling slash commands.
        
        Returns:
            Tuple of (should_continue, message_for_orchestrator).
            If message_for_orchestrator is None, the input was handled as a command.
        """
        if not user_input:
            return True, None
        
        # Check for slash commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command in ("/quit", "/exit"):
                return False, None
            
            if command == "/help":
                self.help_renderer.render_commands(self.COMMANDS)
                return True, None
            
            # Check for registered handlers
            if command in self._command_handlers:
                try:
                    await self._command_handlers[command](args)
                except Exception as e:
                    self.error_renderer.render(str(e), f"Command {command} failed")
                return True, None
            
            # Unknown command
            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
            self.console.print("[dim]Type /help for available commands[/dim]")
            return True, None
        
        # Regular message for orchestrator
        return True, user_input

    def show_user_message(self, message: str) -> None:
        """Display the user's message."""
        # Just show a simple indicator, not the full message
        # (user already typed it)
        pass

    def show_welcome(self, model: str, context_size: int, tool_count: int) -> None:
        """Display welcome information after startup."""
        self.help_renderer.render_welcome(model, context_size, tool_count)

    def show_plan(self, plan: dict[str, Any]) -> None:
        """Display a plan."""
        if not plan:
            self.console.print("[dim]No active plan[/dim]")
            return
        self.plan_renderer.render(plan)

    def show_context_status(self, status: dict[str, Any]) -> None:
        """Display context usage status."""
        self.context_renderer.render(status)

    def show_diff(self, diff: str, filename: str = "") -> None:
        """Display a unified diff."""
        self.diff_renderer.render(diff, filename)

    def show_error(self, error: str, title: str = "Error") -> None:
        """Display an error message."""
        self.error_renderer.render(error, title)

    def show_info(self, message: str) -> None:
        """Display an informational message."""
        self.console.print(f"[dim]{message}[/dim]")

    def show_success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f"[green]{message}[/green]")

    async def confirm(self, prompt: str) -> bool:
        """Ask for yes/no confirmation."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.prompt(
                    [("class:prompt", f"{prompt} (y/n): ")],
                ),
            )
            return response.strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print to console (convenience method)."""
        self.console.print(*args, **kwargs)
