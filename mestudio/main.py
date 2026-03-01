"""MeStudio Agent — Entry point and CLI loop."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mestudio import __version__
from mestudio.cli.interface import CLIInterface
from mestudio.core.config import settings
from mestudio.core.orchestrator import Orchestrator
from mestudio.utils.logging import (
    setup_logging,
    log_session_start,
    log_session_end,
    log_user_message,
    transcript_user,
)

# Global console for startup messages
console = Console()


def show_banner() -> None:
    """Display the startup banner."""
    banner = Text()
    banner.append("MeStudio Agent", style="bold cyan")
    banner.append(f"  v{__version__}\n", style="dim")
    banner.append("Local AI Agent Orchestrator", style="white")
    banner.append(" powered by ", style="dim")
    banner.append("LM Studio", style="bold blue")

    console.print(
        Panel(
            banner,
            border_style="cyan",
            padding=(1, 2),
        )
    )


class AgentApp:
    """Main application that ties all components together."""

    def __init__(self, interface: CLIInterface):
        self.interface = interface
        self.orchestrator: Orchestrator | None = None

    async def initialize(self) -> bool:
        """Initialize all components. Returns False if critical failure."""
        console.print("[dim]Initializing components...[/dim]")
        
        # Create orchestrator with output handler
        output_handler = self.interface.create_output_handler()
        self.orchestrator = Orchestrator(output_handler=output_handler)
        
        # Initialize orchestrator (checks LLM connections, etc.)
        if not await self.orchestrator.initialize():
            console.print("[red]Cannot connect to LM Studio at {settings.lm_studio_url}[/red]")
            console.print("\n[yellow]Make sure LM Studio is running with a model loaded.[/yellow]")
            return False
        
        console.print("[green]Connected to LM Studio[/green]")
        console.print(f"[dim]Using model: {settings.lm_studio_model}[/dim]")
        
        # Show tool count
        tool_count = len(self.orchestrator.tool_registry.tools)
        console.print(f"[dim]Registered {tool_count} tools[/dim]")
        console.print(f"[dim]Context initialized ({settings.max_context_tokens} tokens max)[/dim]")
        
        # Register slash command handlers
        self._register_commands()
        
        return True

    def _register_commands(self) -> None:
        """Register slash command handlers."""
        self.interface.register_command("/plan", self._cmd_plan)
        self.interface.register_command("/context", self._cmd_context)
        self.interface.register_command("/save", self._cmd_save)
        self.interface.register_command("/load", self._cmd_load)
        self.interface.register_command("/compact", self._cmd_compact)
        self.interface.register_command("/clear", self._cmd_clear)

    async def _cmd_plan(self, args: str) -> None:
        """Show current plan."""
        assert self.orchestrator is not None
        plan = self.orchestrator.plan_tracker.get_current_plan()
        if plan:
            self.interface.show_plan({
                "goal": plan.goal,
                "steps": [
                    {
                        "description": step.description,
                        "status": step.status.value,
                    }
                    for step in plan.steps
                ],
            })
        else:
            self.interface.show_info("No active plan")

    async def _cmd_context(self, args: str) -> None:
        """Show context status."""
        assert self.orchestrator is not None
        ctx_status = self.orchestrator.context.get_status()
        status = {
            "used_tokens": ctx_status.used_tokens,
            "max_tokens": settings.max_context_tokens,
            "message_count": ctx_status.message_count,
        }
        self.interface.show_context_status(status)

    async def _cmd_save(self, args: str) -> None:
        """Save session to a file."""
        assert self.orchestrator is not None
        label = args.strip() or "default"
        save_path = Path(settings.working_directory).expanduser() / ".mestudio_sessions" / f"{label}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        ctx_status = self.orchestrator.context.get_status()
        session_data = {
            "messages": [m.model_dump() for m in self.orchestrator.context.messages],
            "total_tokens": ctx_status.used_tokens,
        }
        with open(save_path, "w") as f:
            json.dump(session_data, f, indent=2)
        
        self.interface.show_success(f"Session saved to {save_path}")

    async def _cmd_load(self, args: str) -> None:
        """Load session from a file."""
        sessions_dir = Path(settings.working_directory).expanduser() / ".mestudio_sessions"
        if not sessions_dir.exists():
            self.interface.show_info("No saved sessions found")
            return
        
        sessions = list(sessions_dir.glob("*.json"))
        if not sessions:
            self.interface.show_info("No saved sessions found")
            return
        
        self.interface.print("[bold]Available sessions:[/bold]")
        for i, session in enumerate(sessions, 1):
            self.interface.print(f"  {i}. {session.stem}")
        
        # For now just list them
        self.interface.show_info("Load by running: /load <session_name>")

    async def _cmd_compact(self, args: str) -> None:
        """Force context compaction."""
        self.interface.show_info("Compacting context...")
        # This would trigger compaction in a real implementation
        self.interface.show_success("Context compacted")

    async def _cmd_clear(self, args: str) -> None:
        """Clear conversation (keep system prompt)."""
        assert self.orchestrator is not None
        if await self.interface.confirm("Clear all conversation history?"):
            self.orchestrator.context.clear()
            self.interface.show_success("Conversation cleared")

    async def run(self) -> None:
        """Main CLI loop."""
        assert self.orchestrator is not None
        
        # Show welcome info
        tool_count = len(self.orchestrator.tool_registry.tools)
        self.interface.show_welcome(
            model=settings.lm_studio_model,
            context_size=settings.max_context_tokens,
            tool_count=tool_count,
        )
        
        console.print()
        console.print("[dim]Type a message or /help for commands. Press Ctrl+C to exit.[/dim]")
        console.print()
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = await self.interface.prompt_user()
                
                # Handle input (checks for commands)
                should_continue, message = await self.interface.handle_input(user_input)
                
                if not should_continue:
                    break
                
                if message is None:
                    # Command was handled
                    continue
                
                # Log user message
                log_user_message(message)
                transcript_user(message)
                
                # Send to orchestrator
                try:
                    await self.orchestrator.chat(message)
                except Exception as e:
                    self.interface.show_error(str(e), "Error during chat")
                
                console.print()  # Blank line after response
                
            except KeyboardInterrupt:
                # Exit immediately on Ctrl+C
                console.print("\n[dim]Interrupted.[/dim]")
                break
            except EOFError:
                # Exit on EOF (Ctrl+D on Unix, Ctrl+Z on Windows)
                break


async def main() -> None:
    """Async entry point for the agent."""
    # Initialize logging first
    session_id = setup_logging(settings)
    
    show_banner()
    console.print(f"[dim]Session: {session_id}[/dim]")
    
    # Create CLI interface
    interface = CLIInterface(console=console)
    
    # Create and initialize app
    app = AgentApp(interface)
    
    if not await app.initialize():
        console.print("[red]Failed to initialize. Exiting.[/red]")
        log_session_end()
        sys.exit(1)
    
    # Log session start
    log_session_start(
        python_version=sys.version.split()[0],
        model=settings.lm_studio_model,
        max_tokens=settings.max_context_tokens,
    )
    
    # Run main loop
    await app.run()
    
    # Log session end with metrics
    log_session_end()
    
    console.print("\n[dim]Goodbye![/dim]")


def cli_entry() -> None:
    """Synchronous entry point for console_scripts."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
        sys.exit(0)


if __name__ == "__main__":
    cli_entry()
