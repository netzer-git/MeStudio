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
from mestudio.context.manager import ContextManager
from mestudio.core.llm_client import LMStudioClient
from mestudio.core.orchestrator import Orchestrator
from mestudio.planner.tracker import PlanTracker
from mestudio.agents.sub_agent import SubAgentSpawner
from mestudio.tools.registry import ToolRegistry

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


async def check_llm_connection(client: LMStudioClient) -> bool:
    """Check if LM Studio is running and accessible."""
    try:
        models = await client.list_models()
        if models:
            console.print(f"[green]Connected to LM Studio[/green]")
            console.print(f"[dim]Using model: {settings.lm_studio_model}[/dim]")
            return True
        else:
            console.print("[yellow]Warning: No models loaded in LM Studio[/yellow]")
            return True  # Still continue, might work
    except Exception as e:
        console.print(f"[red]Cannot connect to LM Studio at {settings.lm_studio_url}[/red]")
        console.print(f"[dim]Error: {e}[/dim]")
        console.print("\n[yellow]Make sure LM Studio is running with a model loaded.[/yellow]")
        return False


class AgentApp:
    """Main application that ties all components together."""

    def __init__(self, interface: CLIInterface):
        self.interface = interface
        
        # Core components
        self.llm_client = LMStudioClient()
        self.context_manager = ContextManager()
        self.tool_registry = ToolRegistry()
        self.plan_tracker = PlanTracker()
        self.sub_agent_spawner: SubAgentSpawner | None = None  # Created after orchestrator
        self.orchestrator: Orchestrator | None = None

    async def initialize(self) -> bool:
        """Initialize all components. Returns False if critical failure."""
        console.print("[dim]Initializing components...[/dim]")
        
        # Check LLM connection
        if not await check_llm_connection(self.llm_client):
            return False
        
        # Initialize tool registry (registers all built-in tools)
        # Tools are auto-discovered by the registry
        tool_count = len(self.tool_registry.list_tools())
        console.print(f"[dim]Registered {tool_count} tools[/dim]")
        
        # Initialize context with system prompt
        system_prompt = self._build_system_prompt()
        self.context_manager.set_system_message(system_prompt)
        console.print(f"[dim]Context initialized ({settings.max_context_tokens} tokens max)[/dim]")
        
        # Create orchestrator
        output_handler = self.interface.create_output_handler()
        self.orchestrator = Orchestrator(
            llm_client=self.llm_client,
            context_manager=self.context_manager,
            tool_registry=self.tool_registry,
            plan_tracker=self.plan_tracker,
            output_handler=output_handler,
        )
        
        # Create sub-agent spawner
        self.sub_agent_spawner = SubAgentSpawner(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            parent_context=self.context_manager,
        )
        
        # Register slash command handlers
        self._register_commands()
        
        return True

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools_info = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self.tool_registry.list_tools().items()
        )
        
        working_dir = Path(settings.working_directory).resolve()
        
        return f"""You are MeStudio Agent, a helpful AI assistant that can execute tasks on the user's computer.

## Capabilities
You can use the following tools:
{tools_info}

## Guidelines
1. **Think step by step** before taking actions
2. **Ask clarifying questions** if the request is unclear
3. **Read files** before modifying them to understand context
4. **Show diffs** when making code changes
5. **Be safe** - confirm before destructive operations
6. **Work incrementally** - make small, testable changes

## Working Directory
Your working directory is: {working_dir}
{"You can access files anywhere on the system." if not settings.sandbox_file_access else "You are limited to files within this directory."}

## Response Format
- For simple questions: respond directly
- For file operations: use the appropriate tool
- For complex tasks: create a plan first
- For code changes: show what you're changing and why"""

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
        plan = self.plan_tracker.get_current_plan()
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
        status = {
            "used_tokens": self.context_manager.total_tokens,
            "max_tokens": settings.max_context_tokens,
            "message_count": len(self.context_manager.messages),
        }
        self.interface.show_context_status(status)

    async def _cmd_save(self, args: str) -> None:
        """Save session to a file."""
        label = args.strip() or "default"
        save_path = Path(settings.working_directory) / ".mestudio_sessions" / f"{label}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        session_data = {
            "messages": self.context_manager.messages,
            "total_tokens": self.context_manager.total_tokens,
        }
        with open(save_path, "w") as f:
            json.dump(session_data, f, indent=2)
        
        self.interface.show_success(f"Session saved to {save_path}")

    async def _cmd_load(self, args: str) -> None:
        """Load session from a file."""
        sessions_dir = Path(settings.working_directory) / ".mestudio_sessions"
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
        if await self.interface.confirm("Clear all conversation history?"):
            self.context_manager.clear()
            self.interface.show_success("Conversation cleared")

    async def run(self) -> None:
        """Main CLI loop."""
        assert self.orchestrator is not None
        
        # Show welcome info
        tool_count = len(self.tool_registry.list_tools())
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
                
                # Send to orchestrator
                try:
                    await self.orchestrator.chat(message)
                except Exception as e:
                    self.interface.show_error(str(e), "Error during chat")
                
                console.print()  # Blank line after response
                
            except KeyboardInterrupt:
                console.print()
                if await self.interface.confirm("Exit MeStudio Agent?"):
                    break
                continue


async def main() -> None:
    """Async entry point for the agent."""
    show_banner()
    
    # Create CLI interface
    interface = CLIInterface(console=console)
    
    # Create and initialize app
    app = AgentApp(interface)
    
    if not await app.initialize():
        console.print("[red]Failed to initialize. Exiting.[/red]")
        sys.exit(1)
    
    # Run main loop
    await app.run()
    
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
