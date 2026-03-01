"""MeStudio Agent — Entry point and CLI loop."""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mestudio import __version__

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


async def main() -> None:
    """Async entry point for the agent."""
    show_banner()
    console.print("[dim]MeStudio Agent starting...[/dim]")
    # Future steps will initialize LLM client, context manager,
    # tool registry, and the CLI loop here.


def cli_entry() -> None:
    """Synchronous entry point for console_scripts."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
        sys.exit(0)


if __name__ == "__main__":
    cli_entry()
