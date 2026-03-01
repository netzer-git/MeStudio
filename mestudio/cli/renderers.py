"""Markdown streaming, tool indicators, diffs, plan and context renderers."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from mestudio.cli.theme import (
    ASSISTANT_BORDER,
    ASSISTANT_STYLE,
    DIM_STYLE,
    ERROR_BORDER,
    ERROR_STYLE,
    ICON_STEP_ACTIVE,
    ICON_STEP_DONE,
    ICON_STEP_FAILED,
    ICON_STEP_PENDING,
    ICON_STEP_SKIPPED,
    SUCCESS_STYLE,
    TOOL_BORDER,
    get_context_color,
    get_tool_icon,
)


class StreamingMarkdownRenderer:
    """Renders streaming markdown content.
    
    Simplified version that accumulates text and prints at the end
    to avoid terminal corruption from Live displays.
    """

    def __init__(self, console: Console):
        self.console = console
        self._buffer = ""
        self._started = False

    def start(self) -> None:
        """Start collecting text."""
        self._buffer = ""
        self._started = True
        # Print a simple indicator that response is coming
        self.console.print("[dim]...[/dim]", end="")

    def update(self, chunk: str) -> None:
        """Add a chunk of text."""
        if not self._started:
            return
        self._buffer += chunk
        # Print dots to show progress (simple approach)
        if len(self._buffer) % 100 == 0:
            self.console.print(".", end="")

    def finish(self) -> str:
        """Print the final content."""
        if not self._started:
            return ""
        
        self._started = False
        final = self._buffer
        self._buffer = ""
        
        # Clear the progress indicator line and print final content
        self.console.print()  # New line after dots
        
        if final.strip():
            try:
                self.console.print(Markdown(final))
            except Exception:
                self.console.print(final)
        
        return final


class ToolCallRenderer:
    """Renders tool calls with results."""

    def __init__(self, console: Console):
        self.console = console

    def start(self, name: str, arguments: dict[str, Any]) -> None:
        """Show tool starting."""
        icon = get_tool_icon(name)
        
        # Format arguments preview (truncated)
        args_preview = ", ".join(
            f"{k}={repr(v)[:30]}" for k, v in list(arguments.items())[:3]
        )
        if len(arguments) > 3:
            args_preview += ", ..."
        
        # Build display text
        display = Text()
        display.append(f"{icon} ", style="bold blue")
        display.append(name, style="bold")
        if args_preview:
            display.append(f"({args_preview})", style="dim")
        
        # Print the tool call line
        self.console.print(display)

    def finish(self, name: str, result: str, success: bool) -> None:
        """Show tool result."""
        status_icon = "OK" if success else "X"
        status_style = SUCCESS_STYLE if success else ERROR_STYLE
        
        # Truncate result preview
        preview = result[:300]
        if len(result) > 300:
            preview += "..."
        
        # Build result header
        header = Text()
        header.append(f"  -> ", style="dim")
        header.append(f"[{status_icon}]", style=status_style)
        
        self.console.print(header)
        
        # Show preview if meaningful
        if preview.strip() and len(preview.strip()) > 0:
            lines = preview.split("\n")[:5]
            for line in lines:
                if line.strip():
                    self.console.print(f"     {line[:80]}", style="dim")
            total_lines = result.count("\n") + 1
            if total_lines > 5:
                self.console.print(f"     ... ({total_lines} lines total)", style="dim")

    def cancel_all(self) -> None:
        """No-op for compatibility."""
        pass


class DiffRenderer:
    """Renders unified diffs with syntax highlighting."""

    def __init__(self, console: Console):
        self.console = console

    def render(self, diff: str, filename: str = "") -> None:
        """Display a unified diff."""
        if not diff.strip():
            return
        
        title = f"Changes to {filename}" if filename else "Changes"
        
        syntax = Syntax(
            diff,
            "diff",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )
        
        panel = Panel(
            syntax,
            title=title,
            border_style="yellow",
            padding=(0, 1),
        )
        self.console.print(panel)


class PlanRenderer:
    """Renders task plans as trees."""

    def __init__(self, console: Console):
        self.console = console

    def render(self, plan: dict[str, Any]) -> None:
        """Display a plan as a tree."""
        goal = plan.get("goal", "Unknown goal")
        steps = plan.get("steps", [])
        
        tree = Tree(f"[bold]{goal}[/bold]")
        
        for step in steps:
            status = step.get("status", "pending")
            desc = step.get("description", "")
            index = step.get("index", 0)
            notes = step.get("notes", "")
            
            # Get status icon
            icon = {
                "pending": ICON_STEP_PENDING,
                "active": ICON_STEP_ACTIVE,
                "done": ICON_STEP_DONE,
                "failed": ICON_STEP_FAILED,
                "skipped": ICON_STEP_SKIPPED,
            }.get(status, ICON_STEP_PENDING)
            
            # Style based on status
            style = {
                "pending": "dim",
                "active": "bold yellow",
                "done": "green",
                "failed": "red",
                "skipped": "dim italic",
            }.get(status, "")
            
            label = f"{icon} {index}. [{style}]{desc}[/{style}]"
            branch = tree.add(label)
            
            # Add notes if present
            if notes:
                branch.add(f"[dim italic]{notes}[/dim italic]")
            
            # Add sub-steps if present
            for sub in step.get("sub_steps", []):
                sub_status = sub.get("status", "pending")
                sub_icon = {
                    "pending": ICON_STEP_PENDING,
                    "active": ICON_STEP_ACTIVE,
                    "done": ICON_STEP_DONE,
                    "failed": ICON_STEP_FAILED,
                    "skipped": ICON_STEP_SKIPPED,
                }.get(sub_status, ICON_STEP_PENDING)
                branch.add(f"{sub_icon} {sub.get('description', '')}")
        
        panel = Panel(tree, title="Plan", border_style="cyan")
        self.console.print(panel)


class ContextStatusRenderer:
    """Renders context/token usage status."""

    def __init__(self, console: Console):
        self.console = console

    def render(self, status: dict[str, Any]) -> None:
        """Display context status with progress bar."""
        total = status.get("total_budget", 120000)
        used = status.get("used_tokens", 0)
        percent = status.get("percent_used", 0)
        level = status.get("compaction_level", "none")
        message_count = status.get("message_count", 0)
        
        # Determine color
        color = get_context_color(percent)
        
        # Level indicator
        level_display = {
            "none": "[green]OK[/green]",
            "soft": "[yellow]SOFT[/yellow]",
            "preemptive": "[yellow]PREEMPTIVE[/yellow]",
            "aggressive": "[orange1]AGGRESSIVE[/orange1]",
            "emergency": "[red]EMERGENCY[/red]",
        }.get(level.lower(), level)
        
        # Create progress bar
        progress = Progress(
            TextColumn("[bold]Context:[/bold]"),
            BarColumn(bar_width=30, complete_style=color),
            TextColumn(f"{percent:.0f}%"),
            TextColumn(f"({used:,}/{total:,})"),
            TextColumn(level_display),
            console=self.console,
            transient=False,
        )
        
        with progress:
            task = progress.add_task("", total=total, completed=used)
        
        self.console.print(f"[dim]Messages: {message_count}[/dim]")


class ErrorRenderer:
    """Renders error messages."""

    def __init__(self, console: Console):
        self.console = console

    def render(self, error: str, title: str = "Error") -> None:
        """Display an error message."""
        panel = Panel(
            Text(error, style=ERROR_STYLE),
            title=title,
            border_style=ERROR_BORDER,
            padding=(0, 1),
        )
        self.console.print(panel)


class HelpRenderer:
    """Renders help and command information."""

    def __init__(self, console: Console):
        self.console = console

    def render_commands(self, commands: dict[str, str]) -> None:
        """Display available commands."""
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        
        for cmd, desc in sorted(commands.items()):
            table.add_row(cmd, desc)
        
        self.console.print(table)

    def render_welcome(self, model: str, context_size: int, tool_count: int) -> None:
        """Display welcome information."""
        info = Table.grid(padding=(0, 2))
        info.add_column()
        info.add_column()
        
        info.add_row("[bold]Model:[/bold]", model)
        info.add_row("[bold]Context:[/bold]", f"{context_size:,} tokens")
        info.add_row("[bold]Tools:[/bold]", f"{tool_count} available")
        
        self.console.print(info)
        self.console.print()
        self.console.print("[dim]Type /help for commands, or start chatting![/dim]")
