"""Markdown streaming, tool indicators, diffs, plan and context renderers."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.spinner import Spinner
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
    """Renders streaming markdown content with live updates."""

    def __init__(self, console: Console):
        self.console = console
        self._buffer = ""
        self._live: Live | None = None
        self._refresh_rate = 10  # fps

    def start(self) -> None:
        """Start the live display."""
        self._buffer = ""
        self._live = Live(
            Text("", style=DIM_STYLE),
            console=self.console,
            refresh_per_second=self._refresh_rate,
            transient=True,
        )
        self._live.start()

    def update(self, chunk: str) -> None:
        """Add a chunk of text and update display."""
        if self._live is None:
            return
        
        self._buffer += chunk
        
        # Try to render as markdown, fall back to plain text on error
        try:
            content = Markdown(self._buffer)
        except Exception:
            content = Text(self._buffer)
        
        self._live.update(content)

    def finish(self) -> str:
        """Stop the live display and return final content."""
        if self._live:
            self._live.stop()
            self._live = None
        
        final = self._buffer
        self._buffer = ""
        
        # Print final markdown
        if final.strip():
            try:
                self.console.print(Markdown(final))
            except Exception:
                self.console.print(final)
        
        return final


class ToolCallRenderer:
    """Renders tool calls with spinners and results."""

    def __init__(self, console: Console):
        self.console = console
        self._active_spinners: dict[str, Live] = {}

    def start(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        """Show a spinner for an executing tool."""
        icon = get_tool_icon(name)
        
        # Format arguments preview (truncated)
        args_preview = ", ".join(
            f"{k}={repr(v)[:30]}" for k, v in list(arguments.items())[:3]
        )
        if len(arguments) > 3:
            args_preview += ", ..."
        
        text = Text()
        text.append(f"{icon} ", style="bold blue")
        text.append(name, style="bold")
        if args_preview:
            text.append(f"({args_preview})", style="dim")
        
        spinner_text = Group(
            Spinner("dots", style="blue"),
            text,
        )
        
        live = Live(
            spinner_text,
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        live.start()
        self._active_spinners[call_id] = live

    def finish(
        self, call_id: str, name: str, result: str, success: bool
    ) -> None:
        """Replace spinner with result panel."""
        # Stop spinner
        if call_id in self._active_spinners:
            self._active_spinners[call_id].stop()
            del self._active_spinners[call_id]
        
        icon = get_tool_icon(name)
        status_icon = "OK" if success else "X"
        status_style = SUCCESS_STYLE if success else ERROR_STYLE
        
        # Truncate result preview
        preview = result[:200]
        if len(result) > 200:
            preview += "..."
        
        # Build result display
        header = Text()
        header.append(f"{icon} ", style="bold blue")
        header.append(name, style="bold")
        header.append(f" [{status_icon}]", style=status_style)
        
        self.console.print(header)
        if preview.strip():
            # Show preview indented
            for line in preview.split("\n")[:5]:
                self.console.print(f"   {line}", style="dim")
            if result.count("\n") > 5:
                self.console.print(f"   ... ({result.count(chr(10))} lines total)", style="dim")

    def cancel_all(self) -> None:
        """Cancel all active spinners."""
        for live in self._active_spinners.values():
            live.stop()
        self._active_spinners.clear()


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
