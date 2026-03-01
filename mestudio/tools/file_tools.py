"""File read, write, search, list tools."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from loguru import logger

from mestudio.core.config import get_settings
from mestudio.tools.registry import tool


def _get_working_dir() -> Path:
    """Get the configured working directory."""
    settings = get_settings()
    return Path(settings.working_directory).expanduser().resolve()


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the working directory.
    
    Args:
        path: Relative or absolute path.
    
    Returns:
        Resolved absolute path.
    
    Raises:
        ValueError: If path escapes working directory (when sandbox enabled).
    """
    settings = get_settings()
    working_dir = _get_working_dir()
    
    # Resolve the path (handle ~ for home directory)
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
    else:
        resolved = (working_dir / path_obj).resolve()
    
    # Security check: ensure within working directory (if sandbox enabled)
    if settings.sandbox_file_access:
        try:
            resolved.relative_to(working_dir)
        except ValueError:
            raise ValueError(
                f"Access denied: '{path}' is outside the working directory"
            )
    
    return resolved


def is_binary(path: Path) -> bool:
    """Check if a file appears to be binary.
    
    Sniffs the first 8192 bytes for null bytes.
    
    Args:
        path: Path to the file.
    
    Returns:
        True if file appears binary, False otherwise.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception:
        return False


def _format_with_line_numbers(content: str, start_line: int = 1) -> str:
    """Add line numbers to content.
    
    Args:
        content: File content.
        start_line: Starting line number.
    
    Returns:
        Content with line numbers prefixed.
    """
    lines = content.split("\n")
    width = len(str(start_line + len(lines) - 1))
    numbered = []
    for i, line in enumerate(lines, start=start_line):
        numbered.append(f"{i:>{width}}│ {line}")
    return "\n".join(numbered)


@tool(
    name="read_file",
    description="Read the contents of a file. Supports absolute paths (e.g., C:\\Users\\...) or relative paths.",
    max_result_tokens=16000,
)
async def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read file contents.
    
    Args:
        path: File path (absolute like C:\\Users\\... or relative to working directory).
        start_line: First line to read (1-indexed, inclusive). If omitted, reads from start.
        end_line: Last line to read (1-indexed, inclusive). If omitted, reads to end.
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    
    if not resolved.exists():
        return f"Error: File not found: '{path}'"
    
    if not resolved.is_file():
        return f"Error: '{path}' is not a file"
    
    # Check for binary
    if is_binary(resolved):
        return f"Error: '{path}' appears to be a binary file"
    
    # Check file size
    size = resolved.stat().st_size
    max_size = 1024 * 1024  # 1MB
    
    if size > max_size and start_line is None and end_line is None:
        return (
            f"Error: File '{path}' is too large ({size:,} bytes). "
            f"Specify start_line and end_line to read a portion."
        )
    
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading '{path}': {e}"
    
    total_lines = len(lines)
    
    # Apply line range
    start = 1 if start_line is None else max(1, start_line)
    end = total_lines if end_line is None else min(total_lines, end_line)
    
    if start > total_lines:
        return f"Error: start_line {start} exceeds file length ({total_lines} lines)"
    
    # Extract requested lines (1-indexed to 0-indexed)
    selected = lines[start - 1 : end]
    content = "".join(selected).rstrip("\n")
    
    # Format with line numbers
    result = _format_with_line_numbers(content, start)
    
    # Add metadata header
    header = f"File: {path} ({total_lines} lines total)"
    if start != 1 or end != total_lines:
        header += f"\nShowing lines {start}-{end}"
    
    return f"{header}\n\n{result}"


@tool(
    name="write_file",
    description="Write content to a file. Creates parent directories if needed. Supports absolute paths.",
)
async def write_file(path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        path: File path (absolute like C:\\Users\\... or relative to working directory).
        content: Content to write to the file.
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    
    # Check for binary
    if resolved.exists() and is_binary(resolved):
        return f"Error: Cannot write to binary file '{path}'"
    
    try:
        # Create parent directories
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        
        size = len(content.encode("utf-8"))
        logger.info(f"Wrote {size} bytes to {path}")
        return f"Written {size:,} bytes to {path}"
    
    except Exception as e:
        return f"Error writing '{path}': {e}"


@tool(
    name="edit_file",
    description="Apply search/replace edits to a file. Each 'old' string must match exactly once. Supports absolute paths.",
)
async def edit_file(path: str, edits: list[dict[str, str]]) -> str:
    """Apply search/replace edits to a file.
    
    Args:
        path: File path (absolute like C:\\Users\\... or relative to working directory).
        edits: List of edits, each with 'old' (text to find) and 'new' (replacement text).
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    
    if not resolved.exists():
        return f"Error: File not found: '{path}'"
    
    if is_binary(resolved):
        return f"Error: Cannot edit binary file '{path}'"
    
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception as e:
        return f"Error reading '{path}': {e}"
    
    content = original
    applied_edits = []
    
    for i, edit in enumerate(edits):
        old = edit.get("old", "")
        new = edit.get("new", "")
        
        if not old:
            return f"Error: Edit {i + 1} has empty 'old' field"
        
        # Count occurrences
        count = content.count(old)
        
        if count == 0:
            return f"Error: Edit {i + 1} - '{old[:50]}...' not found in file"
        
        if count > 1:
            return (
                f"Error: Edit {i + 1} - '{old[:50]}...' is ambiguous "
                f"(found {count} occurrences). Add more context to 'old'."
            )
        
        # Apply edit
        content = content.replace(old, new, 1)
        applied_edits.append(f"  {i + 1}. Replaced {len(old)} chars with {len(new)} chars")
    
    # Write back
    try:
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return f"Error writing '{path}': {e}"
    
    # Generate simple diff summary
    result = f"Edited {path}\n\nApplied {len(edits)} edit(s):\n"
    result += "\n".join(applied_edits)
    
    # Show line count change
    old_lines = original.count("\n")
    new_lines = content.count("\n")
    if old_lines != new_lines:
        result += f"\n\nLine count: {old_lines} → {new_lines} ({new_lines - old_lines:+d})"
    
    logger.info(f"Applied {len(edits)} edits to {path}")
    return result


@tool(
    name="list_directory",
    description="List contents of a directory in a tree format. Supports absolute paths (e.g., C:\\ or C:\\Users).",
)
async def list_directory(
    path: str = ".",
    recursive: bool = False,
    max_depth: int = 3,
) -> str:
    """List directory contents.
    
    Args:
        path: Directory path (absolute like C:\\Users or relative). Defaults to working directory.
        recursive: Whether to list subdirectories recursively.
        max_depth: Maximum depth for recursive listing (default: 3).
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    
    if not resolved.exists():
        return f"Error: Directory not found: '{path}'"
    
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory"
    
    def tree(dir_path: Path, prefix: str = "", depth: int = 0) -> list[str]:
        """Generate tree structure."""
        if depth > max_depth:
            return [f"{prefix}[...]"]
        
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return [f"{prefix}[permission denied]"]
        except OSError:
            return [f"{prefix}[access error]"]
        
        lines = []
        entries_list = list(entries)
        
        for i, entry in enumerate(entries_list):
            is_last = i == len(entries_list) - 1
            connector = "└── " if is_last else "├── "
            
            try:
                is_dir = entry.is_dir()
            except OSError:
                # Broken symlink or inaccessible
                lines.append(f"{prefix}{connector}{entry.name} [broken link]")
                continue
            
            if is_dir:
                lines.append(f"{prefix}{connector}{entry.name}/")
                if recursive and depth < max_depth:
                    extension = "    " if is_last else "│   "
                    lines.extend(tree(entry, prefix + extension, depth + 1))
            else:
                try:
                    size = entry.stat().st_size
                    lines.append(f"{prefix}{connector}{entry.name} ({size:,} bytes)")
                except OSError:
                    lines.append(f"{prefix}{connector}{entry.name} [inaccessible]")
        
        return lines
    
    header = f"Directory: {path}"
    tree_lines = tree(resolved)
    
    if not tree_lines:
        return f"{header}\n\n  (empty directory)"
    
    return f"{header}\n\n" + "\n".join(tree_lines)


@tool(
    name="search_files",
    description="Search for text pattern in files (grep-like). Returns matching lines. Supports absolute paths.",
)
async def search_files(
    query: str,
    path: str = ".",
    glob: str = "*",
    max_results: int = 50,
) -> str:
    """Search for text in files.
    
    Args:
        query: Text pattern to search for (supports regex).
        path: Directory to search in (absolute like C:\\Users or relative).
        glob: File pattern to match (e.g., '*.py', '*.txt').
        max_results: Maximum number of results to return.
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    
    if not resolved.exists():
        return f"Error: Path not found: '{path}'"
    
    # Compile regex
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    results = []
    files_searched = 0
    
    # Find matching files
    if resolved.is_file():
        files = [resolved]
    else:
        files = list(resolved.rglob(glob))
    
    for file_path in files:
        try:
            if not file_path.is_file():
                continue
        except OSError:
            # Broken symlink - skip
            continue
        
        if is_binary(file_path):
            continue
        
        files_searched += 1
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        rel_path = file_path.relative_to(resolved.parent if resolved.is_file() else resolved)
                        results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                        
                        if len(results) >= max_results:
                            break
        except Exception:
            continue
        
        if len(results) >= max_results:
            break
    
    if not results:
        return f"No matches found for '{query}' in {files_searched} files"
    
    header = f"Found {len(results)} match(es) for '{query}'"
    if len(results) == max_results:
        header += f" (limited to {max_results})"
    
    return header + "\n\n" + "\n".join(results)


@tool(
    name="find_files",
    description="Find files matching a glob pattern. Examples: '*.py', '**/*.json', '**/RPG*'. Supports absolute paths.",
)
async def find_files(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
) -> str:
    """Find files by glob pattern.
    
    Args:
        pattern: Glob pattern (e.g., '*.py' for py files, '**/*.json' recursive, '**/RPG*' folders with RPG).
        path: Directory to search in (absolute like C:\\Users or relative).
        max_results: Maximum number of results to return.
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    
    if not resolved.exists():
        return f"Error: Path not found: '{path}'"
    
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory"
    
    try:
        matches = list(resolved.rglob(pattern))[:max_results + 1]
    except Exception as e:
        return f"Error searching: {e}"
    
    # Filter to files only and get relative paths
    files = []
    for match in matches[:max_results]:
        try:
            if match.is_file():
                try:
                    rel = match.relative_to(resolved)
                    files.append(str(rel))
                except ValueError:
                    files.append(str(match))
        except OSError:
            # Broken symlink - skip
            continue
    
    if not files:
        return f"No files found matching '{pattern}'"
    
    header = f"Found {len(files)} file(s) matching '{pattern}'"
    if len(matches) > max_results:
        header += f" (showing first {max_results})"
    
    return header + "\n\n" + "\n".join(files)


@tool(
    name="get_environment_info",
    description="Get system environment info: home directory, common user folders (Documents, Desktop, etc.), and available drives.",
)
async def get_environment_info() -> str:
    """Get information about the system environment for file exploration.
    
    Returns OS type, home directory, common user folders, and available drives.
    """
    import platform
    import string
    
    info = [
        f"OS: {platform.system()} {platform.release()}",
        f"Home directory: {Path.home()}",
        f"Working directory: {Path.cwd()}",
        f"User: {os.environ.get('USERNAME') or os.environ.get('USER', 'unknown')}",
        "",
        "Common user folders:",
    ]
    
    # Common user folders
    home = Path.home()
    common_folders = ["Documents", "Desktop", "Downloads", "Pictures", "Videos", "Music"]
    for folder in common_folders:
        path = home / folder
        if path.exists():
            info.append(f"  {folder}: {path}")
    
    # Windows drives
    if platform.system() == "Windows":
        info.append("")
        info.append("Available drives:")
        drives = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
        info.append(f"  {', '.join(drives)}")
    
    return "\n".join(info)


@tool(
    name="list_drives",
    description="List available drives on Windows (C:\\, D:\\, etc.). Use this to discover where to search for files.",
)
async def list_drives() -> str:
    """List available drives on the system.
    
    On Windows, returns all mounted drive letters.
    On Linux/Mac, returns mount points.
    """
    import platform
    import string
    
    if platform.system() == "Windows":
        drives = []
        for letter in string.ascii_uppercase:
            drive_path = Path(f"{letter}:\\")
            if drive_path.exists():
                try:
                    # Get some info about the drive
                    total, used, free = 0, 0, 0
                    try:
                        import shutil
                        total, used, free = shutil.disk_usage(drive_path)
                        total_gb = total / (1024**3)
                        free_gb = free / (1024**3)
                        drives.append(f"{letter}:\\ - {total_gb:.1f} GB total, {free_gb:.1f} GB free")
                    except Exception:
                        drives.append(f"{letter}:\\")
                except Exception:
                    drives.append(f"{letter}:\\")
        
        if not drives:
            return "No drives found"
        
        return "Available drives:\n" + "\n".join(drives)
    else:
        # Linux/Mac - show mount points
        try:
            mounts = []
            with open("/proc/mounts", "r") as f:
                for line in f:
                    parts = line.split()
                    if parts[1].startswith("/home") or parts[1] == "/" or parts[1].startswith("/mnt"):
                        mounts.append(f"{parts[1]} ({parts[0]})")
            return "Mount points:\n" + "\n".join(mounts) if mounts else "/"
        except Exception:
            return f"Root: /\nHome: {Path.home()}"


# Export all tools for easy registration
__all__ = [
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_files",
    "find_files",
    "get_environment_info",
    "list_drives",
    "is_binary",
]
